# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:11:18 2024

@author: EPSla
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import json
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint, MultiPolygon
from shapely.ops import nearest_points, unary_union
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from statistics import mean
from math import ceil
import fiona
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.geometry import Polygon
from functions import *
from functions2 import *
import initialization 
import QGIS_processing_polygon

def poles_clustering_and_cleaning(buildings_filter, crs, chain_upper_bound, pole_upper_bound):
    def create_clusters(buildings_filter, max_distance):
        coordinates = [(point.x, point.y) for point in buildings_filter.geometry]
        kdtree = cKDTree(coordinates)
        assigned = np.zeros(len(buildings_filter.geometry), dtype=bool)
        clusters = []

        def dfs(node, current_cluster):
            neighbors = kdtree.query_ball_point(coordinates[node], chain_upper_bound)
            unassigned_neighbors = [neighbor for neighbor in neighbors if not assigned[neighbor]]
            assigned[unassigned_neighbors] = True

            if node not in current_cluster:
                current_cluster.append(node)

            for neighbor in unassigned_neighbors:
                dfs(neighbor, current_cluster)

        for i, shapely_point in enumerate(buildings_filter.geometry):
            if not assigned[i]:
                current_cluster = []
                dfs(i, current_cluster)
                clusters.append(current_cluster)
        return clusters

    result_clusters = create_clusters(buildings_filter, chain_upper_bound)
    i = 0
    for clus in result_clusters:
        if len(clus) > 2:
            coords = [(point.x, point.y) for point in buildings_filter.loc[clus, 'geometry']]
            distances = squareform(pdist(coords))
            agg_cluster = AgglomerativeClustering(distance_threshold=pole_upper_bound, n_clusters=None, linkage='complete')
            cluster_labels = agg_cluster.fit_predict(distances)
            if len(set(cluster_labels)) > 1:
                for j in list(set(cluster_labels)):
                    indices = [index for index, value in enumerate(cluster_labels) if value == j]
                    buildings_filter.loc[[clus[k] for k in indices], 'Group2'] = i
                    i += 1
            else:
                buildings_filter.loc[clus, 'Group2'] = i
        else:
            buildings_filter.loc[clus, 'Group2'] = i
        i += 1

    collapse_results = [item for sublist in result_clusters for item in sublist]
    for i in range(len(result_clusters)):
        buildings_filter.loc[result_clusters[i], 'Group'] = i

    buildings_adjusted = []
    area = []
    num = []
    elec_access = []
    cons = []

    for group in buildings_filter['Group2'].unique():
        buildings_adjusted.append(MultiPoint(buildings_filter.loc[buildings_filter['Group2'] == group, 'geometry'].values).centroid)
        area.append(buildings_filter.loc[buildings_filter['Group2'] == group, 'area'].sum())
        cons.append(buildings_filter.loc[buildings_filter['Group2'] == group, 'cons (kWh/'].sum())
        num.append(len(buildings_filter.loc[buildings_filter['Group2'] == group, 'area']))
        elec_access.append(buildings_filter.loc[buildings_filter['Group2'] == group, 'elec acces'].mean())

    buildings_adjusted_gdf = gpd.GeoDataFrame({'area': area, 'number': num, 'cons (kWh/': cons, 'elec acces': elec_access},
                                              geometry=buildings_adjusted, crs=crs)
    return buildings_adjusted_gdf

def building_to_cluster_v1(path, crs, radius, dens_filter, flag):
    studyregion_original_path = path
    studyregion_original = gpd.read_file(studyregion_original_path)
    study_area_buffered = studyregion_original.buffer((2500 * 0.1 / 11250))

    base_dir = os.path.dirname(os.path.dirname(studyregion_original_path))
    building_path = os.path.join(base_dir, "OpenEnergyMap", "pixelOfStudy", "merged.shp")

    output_folder_points = os.path.join(os.path.dirname(building_path), 'points')
    output_folder_pointsclipped = os.path.join(os.path.dirname(building_path), 'points_clipped')
    output_path_points = os.path.join(output_folder_points, 'points.shp')
    output_path_points_clipped = os.path.join(output_folder_pointsclipped, 'points_clipped.shp')

    if not flag:
        buildings_df_original = gpd.read_file(building_path)
        buildings_df_original.rename(columns={'area_in_me': 'area'}, inplace=True)
        buildings_df = buildings_df_original.to_crs(crs)
        studyregion = studyregion_original.to_crs(crs)
        buildings_df['geometry'] = buildings_df.geometry.centroid
        buildings_df = gpd.clip(buildings_df, studyregion)
        buildings_df = buildings_df.reset_index(drop=True)
        area_lower_bound = 12
        buildings_df = buildings_df[buildings_df['area'] > area_lower_bound]
        buildings_df['ID'] = [*range(len(buildings_df))]
        buildings_df.reset_index(inplace=True, drop=True)
        urbanity = locate_file(base_dir, folder='Urbanity', extension='.tif')

        with rasterio.open(urbanity) as src:
            out_image, out_transform = mask(src, study_area_buffered.to_crs(src.crs), crop=True)

        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(os.path.join(base_dir, "Urbanity", "Urbanity_clip.tif"), "w", **out_meta) as dest:
            dest.write(out_image)

        with rasterio.open(os.path.join(base_dir, "Urbanity", "Urbanity_clip.tif")) as src:
            warp = src.read(1, out_shape=(src.height, src.width), resampling=Resampling.bilinear)
            output_modified_raster = os.path.join(base_dir, "Urbanity", "Urbanity_clip_rep_convolve.tif")
            raster = src.read(1)
            kernel = np.ones((3, 3))
            neighbor_sum = convolve(raster, kernel, mode='nearest')
            result = raster + 0.5 * (neighbor_sum - raster)
            result = result.astype(np.int16)
            meta = src.meta.copy()

            with rasterio.open(output_modified_raster, "w", **meta) as dest:
                dest.write(result, 1)

        Urbanity_final = rasterio.open(output_modified_raster)
        coords = [(point.x, point.y) for point in buildings_df['geometry']]
        buildings_df['urbanity'] = [x[0] for x in Urbanity_final.sample(coords)]

        if not os.path.exists(output_folder_pointsclipped):
            os.makedirs(output_folder_pointsclipped)
        buildings_df.to_file(output_path_points_clipped)

        if not os.path.exists(output_folder_points):
            os.makedirs(output_folder_points)

        max_distance = 20
        pole_distance = 30
        buildings_df_up = poles_clustering_and_cleaning(buildings_df, crs, max_distance, pole_distance)
        buildings_df_up.to_file(output_path_points)

    else:
        print('Skipped')
        buildings_df = gpd.read_file(output_path_points_clipped)

    buildings_df = buildings_df.reset_index(drop=True)
    x_interp = [55, 150]
    y_interp = [120, 50]

    interpolator = interp1d(x_interp, y_interp, kind='linear', fill_value='extrapolate')
    for index, row in buildings_df.iterrows():
        point_geometry = row.geometry
        urbanity_value = row['urbanity']
        buffer_radius = interpolator(urbanity_value)
        buildings_df.at[index, 'buffer'] = point_geometry.buffer(buffer_radius)

    geometries = buildings_df['buffer'].tolist()
    clusters_MP = unary_union(geometries)

    # Check if the result is a single Polygon or a MultiPolygon
    if isinstance(clusters_MP, MultiPolygon):
        clusters = [poly for poly in clusters_MP.geoms]
    elif isinstance(clusters_MP, Polygon):
        clusters = [clusters_MP]
    else:
        clusters = []


    clusters_gdf = gpd.GeoDataFrame(geometry=clusters, crs=crs)
    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    clusters_gdf['cluster_ID'] = clusters_gdf['cluster_ID'] + 1
    spatial_join = gpd.sjoin(buildings_df, clusters_gdf, how='left', predicate='within')

    try:
        buildings_df['cluster_ID'] = spatial_join['cluster_ID']
    except:
        buildings_df['cluster_ID'] = spatial_join['cluster_ID_right']

    cluster_counts = buildings_df['cluster_ID'].value_counts()
    clusters_to_keep = cluster_counts[cluster_counts >= 40].index
    clusters_gdf = clusters_gdf[clusters_gdf['cluster_ID'].isin(clusters_to_keep)]

    buildings_df.loc[~buildings_df['cluster_ID'].isin(clusters_to_keep), 'cluster_ID'] = -1
    average_elec_access = buildings_df.groupby('cluster_ID')['elec acces'].mean()

    if len(clusters_gdf) > 0:
        clusters_gdf = clusters_gdf.merge(average_elec_access, left_on='cluster_ID', right_index=True, how='left')
        clusters_gdf.drop(columns=['cluster_ID'], inplace=True)
        clusters_gdf['elec acces'] = clusters_gdf['elec acces'] / 100

    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    clusters_gdf['cluster_ID'] = clusters_gdf.index + 1

    output_folder_clusters = os.path.join(os.path.dirname(building_path), 'clusters')
    if not os.path.exists(output_folder_clusters):
        os.makedirs(output_folder_clusters)
    output_path_clusters = os.path.join(output_folder_clusters, 'clusters.shp')

    clusters_gdf.to_file(output_path_clusters)

    return output_path_points, output_path_clusters
