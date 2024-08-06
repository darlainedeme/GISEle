# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:11:18 2024

@author: EPSla
""" 
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
import geopandas as gpd
import numpy as np
import pandas as pd
import pdb 
import os
import pdb
from scipy import sparse
import zipfile
import os
import rasterio.mask
# from osgeo import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point, MultiPoint, MultiPolygon
from shapely.ops import nearest_points
from shapely.geometry import Polygon
from rasterio.plot import show
from rasterio.mask import mask
import json
import matplotlib.pyplot as plt
import fiona
from collections import Counter
from statistics import mean
from math import ceil
from shapely import ops
import initialization 
import QGIS_processing_polygon
import pdb 
from functions import *
from functions2 import *
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform 
from scipy.spatial import cKDTree as KDTree
import pdb
from scipy import sparse
from shapely.ops import unary_union 

def poles_clustering_and_cleaning(buildings_filter, crs, chain_upper_bound,pole_upper_bound):
    def create_clusters(buildings_filter,max_distance):
        coordinates = [(point.x, point.y) for point in buildings_filter.geometry]
        kdtree = cKDTree(coordinates)
        assigned = np.zeros(len(buildings_filter.geometry), dtype=bool) 
        #Here it creates a boolean fixed to false 
        clusters = []
        def dfs(node, current_cluster):
            # Depth-first search to find connected points within the given distance
            neighbors = kdtree.query_ball_point(coordinates[node], chain_upper_bound)
            unassigned_neighbors = [neighbor for neighbor in neighbors if not assigned[neighbor]]
            # pdb.set_trace()
            # Mark neighbors as assigned
            assigned[unassigned_neighbors] = True

            # Add the current point to the current cluster if it hasn't been added already
            if node not in current_cluster:
                current_cluster.append(node)

            # Recursively process unassigned neighbors
            for neighbor in unassigned_neighbors:
                dfs(neighbor, current_cluster)

        # Iterate through points to form clusters
        for i, shapely_point in enumerate(buildings_filter.geometry):
            if not assigned[i]:
                current_cluster = []
                dfs(i, current_cluster)
                clusters.append(current_cluster)
        return clusters
    result_clusters = create_clusters(buildings_filter,chain_upper_bound) 
    i=0
    for clus in result_clusters:
        if len(clus)>2: # if there are more thn 2 big macro areas
            coords = [(point.x, point.y) for point in buildings_filter.loc[clus,'geometry']]
            distances = squareform(pdist(coords))
            agg_cluster = AgglomerativeClustering(distance_threshold=pole_upper_bound,n_clusters=None,  linkage='complete')
            cluster_labels = agg_cluster.fit_predict(distances)
            if len(set(cluster_labels))>1: #if agglomerative clustering find more than 1 subgroups
                for j in list(set(cluster_labels)):
                    indices = [index for index, value in enumerate(cluster_labels) if value == j]
                    buildings_filter.loc[[clus[k] for k in indices],'Group2']=i
                    i+=1
                 
            else:
                buildings_filter.loc[clus,'Group2']=i
        else:
            buildings_filter.loc[clus,'Group2']=i
        i+=1 

    collapse_results = [item for sublist in result_clusters for item in sublist]
    for i in range(len(result_clusters)):
        buildings_filter.loc[result_clusters[i],'Group']=i
    buildings_adjusted = []
    area=[]
    num=[] 
    elec_access = []  
    
    cons = [] 
    # pdb.set_trace()
    for group in buildings_filter['Group2'].unique():
        buildings_adjusted.append(MultiPoint(buildings_filter.loc[buildings_filter['Group2']==group,'geometry'].values).centroid)
        area.append(buildings_filter.loc[buildings_filter['Group2']==group,'area'].sum()) 
        cons.append(buildings_filter.loc[buildings_filter['Group2']==group,'cons (kWh/'].sum())
        num.append(len(buildings_filter.loc[buildings_filter['Group2']==group,'area'])) 
        elec_access.append(buildings_filter.loc[buildings_filter['Group2']==group,'elec acces'].mean())
    
    buildings_adjusted_gdf = gpd.GeoDataFrame({'area':area,'number':num, 'cons (kWh/':cons, 'elec acces':elec_access},geometry=buildings_adjusted,crs=crs)
    
    return buildings_adjusted_gdf


def building_to_cluster_v1(path, crs, radius, dens_filter,flag):   
    #This is the most imporant function
    studyregion_original_path = path 
    studyregion_original = gpd.read_file(studyregion_original_path) 
    study_area_buffered=studyregion_original.buffer((2500*0.1/11250))
    
    base_dir = os.path.dirname(os.path.dirname(studyregion_original_path))  
    building_path =  os.path.join(base_dir, "OpenEnergyMap", "pixelOfStudy", "merged.shp")

    
    output_folder_points = os.path.join(os.path.dirname(building_path),'points') 
    output_folder_pointsclipped = os.path.join(os.path.dirname(building_path),'points_clipped') 
    output_path_points = os.path.join(output_folder_points,'points.shp')
    output_path_points_clipped = os.path.join(output_folder_pointsclipped,'points_clipped.shp')  
    if flag == False: 
        buildings_df_original = gpd.read_file(building_path)   
        buildings_df_original.rename(columns={'area_in_me': 'area'}, inplace=True)
        buildings_df = buildings_df_original.to_crs(crs) 
        # pdb.set_trace()
        studyregion = studyregion_original.to_crs(crs) 
        buildings_df['geometry'] = buildings_df.geometry.centroid   
        buildings_df=gpd.clip(buildings_df,studyregion) 
        buildings_df = buildings_df.reset_index(drop=True)  
        area_lower_bound = 12 
        buildings_df = buildings_df[buildings_df['area']>area_lower_bound]
        buildings_df['ID']=[*range(len(buildings_df))]
        buildings_df.reset_index(inplace=True,drop=True)
        urbanity=locate_file(base_dir,folder='Urbanity',extension='.tif')
        st.write(urbanity)
        # pdb.set_trace()
        with rasterio.open(urbanity,

                mode='r') as src:
            out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)
            print(src.crs)

        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                          "transform": out_transform})
        #darlain
        with rasterio.open( os.path.join(base_dir, "Urbanity", "Urbanity_clip.tif"), "w", **out_meta) as dest:
            dest.write(out_image)
        input_raster = gdal.Open( os.path.join(base_dir, "Urbanity", "Urbanity_clip.tif"))
        output_raster =  os.path.join(base_dir, "Urbanity", "Urbanity_clip_rep.tif")
        warp = gdal.Warp(output_raster, input_raster, dstSRS='EPSG:21095')
        output_modified_raster = os.path.join(base_dir, "Urbanity", "Urbanity_clip_rep_convolve.tif")
        Urbanity = rasterio.open(output_raster) 
        
        raster = Urbanity.read(1)
        
        # Define the convolution kernel (3x3 kernel with all values set to 1)
        kernel = np.ones((3, 3))
        
        neighbor_sum = convolve(raster, kernel,  mode='nearest')

# Compute the modified raster by adding half of the neighbor sum to each cell
        result = raster + 0.5 * (neighbor_sum - raster)
        result = result.astype(np.int16)
        # Update metadata for the output file
        meta = Urbanity.meta.copy()
        # meta.update({"driver": "GTiff",
        #              "height": result.shape[1],
        #              "width": result.shape[2],
        #              "transform": Urbanity.transform})
        
        # Save the result to a new raster file
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
            
        # if not os.path.exists(output_folder_pointsclipped):
        #     os.makedirs(output_folder_pointsclipped) 
            #r"D:\OneDrive - Politecnico di Milano\Corrado\Gisele\giselexMIT\Database\Uganda\OpenEnergyMap\499436_geoms\points\499436_geoms.shp"        
        max_distance=20
        pole_distance =30
        buildings_df_up = poles_clustering_and_cleaning(buildings_df, crs, max_distance, pole_distance)
        buildings_df_up.to_file(output_path_points) 

    else: 
        print('Skipped')
        buildings_df = gpd.read_file(output_path_points_clipped) 
        
        
    buildings_df = buildings_df.reset_index(drop=True)    
    x_interp = [55,150]
    y_interp = [120,50]
    
    # Create the interpolator function
    interpolator = interp1d(x_interp, y_interp, kind='linear', fill_value='extrapolate')
    for index, row in buildings_df.iterrows():
        point_geometry = row.geometry 
        urbanity_value = row['urbanity'] 
        buffer_radius = interpolator(urbanity_value)
        # if urbanity_value == 30:
        #     buffer_radius = 55
        # elif urbanity_value == 23:
        #     buffer_radius = 40 
        # elif urbanity_value == 22:
        #     buffer_radius = 60
        # elif urbanity_value == 21:
        #     buffer_radius = 80
        # elif urbanity_value == 13:
        #     buffer_radius = 90        
        # elif urbanity_value == 12:
        #     buffer_radius = 130
        # elif urbanity_value == 11:
        #     buffer_radius = 200
        # else:
        #     buffer_radius = 500  # Default to 0 or any other value you prefer
        buildings_df.at[index, 'buffer'] = point_geometry.buffer(buffer_radius)
        
    geometries = buildings_df['buffer'].tolist()
    clusters_MP = unary_union(geometries)

    clusters=[poly for poly in clusters_MP] 
    
    clusters_gdf = gpd.GeoDataFrame(geometry=clusters, crs=crs) 
    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    clusters_gdf['cluster_ID'] = clusters_gdf['cluster_ID']+1 
    spatial_join = gpd.sjoin(buildings_df, clusters_gdf, how='left', op='within') 
    
    try: 
        buildings_df['cluster_ID'] = spatial_join['cluster_ID']   
    except: 
        buildings_df['cluster_ID'] = spatial_join['cluster_ID_right']  
    
    cluster_counts = buildings_df['cluster_ID'].value_counts() 
    clusters_to_keep = cluster_counts[cluster_counts >= 40].index 
    clusters_gdf = clusters_gdf[clusters_gdf['cluster_ID'].isin(clusters_to_keep)]
    # clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    # try:
    #     spatial_join_filtered = spatial_join[spatial_join['cluster_ID'].isin(clusters_to_keep)] 
    # except: 
    #     spatial_join_filtered = spatial_join[spatial_join['cluster_ID_right'].isin(clusters_to_keep)] 
    
    # Update cluster_ID to -1 for buildings in clusters with less than 10 elements
    buildings_df.loc[~buildings_df['cluster_ID'].isin(clusters_to_keep), 'cluster_ID'] = -1

    
     
    average_elec_access = buildings_df.groupby('cluster_ID')['elec acces'].mean()   
    #I do here a average electrification access and then for each clustrer I do a random selection of the electrification according to that percentage
    # threshold = 0.3
  
    if len(clusters_gdf) > 0:
        clusters_gdf = clusters_gdf.merge(average_elec_access, left_on='cluster_ID', right_index=True, how='left')  
        clusters_gdf.drop(columns=['cluster_ID'], inplace=True)
        clusters_gdf['elec acces'] = clusters_gdf['elec acces']/100
        # clusters_gdf = clusters_gdf[clusters_gdf['elec acces'] < threshold].reset_index()
        # clusters_gdf.drop(columns=['index'], inplace=True)
        # clusters_gdf['cluster_ID'] = clusters_gdf.index +1
    
    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'}) 
    clusters_gdf['cluster_ID'] = clusters_gdf.index + 1
    
    output_folder_clusters = os.path.join(os.path.dirname(building_path),'clusters')
    if not os.path.exists(output_folder_clusters):
        os.makedirs(output_folder_clusters) 
        #r"D:\OneDrive - Politecnico di Milano\Corrado\Gisele\giselexMIT\Database\Uganda\OpenEnergyMap\499436_geoms\points\499436_geoms.shp"
    output_path_clusters = os.path.join(output_folder_clusters,'clusters.shp') 

    clusters_gdf.to_file(output_path_clusters)  

    return output_path_points , output_path_clusters  



