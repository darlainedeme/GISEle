# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:55:37 2024

@author: EPSla
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pdb 
import os

from functions import *
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform 
from scipy.spatial import cKDTree as KDTree
import pdb
from scipy import sparse




def point_density(point_gdf, calculate_percentiles=True, radius=284):
    '''Much more efficient method with sparse matrix, logarithmic rise'''
    method='MIT' #Old / MIT /New 
    
    n = len(point_gdf)
    loc = [[point_gdf.loc[i, 'geometry'].xy[0][0], point_gdf.loc[i, 'geometry'].xy[1][0]] for i in range(n)]
    print('Coordinates extracted')
    # dist_matrix2 = scipy.spatial.distance_matrix(loc,loc)
    A = KDTree(loc)
    print('KDTree')
    #end = time.time()
    # list all pairs within 0.05 of each other in 2-norm
    # format: (i, j, v) - i, j are indices, v is distance
    D = A.sparse_distance_matrix(A, radius, p=2.0, output_type='ndarray')
    print('Sparse distance matrix done')
    # only keep upper triangle
    DU = D[D['i'] < D['j']]
    print('Simplify by taking the upper triangle')
    # make sparse matrix
    result = sparse.coo_matrix((DU['v'], (DU['i'], DU['j'])), (n, n))
    result_csr = result.tocsr()
    if method=='Old': #this method is less memory efficient - but it allows to save more data on the distance between nodes
        sparse_dist_matrix = [((i, j), result_csr[i, j]) for i, j in zip(*result_csr.nonzero())]
        print('Transformed sparse distance matrix')
        a = [[] for x in range(n)]
        for i in range(len(sparse_dist_matrix)):
            point1 = sparse_dist_matrix[i][0][0]
            point2 = sparse_dist_matrix[i][0][1]
            a[point1].append(point2)
            a[point2].append(point1)
        
        print('Calculation finished')
        population = point_gdf['Population'].tolist()
        new_pop_density_column = []
        
        new_pop_density_column = [sum(population[i] for i in a[counter])+population[counter] for counter in range(n)]
        point_gdf['Pop_'+str(radius)] = new_pop_density_column
    elif method=='New': #more memory efficient but doesn't save anything inbetween
        population = point_gdf['Population'].tolist()
        count=0
        for i,j in zip(*result_csr.nonzero()):
            population[i]+=1
            population[j]+=1
            count+=1
        point_gdf['Pop_'+str(radius)] = population  
        
    elif method=='MIT': 
        population = np.ones(len(point_gdf['cons (kWh/'])).tolist() 
        count=0
        for i,j in zip(*result_csr.nonzero()):
            population[i]+=1
            population[j]+=1
            count+=1
        point_gdf['Pop_'+str(radius)] = population 
            
    if calculate_percentiles:
        sorted_density = np.argsort(new_pop_density_column)
        sorted_population = np.sort(population)
        new_pop_density_sorted_percentile = np.zeros(n)
        percentile_population = np.zeros(n)
        new_pop_density_sorted = np.zeros(n)
        total_population = sum(population)
        sorted_population_cumsum = np.cumsum(sorted_population)
        for i in range(n):
            point = sorted_density[i]
            new_pop_density_sorted[point] = i
            new_pop_density_sorted_percentile[point] = i / n * 100
            percentile_population[point] = sorted_population_cumsum[
                                               i] / total_population * 100  # this is considers the percentile as well. In the end, perhaps the 10%
            # most densely populated points will have 30% of the population. This is quite important for the clustering and sensitivity analysis.
            #print('\r' + str(i) + '/' + str(n),
            #      sep=' ', end='', flush=True)
    
        point_gdf['perce_pts'] = new_pop_density_sorted_percentile  # this is percentiles but just in terms of points, it doesn't consider the population
        point_gdf['perce_ppl'] = percentile_population

    return point_gdf



def poles_clustering_and_cleaning(buildings, crs, area_lower_bound, chain_upper_bound,pole_upper_bound):
    buildings_filter = buildings[buildings['area']>area_lower_bound]
    buildings_filter['ID']=[*range(len(buildings_filter))]
    buildings_filter.reset_index(inplace=True,drop=True) 
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
    return buildings_filter, buildings_adjusted_gdf

def building_to_cluster(path, crs, radius, dens_filter,flag):  
    #This is the most imporant function
    studyregion_original_path = path 
    studyregion_original = gpd.read_file(studyregion_original_path)
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
        area_limit=12
        max_distance=20
        pole_distance =30
        
        buildings_filter, buildings_df_upd = poles_clustering_and_cleaning(buildings_df, crs, area_limit, max_distance, pole_distance) 
    else: 
        print('skipped')
        buildings_df_upd = gpd.read_file(output_path_points_clipped) 
        

    
    percentiles = False
    points_gdf_dens = point_density(buildings_df_upd, percentiles , radius)

    points_filtered = points_gdf_dens[points_gdf_dens['Pop_'+str(radius)]>=dens_filter]
    clusters_MP = points_filtered.geometry.buffer(radius).unary_union
    clusters=[poly for poly in clusters_MP] 
    
    clusters_gdf=gpd.GeoDataFrame(geometry=clusters,crs=points_gdf_dens.crs) 
    clusters_gdf = clusters_gdf.reset_index().rename(columns={'index': 'cluster_ID'})
    clusters_gdf['cluster_ID']=clusters_gdf['cluster_ID']+1
    spatial_join = gpd.sjoin(buildings_df_upd, clusters_gdf, how='left', op='within')
    buildings_df_upd['cluster_ID'] = spatial_join['cluster_ID']  

   
    average_elec_access = buildings_df_upd.groupby('cluster_ID')['elec acces'].mean()   
    #I do here a average electrification access and then for each clustrer I do a random selection of the electrification according to that percentage
    threshold = 0.3

    if len(clusters_gdf) > 0:
        clusters_gdf = clusters_gdf.merge(average_elec_access, left_on='cluster_ID', right_index=True, how='left')  
        clusters_gdf.drop(columns=['cluster_ID'], inplace=True)
        clusters_gdf['elec acces'] = clusters_gdf['elec acces']/100
        clusters_gdf = clusters_gdf[clusters_gdf['elec acces'] < threshold].reset_index()
        clusters_gdf.drop(columns=['index'], inplace=True)
        clusters_gdf['cluster_ID'] = clusters_gdf.index +1
        # clusters_gdf['elec state'] = clusters_gdf['elec acces'].apply(lambda x: random_selection(x))


    if not os.path.exists(output_folder_points):
        os.makedirs(output_folder_points)  
    if not os.path.exists(output_folder_pointsclipped):
        os.makedirs(output_folder_pointsclipped) 
        #r"D:\OneDrive - Politecnico di Milano\Corrado\Gisele\giselexMIT\Database\Uganda\OpenEnergyMap\499436_geoms\points\499436_geoms.shp"
    if flag == False:
        buildings_df_upd.to_file(output_path_points)     
        buildings_df.to_file(output_path_points_clipped)
    
    output_folder_clusters = os.path.join(os.path.dirname(building_path),'clusters')
    if not os.path.exists(output_folder_clusters):
        os.makedirs(output_folder_clusters) 
        #r"D:\OneDrive - Politecnico di Milano\Corrado\Gisele\giselexMIT\Database\Uganda\OpenEnergyMap\499436_geoms\points\499436_geoms.shp"
    output_path_clusters = os.path.join(output_folder_clusters,'clusters.shp') 

    clusters_gdf.to_file(output_path_clusters)  

    return output_path_points , output_path_clusters




# path = r"D:\OneDrive - Politecnico di Milano\Corrado\Gisele\giselexMIT\Database\Uganda\OpenEnergyMap\499436_geoms\499436_geoms.geojson"  
# crs = 21095  
# radius = 300
# dens_filter = 10 # number of buildings in the radius defined   
# buildings_df = building_to_cluster(path,crs,radius,dens_filter)   
# summary = statistics(buildings_df) 

