import os
import geopandas as gpd
import pandas as pd
import rasterio
import streamlit as st
from shapely.geometry import Point, LineString, MultiPoint
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
from networkx import Graph
from shapely.ops import split,nearest_points
import networkx as nx
from itertools import chain
from networkx.utils import pairwise
from scipy.spatial import Delaunay
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from math import ceil
from scripts.gisele_scripts.geneticalgorithm_github import geneticalgorithm as ga
from scipy.spatial import distance_matrix

# Reproject raster utility function
def reproject_raster(input_raster, dst_crs):
    with rasterio.open(input_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
            return memfile.open()
            
# Sample raster values at coordinates
def sample_raster(raster, coords):
    values = []
    for val in raster.sample(coords):
        values.append(val[0])
    return values

def create_grid(crs, resolution, study_area):
    # crs and resolution should be a numbers, while the study area is a polygon
    df = pd.DataFrame(columns=['X', 'Y'])
    min_x, min_y, max_x, max_y = study_area.bounds
    # create one-dimensional arrays for x and y
    lon = np.arange(min_x, max_x, resolution)
    lat = np.arange(min_y, max_y, resolution)
    lon, lat = np.meshgrid(lon, lat)
    df['X'] = lon.reshape((np.prod(lon.shape),))
    df['Y'] = lat.reshape((np.prod(lat.shape),))
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                              crs=crs)
    geo_df_clipped = gpd.clip(geo_df, study_area)
    # geo_df_clipped.to_file(r'Test\grid_of_points.shp')
    return geo_df_clipped

def metric_closure(G, weight='weight'):
    """  Return the metric closure of a graph.

    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.

    """
    M = nx.Graph()

    Gnodes = set(G)

    # check for connected graph while processing first node
    all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
    u, (distance, path) = next(all_paths_iter)
    if Gnodes - set(distance):
        msg = "G is not a connected graph. metric_closure is not defined."
        raise nx.NetworkXError(msg)
    Gnodes.remove(u)
    for v in Gnodes:
        M.add_edge(u, v, distance=distance[v], path=path[v])

    # first node done -- now process the rest
    for u, (distance, path) in all_paths_iter:
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])

    return M

def genetic2(clustered_points, points_new_graph, distance_matrix, n_clusters, graph):
    clustered_points.reset_index(drop=True, inplace=True)
    lookup_edges = [i for i in graph.edges]
    dim = len(lookup_edges) - 1
    dist_matrix_df = pd.DataFrame(distance_matrix, columns=[i for i in points_new_graph['ID']], index=[i for i in points_new_graph['ID']])
    
    varbound = np.array([[0, dim]] * (n_clusters - 1))
    
    def fitness(X):
        T = graph.copy()
        length_deleted_lines = 0
        count = Counter(X)
        for i in count:
            if count[i] > 1:
                return 1000000  # this is in case it is trying to cut the same branch more than once
        
        for i in X:
            delete_edge = lookup_edges[int(i)]
            if T.has_edge(*delete_edge):  # Check if the edge exists
                length_deleted_lines += graph[lookup_edges[int(i)][0]][lookup_edges[int(i)][1]]['weight']['distance']
                T.remove_edge(*delete_edge)
            else:
                return 1000000  # Return a high penalty if the edge doesn't exist
        
        islands = [c for c in nx.connected_components(T)]
        cost = 0
        penalty = 0
        for i in range(len(islands)):
            subgraph = T.subgraph(islands[i])
            subset_IDs = [i for i in subgraph.nodes]
            population = points_new_graph[points_new_graph['ID'].isin(subset_IDs)]['Population'].sum()
            power = population * 0.7 * 0.3
            if power < 25:
                cost += 1500
            elif power < 50:
                cost += 2300
            elif power < 100:
                cost += 3500
            else:
                cost += 100000
            sub_dist_matrix = dist_matrix_df.loc[subset_IDs, subset_IDs]
            max_dist = sub_dist_matrix.max().max()
            if max_dist > 1000:
                penalty = penalty + 50000 + (max_dist - 500) * 25

        cost = cost - length_deleted_lines / 1000 * 10000  # divided by 1000 for m->km and the multiplied by 10000euro/km

        if penalty > 0:
            return penalty
        else:
            return cost

    algorithm_param = {
        'max_num_iteration': 100,  # Reduce iterations
        'population_size': 20,  # Reduce population size
        'mutation_probability': 0.05,  # Adjust as needed
        'elit_ratio': 0.02,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50  # Stop earlier if no improvement
    }              
    model = ga(function=fitness, dimension=n_clusters - 1, variable_type='int', variable_boundaries=varbound,
               function_timeout=10000, algorithm_parameters=algorithm_param)
    model.run()
    cut_edges = model.best_variable
    T = graph.copy()
    for i in cut_edges:
        delete_edge = lookup_edges[int(i)]
        print(f"Attempting to remove edge: {delete_edge}")
        if T.has_edge(*delete_edge):
            T.remove_edge(*delete_edge)
        else:
            print(f"Edge {delete_edge} does not exist in the graph")

    islands = [c for c in nx.connected_components(T)]
    for i in range(len(islands)):
        subgraph = T.subgraph(islands[i])
        subset_IDs = [i for i in subgraph.nodes]
        clustered_points.loc[clustered_points['ID'].isin(subset_IDs), 'Cluster'] = i

    return clustered_points, cut_edges

def steiner_tree(G, terminal_nodes, weight='weight'):
    """ Return an approximation to the minimum Steiner tree of a graph.

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    Notes
    -----
    Steiner tree can be approximated by computing the minimum spanning
    tree of the subgraph of the metric closure of the graph induced by the
    terminal nodes, where the metric closure of *G* is the complete graph in
    which each edge is weighted by the shortest path distance between the
    nodes in *G* .
    This algorithm produces a tree whose weight is within a (2 - (2 / t))
    factor of the weight of the optimal Steiner tree where *t* is number of
    terminal nodes.

    """
    # M is the subgraph of the metric closure induced by the terminal nodes of
    # G.
    M = metric_closure(G, weight=weight)
    # Use the 'distance' attribute of each edge provided by the metric closure
    # graph.
    H = M.subgraph(terminal_nodes)
    mst_edges = nx.minimum_spanning_edges(H, weight='distance', data=True)
    # Create an iterator over each edge in each shortest path; repeats are okay
    edges = chain.from_iterable(pairwise(d['path']) for u, v, d in mst_edges)
    T = G.edge_subgraph(edges)
    return T

def connect_unconnected_graph(graph,lines,points,weight):
    if nx.is_connected(graph):
        return graph,lines
    else:
        islands = [c for c in nx.connected_components(graph)]
        for i in range(len(islands)):
            for j in range(i+1,len(islands)):
                subgraph_1 = [val for val in islands[i]]
                subgraph_2 = [val for val in islands[j]]
                points_s1 = points.loc[points['ID'].isin(subgraph_1),:]
                points_s2 = points.loc[points['ID'].isin(subgraph_2), :]
                multi_point1= MultiPoint([row['geometry'] for i, row in points_s1.iterrows()])
                multi_point2 = MultiPoint([row['geometry'] for i, row in points_s2.iterrows()])
                closest_points = nearest_points(multi_point1,multi_point2)
                distance = multi_point1.distance(multi_point2)#in km
                id_point1 = int(points.loc[points['geometry']==closest_points[0],'ID'])
                id_point2 = int(points.loc[points['geometry'] == closest_points[1], 'ID'])
                lines = pd.concat([lines, gpd.GeoDataFrame({'ID1':[id_point1],'ID2':[id_point2],'length':[distance], 'geometry':[LineString([closest_points[0],closest_points[1]])]})], ignore_index=True)
                graph.add_edge(id_point1,id_point2,weight = distance*weight,length = distance)
    return graph,lines

def categorize_substation(clusters_list, substations):
    values = []
    costs = []
    substations['Rated_power [kVA]'] = substations['Rated_power [kVA]']
    substations1 = substations['Rated_power [kVA]'].to_list()
    for index, row in clusters_list.iterrows():
        load_kVA = row.loc['Load [kW]'] / 0.9  # considering a power factor of 0.9
        substations2 = [i - load_kVA if i - load_kVA > 0 else 10000 for i in substations1]
        power = int(min(substations2) + load_kVA)
        values.append(power)
        locate_cost = substations[substations['Rated_power [kVA]'] == power]['Cost[euro]']
        costs.append(locate_cost.values[0])

    clusters_list['Transformer_rated_power [kVA]'] = values
    clusters_list['Cost[euro]'] = costs

    return clusters_list

def distance_2d(df1, df2, x, y):
    """
    Find the 2D distance matrix between two datasets of points.
    :param df1: first point dataframe
    :param df2: second point dataframe
    :param x: column representing the x coordinates (longitude)
    :param y: column representing the y coordinates (latitude)
    :return value: 2D Distance matrix between df1 and df2
    """

    d1_coordinates = {'x': df1[x], 'y': df1[y]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']


    d2_coordinates = {'x': df2[x], 'y': df2[y]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = distance_matrix(df1_loc, df2_loc)
    return value

def delaunay_test(graph,new_points,new_lines):
    tocki = new_points['geometry'].values
    number_points = new_points.shape[0]
    arr = np.zeros([number_points,2])
    counter=0
    for i in tocki:
        x = i.xy[0][0]
        y=i.xy[1][0]
        arr[counter,0] = x
        arr[counter,1] = y
        counter+=1
    tri = Delaunay(arr)
    triangle_sides = tri.simplices
    final_sides = []
    for i in triangle_sides:
        a=i[0]
        b=i[1]
        c=i[2]
        if a>b:
            final_sides.append((i[0],i[1]))
        else:
            final_sides.append((i[1], i[0]))
        if b>c:
            final_sides.append((i[1],i[2]))
        else:
            final_sides.append((i[2], i[1]))
        if a>c:
            final_sides.append((i[0],i[2]))
        else:
            final_sides.append((i[2], i[0]))
    final_sides2 = list(set(final_sides))
    new_lines_old=new_lines.copy() # dataframe without the new possible connections

    if not nx.is_empty(graph): # this is for the standard case with roads in the cluster
        for i,j in final_sides2:
            point1 = new_points.loc[new_points['order']==i,'geometry'].values[0]
            point2 = new_points.loc[new_points['order'] == j, 'geometry'].values[0]
            id1 = int(new_points.loc[new_points['order'] == i, 'ID'])
            id2 = int(new_points.loc[new_points['order'] == j, 'ID'])
            length = point1.distance(point2)
            line = LineString([point1, point2])
            if length<500 and not graph.has_edge(id1,id2) and ((sum([line.intersects(line1) for line1 in new_lines_old.geometry]) == 0) or
                    ((new_points.loc[new_points['ID'] == id1, 'pop_bool'] == 0).values[0]) or
                    (new_points.loc[new_points['ID'] == id2, 'pop_bool'] == 0).values[0]):
                graph.add_edge(id1,id2 , weight=length, length=length)

                data_segment = {'ID1': [id1], 'ID2': [id2], 'length': [point1.distance(point2) / 1000],
                                'geometry': [line], 'Type': ['Colateral']}
                new_lines = pd.concat([new_lines, gpd.GeoDataFrame(data_segment)], ignore_index=True)

    else: # this is for the case without roads in the cluster, just create the lines in a straightforward way
        new_points = new_points.reset_index()
        for i, j in final_sides2:
            point1 = new_points.loc[new_points.index == i, 'geometry'].values[0]
            point2 = new_points.loc[new_points.index== j, 'geometry'].values[0]
            id1 = int(new_points.loc[new_points.index == i, 'ID'].values[0])
            id2 = int(new_points.loc[new_points.index== j, 'ID'].values[0])
            length = point1.distance(point2)
            line = LineString([point1, point2])
            graph.add_edge(id1, id2, weight=length, length=length)

            data_segment = {'ID1': [id1], 'ID2': [id2], 'length': [point1.distance(point2) / 1000],
                            'geometry': [line], 'Type': ['Colateral']}
            new_lines = pd.concat([new_lines, gpd.GeoDataFrame(data_segment)], ignore_index=True)

    return graph,new_lines

def create_clean_graph(graph,points,terminal_points,T_metric,crs):
    '''This function returns a graph that is composed only of population nodes(translated on the roads) and the intersection points(points
    which are present in the existing graph more than 2 times. The idea is to start cutting the highest cost lines as the path
    to a much better clustering that includes the actual electrical distances.'''
    #WORKS
    #
    # STEP 1. Take all the terminal nodes + the intersection nodes

    terminal_IDs = terminal_points['ID'].to_list()
    edges_tuples = [i for i in graph.edges]
    nodes = [edges_tuples[i][0] for i in range(len(edges_tuples))]
    nodes+=[edges_tuples[i][1] for i in range(len(edges_tuples))]
    occurence = Counter(nodes)
    intersection_IDs=[]
    for i in occurence:
        if occurence[i]>2 and not i in terminal_IDs:
            intersection_IDs.append(i)
    new_nodes = terminal_IDs + intersection_IDs

    # STEP 2. Create the new graph
    start_node = new_nodes[0]
    #start_node = 154
    current_node = start_node
    graph_copy = graph.copy()
    new_graph=nx.Graph()
    terminal_IDs_2 = terminal_IDs.copy()
    unique_nodes=new_nodes.copy()
    while True:
        try:
            next_node = [i for i in graph_copy[current_node]][0]
            #print(next_node)
        except:
            print('A terminal node has been reached, back to the set of points')
            if current_node in terminal_IDs_2:
                terminal_IDs_2.remove(current_node)
                #print('Node ' + str(current_node) + ' was deleted.')
            #print('Next point is '+str(unique_nodes[0]))
            start_node = unique_nodes[0]
            current_node = start_node
            next_node = [i for i in graph_copy[start_node]][0]
        if next_node in new_nodes:
            new_graph.add_edge(start_node, next_node, weight=T_metric[start_node][next_node])
            #print('add ' + str(start_node) + ' and ' + str(next_node))
            graph_copy.remove_edge(current_node,next_node)
            #print('remove '+str(current_node)+' and ' + str(next_node))
            if start_node in terminal_IDs_2:
                terminal_IDs_2.remove(start_node)
                print('Node '+ str(start_node)+' was deleted.')
            start_node = next_node
            current_node = start_node
        else:
            graph_copy.remove_edge(current_node, next_node)
            #print('remove ' + str(current_node) + ' and ' + str(next_node))
            current_node = next_node

        if nx.is_empty(graph_copy):
            break
        new_edges = [i for i in graph_copy.edges]
        unique_nodes = list(set([new_edges[i][0] for i in range(len(new_edges))] + [new_edges[i][1] for i in range(len(new_edges))]))
        unique_nodes = list(set(unique_nodes) & set(new_nodes))
    new_edges = [i for i in new_graph.edges]
    new_lines=gpd.GeoDataFrame()
    for j in new_edges:
        point1 = points.loc[points['ID'] == j[0],'geometry'].values[0]
        point2 = points.loc[points['ID'] == j[1],'geometry'].values[0]
        length = new_graph[j[0]][j[1]]['weight']['distance']
        new_data = pd.DataFrame([{'geometry': LineString([point1, point2]), 'length': length}])
        new_lines = pd.concat([new_lines, new_data], ignore_index=True)

    new_lines.crs = crs


    new_points = points[points['ID'].isin(new_nodes)]
    return new_lines, new_points, new_graph

# Main optimization function
def optimize(crs, country, resolution, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance, ss_data,
             landcover_option, gisele_dir, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist, LV_base_cost, population_dataset_type):

    dir_input_1 = os.path.join(gisele_dir, 'data', '2_downloaded_input_data')
    dir_input = os.path.join(gisele_dir, 'data', '4_intermediate_output')
    dir_output = os.path.join(gisele_dir, 'data', '5_final_output')
    grid_of_points_path = os.path.join(dir_input, 'grid_of_points', 'weighted_grid_of_points_with_roads.csv')
    population_points_path = os.path.join(gisele_dir, 'data', '2_downloaded_input_data', 'buildings', 'mit', 'points_clipped', 'points_clipped.shp')
    roads_points_path = os.path.join(gisele_dir, 'data', '2_downloaded_input_data', 'roads', 'roads_points.shp')
    roads_lines_path = os.path.join(gisele_dir, 'data', '2_downloaded_input_data', 'roads', 'roads_lines.shp')
    ss_data_path = os.path.join(gisele_dir, 'data', '0_configuration_files', ss_data)

    grid_of_points = pd.read_csv(grid_of_points_path)
    grid_of_points_GDF = gpd.GeoDataFrame(grid_of_points, geometry=gpd.points_from_xy(grid_of_points.X, grid_of_points.Y), crs=crs)
    # Ensure the grid is in the desired CRS
    if grid_of_points_GDF.crs != crs:
        grid_of_points_GDF = grid_of_points_GDF.to_crs(crs)

    if 'Elevation.1' in grid_of_points_GDF.columns:
        grid_of_points_GDF.drop(columns=['Elevation.1'], inplace=True)

    Starting_node = int(grid_of_points_GDF['ID'].max() + 1)

    LV_resume = pd.DataFrame()
    LV_grid = gpd.GeoDataFrame()
    MV_grid = gpd.GeoDataFrame()
    secondary_substations = gpd.GeoDataFrame()
    all_houses = gpd.GeoDataFrame()

    Population = gpd.read_file(population_points_path)

    for index, row in Clusters.iterrows():
        dir_cluster = os.path.join(gisele_dir, 'data', '4_intermediate_output', 'optimization', str(row["cluster_ID"]))
        clus = row['cluster_ID'] 
        os.makedirs(dir_cluster, exist_ok=True)
        os.makedirs(os.path.join(dir_cluster, 'grids'), exist_ok=True)

        area = row['geometry']
        area_buffered = area.buffer(resolution)

        if population_dataset_type == 'mit':
            grid_of_points = create_grid(crs, resolution, area)
            Population = rasterio.open(os.path.join(dir_input_1, 'population', 'Population.tif'))
            # Reproject the grid points to the CRS of the raster before sampling
            if grid_of_points.crs != Population.crs:
                grid_of_points = grid_of_points.to_crs(Population.crs)

        else:
            grid_of_points = gpd.clip(Population, area_buffered)
            grid_of_points['X'] = grid_of_points.geometry.x
            grid_of_points['Y'] = grid_of_points.geometry.y
            grid_of_points = grid_of_points[['X', 'Y', 'geometry']]

        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        road_points = gpd.read_file(roads_points_path)
        # Ensure the CRS is correct
        if road_points.crs != crs:
            road_points = road_points.to_crs(crs)
        road_points = gpd.clip(road_points, area_buffered)
        road_lines = gpd.read_file(roads_lines_path)
        if road_lines.crs != crs:
            road_lines = road_lines.to_crs(crs)
        road_lines = road_lines[(road_lines['ID1'].isin(road_points.ID.to_list()) & road_lines['ID2'].isin(road_points.ID.to_list()))]

        Elevation = reproject_raster(os.path.join(dir_input_1, 'elevation', 'Elevation.tif'), crs)
        Slope = reproject_raster(os.path.join(dir_input_1, 'slope', 'slope.tif'), crs)
        LandCover = reproject_raster(os.path.join(dir_input_1, 'landcover', 'LandCover.tif'), crs)

        coords = [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)]
        grid_of_points = grid_of_points.reset_index(drop=True)
        grid_of_points['ID'] = grid_of_points.index

        if population_dataset_type == 'raster':
            grid_of_points['Population'] = sample_raster(Population, coords)
        else:
            try:
                st.write("Grid of points (without geometry):")
                st.write(grid_of_points.drop(columns='geometry').head())

                grid_of_points['Population'] = pop_per_household
                grid_of_points.loc[(grid_of_points['building'] == 'residential') & (grid_of_points['area'] > 120), 'Population'] = 10
                grid_of_points.loc[(grid_of_points['building'] == 'residential') & (grid_of_points['height'].astype(float) > 12), 'Population'] = 10
                grid_of_points.loc[(grid_of_points['building'] == 'residential') & (grid_of_points['area'] > 120) & (grid_of_points['height'].astype(float) > 12), 'Population'] = 25

            except: 
                grid_of_points['Population'] = 1    #TODO check this assumption   
        grid_of_points['Elevation'] = sample_raster(Elevation, coords)
        grid_of_points['Slope'] = sample_raster(Slope, coords)
        grid_of_points['Land_cover'] = sample_raster(LandCover, coords)
        grid_of_points['Protected_area'] = ['FALSE' for _ in range(len(coords))]

        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        Population_clus = grid_of_points[grid_of_points['Population'] > 0]
        Population_clus['ID'] = [*range(Starting_node, Starting_node + Population_clus.shape[0])]
        Population_clus['pop_bool'] = 1
        Starting_node += Population_clus.shape[0]
        Population_clus.to_file(os.path.join(dir_cluster, 'population.shp'))

        if not len(road_points) < 5 and not road_lines.empty:
            roads_multipoint = MultiPoint([point for point in road_points['geometry']])
            road_points['Population'] = 0
            road_points['pop_bool'] = 0
            road_points_populated = road_points.copy()
            for i, pop in Population_clus.iterrows():
                point = pop['geometry']
                nearest_geoms = nearest_points(point, roads_multipoint)
                closest_road_point = nearest_geoms[1]
                road_points_populated.loc[road_points_populated['geometry'] == closest_road_point, 'Population'] = pop['Population']
                road_points_populated.loc[road_points_populated['geometry'] == closest_road_point, 'pop_bool'] = 1
            road_points_populated.crs = crs
            road_points_populated.to_file(os.path.join(dir_cluster, 'road_points_populated.shp'))

            # Check if 'ID' exists in the columns
            if 'ID' not in road_points_populated.columns:
                road_points_populated = road_points_populated.reset_index(drop=False).set_index('ID', drop=False)
            else:
                road_points_populated = road_points_populated.set_index('ID', drop=False)


            # road_points_populated = road_points_populated.set_index('ID', drop=False)
            road_lines = road_lines.set_index(pd.Index([*range(road_lines.shape[0])]))

            graph = nx.Graph()
            for index, ROW in road_lines.iterrows():
                id1 = ROW['ID1']
                id2 = ROW['ID2']
                graph.add_edge(id1, id2, weight=ROW['length'] * 1000, length=ROW['length'] * 1000)
            graph, new_lines = connect_unconnected_graph(graph, road_lines, road_points_populated, weight=5)
            road_lines.to_file(os.path.join(dir_cluster, 'road_lines.shp'))

            populated_points = road_points_populated[road_points_populated['pop_bool'] == 1]
            terminal_nodes = list(populated_points['ID'])
            st.write(road_points_populated.drop(columns='geometry').head())
            road_points_populated = road_points_populated.reset_index(drop=True)
            road_points_populated.to_file(os.path.join(dir_cluster, 'road_terminal_points.shp'))
            tree = steiner_tree(graph, terminal_nodes)
            path = list(tree.edges)
            grid_routing = gpd.GeoDataFrame()
            counter = 0
            for i in path:
                point1 = road_points_populated.loc[road_points_populated['ID'] == i[0], 'geometry'].values[0]
                point2 = road_points_populated.loc[road_points_populated['ID'] == i[1], 'geometry'].values[0]
                geom = LineString([point1, point2])
                grid_routing = pd.concat([grid_routing, gpd.GeoDataFrame({'ID': [counter], 'geometry': [geom]})], ignore_index=True)
                counter += 1
            grid_routing.crs = crs
            grid_routing.to_file(os.path.join(dir_cluster, 'LV_backbone.shp'))

            road_points_backbone = road_points_populated[road_points_populated['ID'].isin(list(tree.nodes))]
            road_points_backbone['Population'] = 0
            road_points_backbone['pop_bool'] = 0
            index = [*range(road_points_backbone['ID'].astype(int).max() + 1,
                            road_points_backbone['ID'].astype(int).max() + 1 + Population_clus.shape[0])]
            Population_clus['ind'] = index
            Population_clus.set_index('ind', inplace=True, drop=True)

            all_points = pd.concat([road_points_backbone, Population_clus], ignore_index=True)
            new_graph = tree.copy()
            for n in new_graph.edges:
                new_graph[n[0]][n[1]]['weight'] = new_graph[n[0]][n[1]]['weight'] * 0.03

            road_lines_copy = road_lines.copy()
            road_lines['Type'] = 'Road'
            all_points['order'] = [*range(all_points.shape[0])]

            new_graph, all_lines = delaunay_test(new_graph, all_points, road_lines)
            new_graph, new_lines = connect_unconnected_graph(new_graph, new_lines, all_points, weight=3)
            terminal_nodes = all_points.loc[all_points['pop_bool'] == 1, 'ID'].to_list()

        else:
            dist_2d_matrix = distance_2d(Population_clus, Population_clus, 'X', 'Y')
            dist_2d_matrix = pd.DataFrame(dist_2d_matrix, columns=Population_clus.ID, index=Population_clus.ID)
            terminal_nodes = Population_clus['ID'].to_list()
            graph = nx.Graph()
            new_graph, all_lines = delaunay_test(graph, Population_clus, road_lines)
            all_points = Population_clus.copy()
        tree_final = steiner_tree(new_graph, terminal_nodes)
        grid_final = gpd.GeoDataFrame()
        path = list(tree_final.edges)
        counter = 0
        for i in path:
            point1 = all_points.loc[all_points['ID'] == i[0], 'geometry'].values[0]
            point2 = all_points.loc[all_points['ID'] == i[1], 'geometry'].values[0]
            geom = LineString([point1, point2])
            
            grid_final = pd.concat([grid_final, gpd.GeoDataFrame({'ID': [counter], 'geometry': [geom]})], ignore_index=True)
            counter += 1
        grid_final.crs = crs
        grid_final.to_file(os.path.join(dir_cluster, 'grid_final.shp'))

        T_metric = metric_closure(tree_final, weight='length')
        populated_points = all_points[all_points['pop_bool'] == 1]
        lines_new_graph, points_new_graph, new_graph = create_clean_graph(tree_final, all_points, populated_points, T_metric, crs)
        points_set = all_points.loc[all_points['pop_bool'] == 1, 'ID'].values
        dist_matrix = np.zeros((len(points_set), len(points_set)))
        for i in range(len(points_set)):
            for j in range(len(points_set)):
                if not i == j:
                    dist_matrix[i, j] = T_metric[points_set[i]][points_set[j]]['distance']
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete',
                                             distance_threshold=2 * LV_distance).fit(dist_matrix)

        populated_points.loc[:, 'Cluster'] = clustering.labels_
        clustered_points = populated_points.copy()

        clustered_points.to_file(os.path.join(dir_cluster, 'Clustered_points.shp'))
        populated_points['Population'] = [ceil(i) for i in populated_points['Population']]
        number_clusters = populated_points['Cluster'].max() + 1
        if number_clusters > 1:
            lookup_edges = [i for i in new_graph.edges]

            if run_genetic:
                points_set = points_new_graph.ID.to_list()
                dist_matrix2 = np.zeros((len(points_set), len(points_set)))
                for i in range(len(points_set)):
                    for j in range(len(points_set)):
                        if not i == j:
                            dist_matrix2[i, j] = T_metric[points_set[i]][points_set[j]]['distance']

                clustered_points, cut_edges = genetic2(populated_points, points_new_graph, dist_matrix2, number_clusters, new_graph)
                clustered_points.to_file(os.path.join(dir_cluster, 'Clustered_points_after_genetic.shp'))
            else:
                for number_clus in range(clustered_points['Cluster'].max() + 1):
                    subset = clustered_points[clustered_points['Cluster'] == number_clus]
                    if len(subset) == 1:
                        points_new_graph.loc[points_new_graph['ID'] == int(clustered_points.loc[clustered_points['Cluster'] == number_clus, 'ID']), 'Cluster'] = number_clus
                    else:
                        edges = nx.edges(new_graph, subset.ID.to_list())
                        edges_nodes = [node for tuple in edges for node in tuple]
                        count = Counter(edges_nodes)
                        terminal_node = [i for i in count if count[i] == 1]
                        terminal_node = [i for i in terminal_node if int(points_new_graph.loc[points_new_graph['ID'] == i, 'pop_bool']) == 1]
                        for i in range(len(terminal_node) - 1):
                            for j in range(i + 1, len(terminal_node)):
                                path = T_metric[terminal_node[i]][terminal_node[j]]['path']
                                points_new_graph.loc[points_new_graph['ID'].isin(path), 'Cluster'] = number_clus
                for ind, row in clustered_points.iterrows():
                    points_new_graph.loc[points_new_graph['ID'] == row['ID'], 'Cluster'] = row['Cluster']
                cut_edges = []
                for i in range(len(lookup_edges)):
                    try:
                        line = lookup_edges[i]
                        point1_cluster = int(points_new_graph.loc[points_new_graph['ID'] == line[0], 'Cluster'])
                        point2_cluster = int(points_new_graph.loc[points_new_graph['ID'] == line[1], 'Cluster'])
                        if not point1_cluster == point2_cluster:
                            cut_edges.append(i)
                    except:
                        cut_edges.append(i)
            tree_final = nx.Graph(tree_final)
            tree_final_copy = tree_final.copy()
            for i in cut_edges:
                edge = lookup_edges[int(i)]

                if nx.has_path(tree_final, edge[0], edge[1]): # darlain
                    edge_path = nx.dijkstra_path(tree_final, edge[0], edge[1])
                
                else:
                    # Handle the case where no path exists
                    print(f"No path exists between nodes {edge[0]} and {edge[1]}")
                    continue  # or perform any other necessary action
                    
                for j in range(len(edge_path) - 1):
                    tree_final.remove_edge(*(edge_path[j], edge_path[j + 1]))
        islands = [c for c in nx.connected_components(tree_final)]
        islands = [i for i in islands if len(i) > 1]
        if len(islands) > clustered_points['Cluster'].max() + 1:
            for i in range(len(islands)):
                subgraph_IDs = list(islands[i])
                clustered_points.loc[clustered_points['ID'].isin(subgraph_IDs), 'Cluster'] = i

            number_clusters = len(islands)
            clustered_points.to_file(os.path.join(dir_cluster, 'Clustered_points.shp'))
        for i in range(len(islands)):
            subgraph = tree_final.subgraph(islands[i])
            LV_grid_length = 0
            for i in subgraph.edges:
                LV_grid_length += subgraph[i[0]][i[1]]['length']
            check_cluster = True
            subset_IDs = [i for i in subgraph.nodes]
            all_points_subset = all_points.loc[all_points['ID'].isin(subset_IDs), :]
            all_points_subset['total_distance'] = 10000
            all_points_subset['feasible'] = True
            for index, row in all_points_subset.iterrows():
                if check_cluster and row['pop_bool'] == 1:
                    check_id = row['ID']
                    cluster = int(clustered_points.loc[clustered_points['ID'] == check_id, 'Cluster'])
                total_weighted_distance = 0
                max_dist = 0
                if all_points_subset.loc[index, 'feasible'] == True:
                    for index1, row1 in all_points_subset.loc[all_points_subset['pop_bool'] == 1, :].iterrows():

                        if index == index1:
                            total_distance = 0
                        else:
                            total_distance = T_metric[int(row['ID'])][int(row1['ID'])]['distance']
                        if total_distance > LV_grid_length * 1.3:
                            all_points_subset.loc[index, 'feasible'] = False
                            all_points_subset.loc[index1, 'feasible'] = False
                            continue
                        elif not total_distance == 0:
                            total_weighted_distance += total_distance
                            if total_distance > max_dist:
                                max_dist = total_distance
                    all_points_subset.loc[index, 'av_distance'] = total_weighted_distance / len(all_points_subset)
                    all_points_subset.loc[index, 'max_distance'] = max_dist
                    all_points_subset.loc[index, 'final_distance'] = total_weighted_distance / len(
                        all_points_subset) * 0.9 + max_dist * 0.1
            feasible_sites = all_points_subset.loc[all_points_subset['feasible'] == True, :]
            best_site_ID = int(feasible_sites.loc[feasible_sites['final_distance'] == feasible_sites['final_distance'].min(), 'ID'].values[0])
            all_points.loc[all_points['ID'] == best_site_ID, 'substations'] = True
            all_points.loc[all_points['ID'] == best_site_ID, 'Cluster'] = cluster
            all_points.loc[all_points['ID'] == best_site_ID, 'LV_length'] = LV_grid_length
            all_points.loc[all_points['ID'] == best_site_ID, 'max_distance'] = float(feasible_sites.loc[feasible_sites['ID'] == best_site_ID, 'max_distance'])
        MV_LV_substations = all_points.loc[all_points['substations'] == True, :]

        grid_final = gpd.GeoDataFrame()
        path = list(tree_final.edges)
        counter = 0
        for i in path:
            point1 = all_points.loc[all_points['ID'] == i[0], 'geometry'].values[0]
            point2 = all_points.loc[all_points['ID'] == i[1], 'geometry'].values[0]
            length = T_metric[i[0]][i[1]]['distance'] / 1000
            cost = length * LV_base_cost
            geom = LineString([point1, point2])
            grid_final = pd.concat([grid_final, gpd.GeoDataFrame({'ID': [counter], 'geometry': [geom], 'Length [km]': [length], 'Cost [euro]': [cost]})], ignore_index=True)

            counter += 1
        grid_final.crs = crs
        grid_final.to_file(os.path.join(dir_cluster, 'grid_final_cut.shp'))
        LV_grid = pd.concat([LV_grid, grid_final], ignore_index=True)

        all_points[all_points['substations'] == True].to_file(os.path.join(dir_cluster, 'secondary_substations.shp'))

        clusters_list = pd.DataFrame(columns=['Cluster', 'Sub_cluster', 'Population', 'Load [kW]'])
        normalization = 30 * 9.25

        for i in range(int(number_clusters)):
            subset = clustered_points[clustered_points['Cluster'] == i]
            if len(subset) == 1:
                LV_grid_length = 0
                LV_grid_cost = 0
                max_length = 0
                try:
                    sum_pop = subset['cons (kWh/'].sum()
                    load = sum_pop / normalization
                except:
                    sum_pop = subset['Population'].sum()
                    load = sum_pop * load_capita
                MV_LV_substations = pd.concat([MV_LV_substations, subset], ignore_index=True)
                MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'LV_length'] = 0
                MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'max_distance'] = 0
                MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'MV_Power'] = load
            else:
                lv_length_values = MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'LV_length'].values
                max_distance_values = MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'max_distance'].values
                
                # Check if lv_length_values is empty
                if len(lv_length_values) > 0:
                    LV_grid_length = float(lv_length_values[0]) / 1000
                    LV_grid_cost = LV_grid_length * LV_base_cost
                else:
                    st.write(f"No LV length found for cluster {i}. Skipping cost calculation for this cluster.")
                    LV_grid_length = 0
                    LV_grid_cost = 0  # Default to zero if no LV length found
                
                # Check if max_distance_values is empty
                if len(max_distance_values) > 0:
                    max_length = float(max_distance_values[0]) / 1000
                else:
                    st.write(f"No max distance found for cluster {i}. Defaulting max length to zero.")
                    max_length = 0



                try:
                    sum_pop = subset['cons (kWh/'].sum()
                    load = sum_pop / normalization * 0.35
                except:
                    sum_pop = subset['Population'].sum()
                    load = sum_pop * load_capita * 0.35 # coincidence_factor(sum_pop, pop_per_household)
            try:
                ID_substation = int(MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'ID'].values[0])
            except IndexError:
                print(f"No substation ID found for cluster {i}")
            except Exception as e:
                print(f"An error occurred: {e}")

            data = np.array([[int(clus), int(i), sum_pop, load, LV_grid_length, LV_grid_cost, max_length]])
            df2 = pd.DataFrame(data, columns=['Cluster', 'Sub_cluster', 'Population', 'Load [kW]', 'Grid_Length [km]', 'Grid Cost [euro]', 'Max length [km]'])
            clusters_list = pd.concat([clusters_list, df2], ignore_index=True)
            MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'MV_Power'] = load
            MV_LV_substations.loc[MV_LV_substations['Cluster'] == i, 'Population'] = sum_pop
            MV_LV_substations.to_file(os.path.join(dir_cluster, 'secondary_substations.shp'))
        MV_LV_substations['Cluster2'] = MV_LV_substations['Cluster']
        MV_LV_substations['Cluster'] = clus
        secondary_substations = pd.concat([secondary_substations, MV_LV_substations], ignore_index=True)
        st.write(secondary_substations.drop(columns='geometry').head())
        substation_data = pd.read_csv(os.path.join(gisele_dir, 'data', '0_configuration_files', ss_data))
        st.write("substations:")
        clusters_list = categorize_substation(clusters_list, substation_data)
        clusters_list['Population'] = [ceil(i) for i in clusters_list['Population']]
        clusters_list.to_csv(os.path.join(dir_cluster, 'LV_networks_resume.csv'), index=False)
        LV_resume = pd.concat([LV_resume, clusters_list], ignore_index=True)
        all_houses = pd.concat([all_houses, clustered_points], ignore_index=True)
        LV_resume = pd.concat([LV_resume, clusters_list], ignore_index=True)
        terminal_MV_nodes = MV_LV_substations['ID'].to_list()

        if len(terminal_MV_nodes) > 1:
            for i in tree_final_copy.edges:
                if tree_final.has_edge(*i):
                    tree_final_copy[i[0]][i[1]]['weight'] *= 1

            tree_MV = steiner_tree(tree_final_copy, terminal_MV_nodes)
            grid_MV = gpd.GeoDataFrame()
            path = list(tree_MV.edges)
            counter = 0
            for i in path:
                point1 = all_points.loc[all_points['ID'] == i[0], 'geometry'].values[0]
                point2 = all_points.loc[all_points['ID'] == i[1], 'geometry'].values[0]
                id1 = int(all_points.loc[all_points['ID'] == i[0], 'ID'])
                id2 = int(all_points.loc[all_points['ID'] == i[1], 'ID'])
                length = T_metric[i[0]][i[1]]['distance']
                cost = length * LV_base_cost
                geom = LineString([point1, point2])
                grid_MV = grid_MV.append(gpd.GeoDataFrame({'ID1': [id1], 'ID2': [id2], 'geometry': [geom], 'Length': [length], 'Cost': [cost]}))
                counter += 1
            grid_MV.crs = crs
            grid_MV['Cluster'] = clus
            grid_MV.to_file(os.path.join(dir_cluster, 'grid_MV.shp'))
            MV_grid = pd.concat([MV_grid, grid_MV], ignore_index=True)

    LV_resume.to_csv(os.path.join(gisele_dir, dir_output, 'LV_resume.csv'))
    LV_grid.to_file(os.path.join(gisele_dir, dir_output, 'LV_grid.shp'))
    secondary_substations.to_file(os.path.join(gisele_dir, dir_output, 'secondary_substations.shp'))
    all_houses.to_file(os.path.join(gisele_dir, dir_output, 'final_users.shp'))
    if not MV_grid.empty:
        MV_grid.to_file(os.path.join(gisele_dir, dir_output, 'MV_grid.shp'), index=False)

    secondary_substations['Substation'] = 1
    secondary_substations['Weight'] = 3
    secondary_substations['Type'] = 'Secondary Substation'
    terminal_MV_nodes = secondary_substations.ID.to_list()
    grid_of_points_GDF.drop(grid_of_points_GDF[grid_of_points_GDF['ID'].isin(terminal_MV_nodes)].index, axis=0, inplace=True)
    grid_of_points_GDF = pd.concat([grid_of_points_GDF, secondary_substations], ignore_index=True)
    grid_of_points_GDF[['X', 'Y', 'ID', 'Population', 'Elevation', 'Weight', 'geometry', 'Land_cover', 'Cluster', 'MV_Power', 'Substation', 'Type']].to_csv(os.path.join(gisele_dir, dir_input, 'weighted_grid_of_points_with_ss_and_roads.csv'), index=False)

    return LV_grid, MV_grid, secondary_substations, all_houses
    
def show():

    # Example parameters
    parameters = {
        "crs": "EPSG:4326",  # Example CRS
        "country": "example_country",  # Example country
        "resolution": 100,  # Example resolution
        "load_capita": 100,  # Example load per capita
        "pop_per_household": 5,  # Example population per household
        "road_coef": 1.5,  # Example road coefficient
        "case_study": "example_case_study",  # Example case study
        "LV_distance": 500,  # Example LV distance
        "ss_data": "ss_data_evn.csv",  # Example SS data
        "landcover_option": "ESACCI",  # Example land cover option
        "gisele_dir": "/mount/src/gisele",  # Example GISELE directory
        "roads_weight": 2,  # Example roads weight
        "run_genetic": True,  # Example genetic algorithm flag
        "max_length_segment": 1000,  # Example max length segment
        "simplify_coef": 0.05,  # Example simplify coefficient
        "crit_dist": 100,  # Example critical distance
        "LV_base_cost": 10000,  # Example LV base cost
        "population_dataset_type": "mit"  # Example population dataset type
    }
    path_to_clusters = os.path.join(parameters["gisele_dir"], 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')
    Clusters = gpd.read_file(path_to_clusters)

    # Run the optimization
    LV_grid, MV_grid, secondary_substations, all_houses = optimize(
        parameters["crs"], parameters["country"], parameters["resolution"], parameters["load_capita"],
        parameters["pop_per_household"], parameters["road_coef"], Clusters, parameters["case_study"],
        parameters["LV_distance"], parameters["ss_data"], parameters["landcover_option"], parameters["gisele_dir"],
        parameters["roads_weight"], parameters["run_genetic"], parameters["max_length_segment"],
        parameters["simplify_coef"], parameters["crit_dist"], parameters["LV_base_cost"],
        parameters["population_dataset_type"]
    )

    # Display the results in Streamlit
    st.write("LV_grid:")
    if not LV_grid.empty:
        st.write(LV_grid.drop(columns='geometry').head())
    else:
        st.write("LV_grid is empty or not generated correctly.")

    st.write("MV_grid:")
    if not MV_grid.empty:
        st.write(MV_grid.drop(columns='geometry').head())
    else:
        st.write("MV_grid is empty or not generated correctly.")

    st.write("Secondary Substations:")
    if not secondary_substations.empty:
        st.write(secondary_substations.drop(columns='geometry').head())
    else:
        st.write("Secondary Substations are empty or not generated correctly.")

    st.write("All Houses:")
    if not all_houses.empty:
        st.write(all_houses.drop(columns='geometry').head())
    else:
        st.write("All Houses dataset is empty or not generated correctly.")
