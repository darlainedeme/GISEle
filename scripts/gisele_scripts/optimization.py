import os
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point, LineString, MultiPoint
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import networkx as nx
from shapely.ops import nearest_points
from scipy.spatial import Delaunay
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from math import ceil
from scipy.spatial.distance import cdist
from geneticalgorithm import geneticalgorithm as ga
import streamlit as st

# Helper Functions

def reproject_raster(input_raster, dst_crs):
    """Reproject a raster file to a new coordinate reference system (CRS)."""
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

def sample_raster(raster, coords):
    """Sample raster values at given coordinates."""
    values = [val[0] for val in raster.sample(coords)]
    return values

def create_grid(crs, resolution, study_area):
    """Create a grid of points within a study area."""
    min_x, min_y, max_x, max_y = study_area.bounds
    lon, lat = np.meshgrid(
        np.arange(min_x, max_x, resolution),
        np.arange(min_y, max_y, resolution)
    )
    df = pd.DataFrame({'X': lon.ravel(), 'Y': lat.ravel()})
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=crs)
    return gpd.clip(geo_df, study_area)

def metric_closure(G, weight='weight'):
    """Return the metric closure of a graph."""
    M = nx.Graph()
    for u, (distance, path) in nx.all_pairs_dijkstra(G, weight=weight):
        for v in distance:
            if u != v:
                M.add_edge(u, v, distance=distance[v], path=path[v])
    return M

def steiner_tree(G, terminal_nodes, weight='weight'):
    """Return an approximation to the minimum Steiner tree of a graph."""
    M = metric_closure(G, weight=weight)
    H = M.subgraph(terminal_nodes)
    mst_edges = nx.minimum_spanning_edges(H, weight='distance', data=True)
    edges = chain.from_iterable(nx.utils.pairwise(d['path']) for u, v, d in mst_edges)
    return G.edge_subgraph(edges)

def connect_unconnected_graph(graph, lines, points, weight):
    """Connect unconnected components of a graph."""
    if nx.is_connected(graph):
        return graph, lines
    
    islands = [c for c in nx.connected_components(graph)]
    for i, j in combinations(range(len(islands)), 2):
        subgraph_1 = [val for val in islands[i]]
        subgraph_2 = [val for val in islands[j]]
        points_s1 = points.loc[points['ID'].isin(subgraph_1), :]
        points_s2 = points.loc[points['ID'].isin(subgraph_2), :]
        closest_points = nearest_points(
            MultiPoint(points_s1.geometry),
            MultiPoint(points_s2.geometry)
        )
        distance = closest_points[0].distance(closest_points[1])
        id_point1 = int(points.loc[points.geometry == closest_points[0], 'ID'])
        id_point2 = int(points.loc[points.geometry == closest_points[1], 'ID'])
        lines = pd.concat([lines, gpd.GeoDataFrame({
            'ID1': [id_point1], 'ID2': [id_point2],
            'length': [distance], 'geometry': [LineString([closest_points[0], closest_points[1]])]
        })], ignore_index=True)
        graph.add_edge(id_point1, id_point2, weight=distance * weight, length=distance)
    return graph, lines

def distance_2d(df1, df2, x='X', y='Y'):
    """Compute the 2D distance matrix between two sets of points."""
    d1_coordinates = df1[[x, y]].values
    d2_coordinates = df2[[x, y]].values
    return cdist(d1_coordinates, d2_coordinates)

def delaunay_graph(points, max_length=500):
    """Create a Delaunay triangulation graph from a set of points."""
    coords = np.array([[point.xy[0][0], point.xy[1][0]] for point in points.geometry])
    tri = Delaunay(coords)
    edges = set()
    for triangle in tri.simplices:
        for i, j in combinations(triangle, 2):
            if i > j: i, j = j, i
            edges.add((i, j))
    
    graph = nx.Graph()
    for i, j in edges:
        point1, point2 = points.iloc[i].geometry, points.iloc[j].geometry
        length = point1.distance(point2)
        if length < max_length:
            graph.add_edge(i, j, weight=length, length=length)
    return graph

def create_clean_graph(graph, points, terminal_points, T_metric, crs):
    """Clean a graph by reducing it to terminal nodes and intersections."""
    terminal_IDs = terminal_points['ID'].tolist()
    intersection_IDs = [node for node, count in Counter(chain(*graph.edges)).items() if count > 2 and node not in terminal_IDs]
    new_nodes = set(terminal_IDs + intersection_IDs)

    new_graph = nx.Graph()
    for u, v, d in graph.edges(data=True):
        if u in new_nodes and v in new_nodes:
            new_graph.add_edge(u, v, **d)

    new_lines = gpd.GeoDataFrame({
        'geometry': [LineString([points.loc[u].geometry, points.loc[v].geometry]) for u, v in new_graph.edges],
        'length': [d['length'] for u, v, d in new_graph.edges(data=True)]
    }, crs=crs)
    
    new_points = points[points['ID'].isin(new_nodes)]
    return new_lines, new_points, new_graph

# Genetic Algorithm for Network Optimization

def optimize_network(graph, clustered_points, points_new_graph, distance_matrix, n_clusters):
    """Optimize the network using a genetic algorithm."""
    lookup_edges = list(graph.edges)
    dim = len(lookup_edges) - 1
    dist_matrix_df = pd.DataFrame(distance_matrix, index=points_new_graph['ID'], columns=points_new_graph['ID'])
    
    varbound = np.array([[0, dim]] * (n_clusters - 1))
    
    def fitness(X):
        T = graph.copy()
        length_deleted_lines = 0
        count = Counter(X)
        for i in count:
            if count[i] > 1:
                return 1000000  # High penalty for cutting the same branch more than once
        
        for i in X:
            delete_edge = lookup_edges[int(i)]
            if T.has_edge(*delete_edge):  # Check if the edge exists
                length_deleted_lines += T[delete_edge[0]][delete_edge[1]]['weight']
                T.remove_edge(*delete_edge)
            else:
                return 1000000  # High penalty if the edge doesn't exist
        
        islands = list(nx.connected_components(T))
        cost = 0
        for island in islands:
            subgraph = T.subgraph(island)
            population = points_new_graph.loc[subgraph.nodes, 'Population'].sum()
            power = population * 0.7 * 0.3
            cost += calculate_cost(population, power)
            
            max_dist = dist_matrix_df.loc[subgraph.nodes, subgraph.nodes].max().max()
            if max_dist > 1000:
                cost += 50000 + (max_dist - 500) * 25
        
        cost -= length_deleted_lines / 1000 * 10000
        return cost

    algorithm_param = {
        'max_num_iteration': 100,
        'population_size': 20,
        'mutation_probability': 0.05,
        'elit_ratio': 0.02,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50
    }

    model = ga(
        function=fitness, dimension=n_clusters - 1, variable_type='int', 
        variable_boundaries=varbound, algorithm_parameters=algorithm_param
    )
    model.run()

    cut_edges = model.best_variable
    T = graph.copy()
    for i in cut_edges:
        delete_edge = lookup_edges[int(i)]
        if T.has_edge(*delete_edge):
            T.remove_edge(*delete_edge)
    
    islands = list(nx.connected_components(T))
    for i, island in enumerate(islands):
        subgraph = T.subgraph(island)
        clustered_points.loc[clustered_points['ID'].isin(subgraph.nodes), 'Cluster'] = i
    
    return clustered_points, cut_edges

def calculate_cost(population, power):
    """Calculate the cost based on population and power."""
    if power < 25:
        return 1500
    elif power < 50:
        return 2300
    elif power < 100:
        return 3500
    else:
        return 100000

# Main Function

def optimize(crs, country, resolution, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance, ss_data,
             landcover_option, gisele_dir, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist, LV_base_cost, population_dataset_type):
    
    dir_input_1 = os.path.join(gisele_dir, 'data', '2_downloaded_input_data')
    dir_input = os.path.join(gisele_dir, 'data', '4_intermediate_output')
    dir_output = os.path.join(gisele_dir, 'data', '5_final_output')
    
    grid_of_points_path = os.path.join(dir_input, 'grid_of_points', 'weighted_grid_of_points_with_roads.csv')
    population_points_path = os.path.join(dir_input_1, 'buildings', 'mit', 'points_clipped', 'points_clipped.shp')
    roads_points_path = os.path.join(dir_input_1, 'roads', 'roads_points.shp')
    roads_lines_path = os.path.join(dir_input_1, 'roads', 'roads_lines.shp')
    ss_data_path = os.path.join(gisele_dir, 'data', '0_configuration_files', ss_data)

    grid_of_points_GDF = gpd.read_file(grid_of_points_path)
    Population = gpd.read_file(population_points_path)

    LV_resume = pd.DataFrame()
    LV_grid = gpd.GeoDataFrame()
    MV_grid = gpd.GeoDataFrame()
    secondary_substations = gpd.GeoDataFrame()
    all_houses = gpd.GeoDataFrame()

    Starting_node = int(grid_of_points_GDF['ID'].max() + 1)

    for index, row in Clusters.iterrows():
        cluster_id = row["cluster_ID"]
        dir_cluster = os.path.join(dir_input, 'optimization', str(cluster_id))
        os.makedirs(dir_cluster, exist_ok=True)

        area = row['geometry'].buffer(resolution)

        grid_of_points = create_grid(crs, resolution, area)
        grid_of_points['Population'] = sample_raster(
            rasterio.open(os.path.join(dir_input_1, 'population', 'Population.tif')), 
            [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)]
        )
        grid_of_points['ID'] = grid_of_points.index

        road_points = gpd.read_file(roads_points_path)
        road_points = gpd.clip(road_points, area)
        road_lines = gpd.read_file(roads_lines_path)
        road_lines = road_lines[(road_lines['ID1'].isin(road_points.ID.to_list()) & road_lines['ID2'].isin(road_points.ID.to_list()))]

        Elevation = reproject_raster(os.path.join(dir_input_1, 'elevation', 'Elevation.tif'), crs)
        Slope = reproject_raster(os.path.join(dir_input_1, 'slope', 'slope.tif'), crs)
        LandCover = reproject_raster(os.path.join(dir_input_1, 'landcover', 'LandCover.tif'), crs)

        grid_of_points['Elevation'] = sample_raster(Elevation, [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)])
        grid_of_points['Slope'] = sample_raster(Slope, [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)])
        grid_of_points['Land_cover'] = sample_raster(LandCover, [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)])

        Population_clus = grid_of_points[grid_of_points['Population'] > 0]
        Population_clus['ID'] = range(Starting_node, Starting_node + len(Population_clus))
        Population_clus['pop_bool'] = 1
        Starting_node += len(Population_clus)

        road_points['Population'] = 0
        road_points['pop_bool'] = 0

        for i, pop in Population_clus.iterrows():
            nearest_geom = nearest_points(pop.geometry, MultiPoint(road_points.geometry))[1]
            road_points.loc[road_points.geometry == nearest_geom, 'Population'] = pop['Population']
            road_points.loc[road_points.geometry == nearest_geom, 'pop_bool'] = 1

        graph = nx.Graph()
        for _, row in road_lines.iterrows():
            graph.add_edge(row['ID1'], row['ID2'], weight=row['length'] * 1000, length=row['length'] * 1000)

        graph, new_lines = connect_unconnected_graph(graph, road_lines, road_points, weight=5)

        tree = steiner_tree(graph, list(road_points.loc[road_points['pop_bool'] == 1, 'ID']))
        grid_routing = gpd.GeoDataFrame({
            'geometry': [LineString([road_points.loc[road_points['ID'] == i[0], 'geometry'].values[0], 
                                     road_points.loc[road_points['ID'] == i[1], 'geometry'].values[0]]) 
                         for i in tree.edges]
        }, crs=crs)

        new_graph, all_lines = delaunay_graph(Population_clus), road_lines
        new_graph, new_lines = connect_unconnected_graph(new_graph, new_lines, Population_clus, weight=3)

        tree_final = steiner_tree(new_graph, list(Population_clus['ID']))

        # Create metric closure and prepare for clustering
        T_metric = metric_closure(tree_final, weight='length')
        populated_points = Population_clus.loc[Population_clus['pop_bool'] == 1]
        lines_new_graph, points_new_graph, new_graph = create_clean_graph(tree_final, Population_clus, populated_points, T_metric, crs)

        # Distance Matrix
        points_set = points_new_graph['ID'].tolist()
        dist_matrix = np.zeros((len(points_set), len(points_set)))
        for i in range(len(points_set)):
            for j in range(len(points_set)):
                if i != j:
                    dist_matrix[i, j] = T_metric[points_set[i]][points_set[j]]['distance']

        # Clustering
        clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete',
                                             distance_threshold=2 * LV_distance).fit(dist_matrix)
        populated_points['Cluster'] = clustering.labels_

        # Genetic Algorithm Optimization
        if run_genetic:
            clustered_points, cut_edges = optimize_network(new_graph, populated_points, points_new_graph, dist_matrix, populated_points['Cluster'].max() + 1)
        else:
            clustered_points = populated_points
            cut_edges = []
        
        # Final cleanup and save results
        finalize_results(clustered_points, cut_edges, new_graph, points_new_graph, T_metric, dir_cluster, crs, LV_base_cost)

    return LV_grid, MV_grid, secondary_substations, all_houses

def finalize_results(clustered_points, cut_edges, tree_final, points_new_graph, T_metric, dir_cluster, crs, LV_base_cost):
    """Finalize the results after clustering and optimization."""
    islands = [c for c in nx.connected_components(tree_final) if len(c) > 1]
    for i, island in enumerate(islands):
        subgraph = tree_final.subgraph(island)
        LV_grid_length = sum(d['length'] for u, v, d in subgraph.edges(data=True))
        points_new_graph.loc[points_new_graph['ID'].isin(island), 'Cluster'] = i
        best_site = find_best_substation_site(points_new_graph.loc[points_new_graph['ID'].isin(island)], T_metric, LV_grid_length)

        points_new_graph.loc[points_new_graph['ID'] == best_site, 'substations'] = True
        points_new_graph.loc[points_new_graph['ID'] == best_site, 'LV_length'] = LV_grid_length
        points_new_graph.loc[points_new_graph['ID'] == best_site, 'max_distance'] = max(T_metric[best_site][node]['distance'] for node in island)

    save_final_grid(tree_final, points_new_graph, T_metric, dir_cluster, crs, LV_base_cost)

def find_best_substation_site(points_subset, T_metric, LV_grid_length):
    """Find the best site for placing a substation."""
    feasible_sites = points_subset.copy()
    feasible_sites['final_distance'] = float('inf')

    for index, row in points_subset.iterrows():
        total_weighted_distance = sum(T_metric[row['ID']][node]['distance'] for node in points_subset['ID'])
        max_distance = max(T_metric[row['ID']][node]['distance'] for node in points_subset['ID'])
        feasible_sites.at[index, 'final_distance'] = 0.9 * total_weighted_distance + 0.1 * max_distance

    return feasible_sites.loc[feasible_sites['final_distance'].idxmin(), 'ID']

def save_final_grid(tree_final, points_new_graph, T_metric, dir_cluster, crs, LV_base_cost):
    """Save the final LV grid and substation information."""
    grid_final = gpd.GeoDataFrame({
        'geometry': [LineString([points_new_graph.loc[points_new_graph['ID'] == i[0], 'geometry'].values[0], 
                                 points_new_graph.loc[points_new_graph['ID'] == i[1], 'geometry'].values[0]]) 
                     for i in tree_final.edges],
        'Length [km]': [T_metric[i[0]][i[1]]['distance'] / 1000 for i in tree_final.edges],
        'Cost [euro]': [T_metric[i[0]][i[1]]['distance'] / 1000 * LV_base_cost for i in tree_final.edges]
    }, crs=crs)

    grid_final.to_file(os.path.join(dir_cluster, 'grid_final_cut.shp'))

def show():
    """Run the optimization and display the results."""
    parameters = {
        "crs": "EPSG:4326", "country": "example_country", "resolution": 100,
        "load_capita": 100, "pop_per_household": 5, "road_coef": 1.5,
        "case_study": "example_case_study", "LV_distance": 500, "ss_data": "ss_data_evn",
        "landcover_option": "ESACCI", "gisele_dir": "/mount/src/gisele",
        "roads_weight": 2, "run_genetic": True, "max_length_segment": 1000,
        "simplify_coef": 0.05, "crit_dist": 100, "LV_base_cost": 10000,
        "population_dataset_type": "mit"
    }
    
    path_to_clusters = os.path.join(parameters["gisele_dir"], 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')
    Clusters = gpd.read_file(path_to_clusters)

    LV_grid, MV_grid, secondary_substations, all_houses = optimize(
        parameters["crs"], parameters["country"], parameters["resolution"], parameters["load_capita"],
        parameters["pop_per_household"], parameters["road_coef"], Clusters, parameters["case_study"],
        parameters["LV_distance"], parameters["ss_data"], parameters["landcover_option"], parameters["gisele_dir"],
        parameters["roads_weight"], parameters["run_genetic"], parameters["max_length_segment"],
        parameters["simplify_coef"], parameters["crit_dist"], parameters["LV_base_cost"],
        parameters["population_dataset_type"]
    )

    st.write("LV_grid:")
    st.write(LV_grid.head() if not LV_grid.empty else "LV_grid is empty.")
    st.write("MV_grid:")
    st.write(MV_grid.head() if not MV_grid.empty else "MV_grid is empty.")
    st.write("Secondary Substations:")
    st.write(secondary_substations.head() if not secondary_substations.empty else "Secondary Substations are empty.")
    st.write("All Houses:")
    st.write(all_houses.head() if not all_houses.empty else "All Houses dataset is empty.")

if __name__ == "__main__":
    show()
