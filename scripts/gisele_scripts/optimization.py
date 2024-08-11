import os
import geopandas as gpd
import pandas as pd
import rasterio
import streamlit as st
from shapely.geometry import Point, LineString
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
from networkx import Graph

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

def genetic2(clustered_points,points_new_graph,distance_matrix,n_clusters,graph):
    clustered_points.reset_index(drop=True,inplace=True)
    lookup_edges = [i for i in graph.edges]
    dim = len(lookup_edges)-1
    dist_matrix_df = pd.DataFrame(distance_matrix,columns = [i for i in points_new_graph['ID']],index = [i for i in points_new_graph['ID']])
    #initial_solution = np.array(clustered_points['Cluster'].to_list())
    varbound=np.array([[0,dim]]*(n_clusters-1))
    #lookup_edges = [i for i in graph.edges([190,184,29,171,202,201,206,205,209,210,22,221,231,127,235,244,230,229,228,220,210,227,215,216,226,234,204,198,197,56,194,191,179])]
    #dim = len(lookup_edges) - 1
    def fitness(X):
        T = graph.copy()
        length_deleted_lines = 0
        count=Counter(X)
        for i in count:
            if count[i]>1:
                return 1000000 # this is in case it is trying to cut the same branch more than once
        for i in X:
            delete_edge = lookup_edges[int(i)]
            length_deleted_lines += graph[lookup_edges[int(i)][0]][lookup_edges[int(i)][1]]['weight']['distance']
            T.remove_edge(*delete_edge)
        islands = [c for c in nx.connected_components(T)]
        cost = 0
        penalty = 0
        for i in range(len(islands)):
            subgraph = T.subgraph(islands[i])
            subset_IDs = [i for i in subgraph.nodes]
            population =points_new_graph[points_new_graph['ID'].isin(subset_IDs)]['Population'].sum()
            power = population*0.7*0.3
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
            if max_dist >1000:
                penalty = penalty+ 50000+ (max_dist-500)*25

        cost = cost - length_deleted_lines/1000*10000 # divided by 1000 for m->km and the multiplied by 10000euro/km

        if penalty>0:
            return penalty
        else:
            return cost

    algorithm_param = {'max_num_iteration': 1000, 'population_size': 40, 'mutation_probability': 0.1,
                       'elit_ratio': 0.025, 'crossover_probability': 0.6, 'parents_portion': 0.25,
                       'crossover_type': 'one_point', 'max_iteration_without_improv': 100}
    model = ga(function=fitness, dimension=n_clusters-1, variable_type='int', variable_boundaries=varbound,
             function_timeout=10000,algorithm_parameters=algorithm_param)
    model.run()
    cut_edges = model.best_variable
    T=graph.copy()
    for i in cut_edges:
        delete_edge = lookup_edges[int(i)]
        T.remove_edge(*delete_edge)

    islands = [c for c in nx.connected_components(T)]
    for i in range(len(islands)):
        subgraph = T.subgraph(islands[i])
        subset_IDs = [i for i in subgraph.nodes]
        clustered_points.loc[clustered_points['ID'].isin(subset_IDs),'Cluster']=i

    return clustered_points, cut_edges

# Main optimization function
def optimize(crs, country, resolution, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance, ss_data,
             landcover_option, gisele_dir, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist, LV_base_cost, population_dataset_type):

    print("Starting optimization...")

    # Set up directories
    dir_input_1 = os.path.join(gisele_dir, 'data', '2_downloaded_input_data')
    dir_input = os.path.join(gisele_dir, 'data', '4_intermediate_output')
    dir_output = os.path.join(gisele_dir, 'data', '5_final_output')
    grid_of_points_path = os.path.join(dir_input, 'grid_of_points', 'weighted_grid_of_points_with_roads.csv')
    population_points_path = os.path.join(gisele_dir, 'data', '2_downloaded_input_data', 'buildings', 'mit', 'points_clipped', 'points_clipped.shp')
    roads_points_path = os.path.join(gisele_dir, 'data', '2_downloaded_input_data', 'roads', 'roads_points.shp')
    roads_lines_path = os.path.join(gisele_dir, 'data', '2_downloaded_input_data', 'roads', 'roads_lines.shp')
    ss_data_path = os.path.join(gisele_dir, 'data', '0_configuration_files', ss_data)

    # Load initial grid of points with roads
    st.write("Loading grid of points with roads...")
    grid_of_points = pd.read_csv(grid_of_points_path)
    grid_of_points_GDF = gpd.GeoDataFrame(grid_of_points, geometry=gpd.points_from_xy(grid_of_points.X, grid_of_points.Y), crs=crs)
    st.write("Initial grid of points (without geometry):")
    st.write(grid_of_points_GDF.drop(columns='geometry').head())

    # Drop duplicate Elevation column if exists
    if 'Elevation.1' in grid_of_points_GDF.columns:
        grid_of_points_GDF.drop(columns=['Elevation.1'], inplace=True)

    # Initialize GeoDataFrames for results
    LV_grid = gpd.GeoDataFrame()
    MV_grid = gpd.GeoDataFrame()
    secondary_substations = gpd.GeoDataFrame()
    all_houses = gpd.GeoDataFrame()

    # Load population data
    st.write("Loading population data...")
    Population = gpd.read_file(population_points_path)
    st.write("Population data (without geometry):")
    st.write(Population.drop(columns='geometry').head())

    for index, row in Clusters.iterrows():
        print(f"Processing cluster {row['cluster_ID']}...")
        dir_cluster = os.path.join(gisele_dir, 'data', '4_intermediate_output', 'optimization', str(row["cluster_ID"]))
        os.makedirs(dir_cluster, exist_ok=True)
        os.makedirs(os.path.join(dir_cluster, 'grids'), exist_ok=True)

        area = row['geometry']
        area_buffered = area.buffer(resolution)

        # Create or clip grid of points
        if population_dataset_type == 'mit':
            grid_of_points = create_grid(crs, resolution, area)
            Population_raster = rasterio.open(os.path.join(dir_input_1, f'Population_{crs}.tif'))
            grid_of_points['Population'] = sample_raster(Population_raster, grid_of_points[['X', 'Y']].values)
        else:
            grid_of_points = gpd.clip(Population, area_buffered)
            grid_of_points['X'] = grid_of_points.geometry.x
            grid_of_points['Y'] = grid_of_points.geometry.y

        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        # Clip roads points and lines to the study area
        road_points = gpd.read_file(roads_points_path)
        road_points = gpd.clip(road_points, area_buffered)
        road_lines = gpd.read_file(roads_lines_path)
        road_lines = road_lines[(road_lines['ID1'].isin(road_points.ID.to_list()) & road_lines['ID2'].isin(road_points.ID.to_list()))]

        # Reproject and sample raster data
        Elevation = reproject_raster(os.path.join(dir_input_1, 'elevation', 'Elevation.tif'), crs)
        Slope = reproject_raster(os.path.join(dir_input_1, 'slope', 'slope.tif'), crs)
        LandCover = reproject_raster(os.path.join(dir_input_1, 'landcover', 'LandCover.tif'), crs)
        coords = [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)]
        grid_of_points['Elevation'] = sample_raster(Elevation, coords)
        grid_of_points['Slope'] = sample_raster(Slope, coords)
        grid_of_points['Land_cover'] = sample_raster(LandCover, coords)
        grid_of_points['Protected_area'] = ['FALSE' for _ in range(len(coords))]

        # Finding the backbone and creating LV grid
        Population_clus = grid_of_points[grid_of_points['Population'] > 0]
        Population_clus['ID'] = range(Starting_node, Starting_node + len(Population_clus))
        Starting_node += len(Population_clus)

        road_points['Population'] = 0
        road_points['pop_bool'] = 0

        # Connect population points to roads
        for i, pop in Population_clus.iterrows():
            point = pop.geometry
            nearest_geoms = nearest_points(point, MultiPoint(road_points.geometry))
            closest_road_point = nearest_geoms[1]
            road_points.loc[road_points.geometry == closest_road_point, 'Population'] = pop['Population']
            road_points.loc[road_points.geometry == closest_road_point, 'pop_bool'] = 1

        # Create graph for roads
        graph = Graph()
        for _, row in road_lines.iterrows():
            id1, id2 = row['ID1'], row['ID2']
            graph.add_edge(id1, id2, weight=row['length'] * roads_weight)

        # Connect unconnected components
        graph, road_lines = connect_unconnected_graph(graph, road_lines, road_points, weight=5)

        # Create backbone using Steiner tree
        populated_points = road_points[road_points['pop_bool'] == 1]
        terminal_nodes = list(populated_points['ID'])
        tree = steiner_tree(graph, terminal_nodes)

        # Build LV grid
        LV_grid_cluster = gpd.GeoDataFrame()
        for i, (n1, n2) in enumerate(tree.edges):
            p1 = road_points.loc[road_points['ID'] == n1, 'geometry'].values[0]
            p2 = road_points.loc[road_points['ID'] == n2, 'geometry'].values[0]
            line = LineString([p1, p2])
            LV_grid_cluster = LV_grid_cluster.append({'ID': i, 'geometry': line}, ignore_index=True)

        LV_grid = LV_grid.append(LV_grid_cluster, ignore_index=True)
        LV_grid.crs = crs
        LV_grid.to_file(os.path.join(dir_cluster, 'LV_backbone.shp'))

        # Connect houses
        road_points_backbone = road_points[road_points['ID'].isin(tree.nodes)]
        road_points_backbone['Population'] = 0
        road_points_backbone['pop_bool'] = 0

        # Implement connection logic for houses
        index = range(road_points_backbone['ID'].max() + 1, road_points_backbone['ID'].max() + 1 + len(Population_clus))
        Population_clus['ind'] = index
        Population_clus.set_index('ind', inplace=True, drop=True)

        all_points = road_points_backbone.append(Population_clus)
        new_graph, new_lines = delaunay_test(tree, all_points, road_lines)

        LV_grid_cluster = gpd.GeoDataFrame()
        for i, (n1, n2) in enumerate(new_graph.edges):
            p1 = all_points.loc[all_points['ID'] == n1, 'geometry'].values[0]
            p2 = all_points.loc[all_points['ID'] == n2, 'geometry'].values[0]
            line = LineString([p1, p2])
            LV_grid_cluster = LV_grid_cluster.append({'ID': i, 'geometry': line}, ignore_index=True)

        LV_grid = LV_grid.append(LV_grid_cluster, ignore_index=True)
        LV_grid.crs = crs
        LV_grid.to_file(os.path.join(dir_cluster, 'LV_final.shp'))

        # The rest of the process including MV grid and secondary substations creation follows here...
        # This includes clustering, Steiner tree for MV grid, and updating relevant GeoDataFrames.

    # Finalizing and saving outputs
    LV_grid.to_file(os.path.join(dir_output, 'LV_grid.shp'))
    MV_grid.to_file(os.path.join(dir_output, 'MV_grid.shp'))
    secondary_substations.to_file(os.path.join(dir_output, 'secondary_substations.shp'))
    all_houses.to_file(os.path.join(dir_output, 'final_users.shp'))

    return LV_grid, MV_grid, secondary_substations, all_houses
    
def show():
    st.title("Optimization")

    parameters = {
        "crs": "EPSG:4326",
        "country": "example_country",
        "resolution": 100,
        "load_capita": 100,
        "pop_per_household": 5,
        "road_coef": 1.5,
        "case_study": "example_case_study",
        "LV_distance": 500,
        "ss_data": "ss_data_evn",
        "landcover_option": "ESACCI",
        "gisele_dir": "/mount/src/gisele",
        "roads_weight": 2,
        "run_genetic": True,
        "max_length_segment": 1000,
        "simplify_coef": 0.05,
        "crit_dist": 100,
        "LV_base_cost": 10000,
        "population_dataset_type": "raster"
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
