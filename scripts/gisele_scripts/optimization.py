import os
import geopandas as gpd
import pandas as pd
import rasterio
import streamlit as st
from shapely.geometry import Point
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np

def reproject_raster(input_raster, dst_crs):
    """
    Reproject a raster to a different CRS using rasterio.
    Returns the reprojected raster in memory.
    """
    print(f"Reprojecting raster {input_raster} to CRS {dst_crs}")
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
            print(f"Reprojection completed for {input_raster}")
            return memfile.open()

def sample_raster(raster, coords):
    """
    Sample values from a raster file at the specified coordinates.
    """
    print(f"Sampling raster at {len(coords)} coordinates.")
    values = []
    for val in raster.sample(coords):
        values.append(val[0])
    print("Sampling completed.")
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

def optimize(crs, country, resolution, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance, ss_data,
             landcover_option, gisele_dir, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist, LV_base_cost, population_dataset_type):

    print("Starting optimization...")
    
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
    print(f"Reading grid of points from {grid_of_points_path}")
    grid_of_points = pd.read_csv(grid_of_points_path)
    grid_of_points_GDF = gpd.GeoDataFrame(grid_of_points, geometry=gpd.points_from_xy(grid_of_points.X, grid_of_points.Y), crs=crs)
    st.write("Initial grid of points (without geometry):")
    st.write(grid_of_points_GDF.drop(columns='geometry').head())

    # Drop duplicate Elevation column if exists
    if 'Elevation.1' in grid_of_points_GDF.columns:
        print("Dropping duplicate Elevation column.")
        grid_of_points_GDF.drop(columns=['Elevation.1'], inplace=True)

    Starting_node = int(grid_of_points_GDF['ID'].max() + 1)
    print(f"Starting node set to {Starting_node}")

    LV_resume = pd.DataFrame()
    LV_grid = gpd.GeoDataFrame()
    MV_grid = gpd.GeoDataFrame()
    secondary_substations = gpd.GeoDataFrame()
    all_houses = gpd.GeoDataFrame()

    # Load population data
    st.write("Loading population data...")
    print(f"Reading population data from {population_points_path}")
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

        # Clip population points to the study area
        grid_of_points = gpd.clip(Population, area_buffered)
        grid_of_points['X'] = grid_of_points.geometry.x
        grid_of_points['Y'] = grid_of_points.geometry.y
        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        # Clip roads points and lines to the study area
        print("Clipping roads data to study area...")
        road_points = gpd.read_file(roads_points_path)
        road_points = gpd.clip(road_points, area_buffered)
        road_lines = gpd.read_file(roads_lines_path)
        road_lines = road_lines[(road_lines['ID1'].isin(road_points.ID.to_list()) & road_lines['ID2'].isin(road_points.ID.to_list()))]

        # Reproject rasters to the specified CRS on the fly
        print("Reprojecting rasters...")
        Elevation = reproject_raster(os.path.join(dir_input_1, 'elevation', 'Elevation.tif'), crs)
        Slope = reproject_raster(os.path.join(dir_input_1, 'slope', 'slope.tif'), crs)
        LandCover = reproject_raster(os.path.join(dir_input_1, 'landcover', 'LandCover.tif'), crs)

        # Populate the grid of points with raster data
        print("Sampling rasters...")
        coords = [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)]
        grid_of_points['Population'] = sample_raster(Elevation, coords)  # Placeholder, replace with actual population sampling logic
        grid_of_points['Elevation'] = sample_raster(Elevation, coords)
        grid_of_points['Slope'] = sample_raster(Slope, coords)
        grid_of_points['Land_cover'] = sample_raster(LandCover, coords)
        grid_of_points['Protected_area'] = ['FALSE' for _ in range(len(coords))]

        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        # Backbone finding and other processing...
        # Ensure the necessary columns are populated

    # Add missing columns to grid_of_points_GDF
    print("Checking and adding missing columns to grid_of_points_GDF...")
    if 'Population' not in grid_of_points_GDF.columns:
        grid_of_points_GDF['Population'] = np.nan
    if 'Land_cover' not in grid_of_points_GDF.columns:
        grid_of_points_GDF['Land_cover'] = np.nan
    if 'Cluster' not in grid_of_points_GDF.columns:
        grid_of_points_GDF['Cluster'] = np.nan
    if 'MV_Power' not in grid_of_points_GDF.columns:
        grid_of_points_GDF['MV_Power'] = np.nan
    if 'Substation' not in grid_of_points_GDF.columns:
        grid_of_points_GDF['Substation'] = np.nan

    # Check columns present in grid_of_points_GDF
    print("Final columns in grid_of_points_GDF:")
    st.write(f"Final columns in grid_of_points_GDF: {grid_of_points_GDF.columns.tolist()}")

    # Ensure GeoDataFrames are not empty and contain valid geometry before saving
    if not LV_grid.empty and LV_grid.geometry.notnull().all():
        print("Saving LV_grid...")
        LV_grid.to_file(os.path.join(dir_output, 'LV_grid'))
    else:
        st.error("LV_grid is empty or has invalid geometries.")
        print("LV_grid is empty or has invalid geometries.")

    if not secondary_substations.empty and secondary_substations.geometry.notnull().all():
        print("Saving secondary_substations...")
        secondary_substations.to_file(os.path.join(dir_output, 'secondary_substations'))
    else:
        st.error("secondary_substations is empty or has invalid geometries.")
        print("secondary_substations is empty or has invalid geometries.")

    if not all_houses.empty and all_houses.geometry.notnull().all():
        print("Saving all_houses...")
        all_houses.to_file(os.path.join(dir_output, 'final_users'))
    else:
        st.error("all_houses is empty or has invalid geometries.")
        print("all_houses is empty or has invalid geometries.")

    if not MV_grid.empty and MV_grid.geometry.notnull().all():
        print("Saving MV_grid...")
        MV_grid.to_file(os.path.join(dir_output, 'MV_grid'), index=False)
    else:
        st.warning("MV_grid is empty or has invalid geometries.")
        print("MV_grid is empty or has invalid geometries.")

    # Save the final grid with secondary substations and roads
    final_grid_path = os.path.join(dir_input, 'grid_of_points', 'weighted_grid_of_points_with_ss_and_roads.csv')
    if not grid_of_points_GDF.empty and grid_of_points_GDF.geometry.notnull().all():
        # Check and log missing columns before saving
        required_columns = ['X', 'Y', 'ID', 'Population', 'Elevation', 'Weight', 'geometry', 'Land_cover', 'Cluster', 'MV_Power', 'Substation', 'Type']
        missing_columns = [col for col in required_columns if col not in grid_of_points_GDF.columns]

        if missing_columns:
            st.error(f"Missing columns in grid_of_points_GDF: {missing_columns}")
            print(f"Missing columns in grid_of_points_GDF: {missing_columns}")
        else:
            print(f"Saving final grid of points to {final_grid_path}")
            grid_of_points_GDF[required_columns].to_csv(final_grid_path, index=False)
    else:
        st.error("Final grid is empty or has invalid geometries.")
        print("Final grid is empty or has invalid geometries.")

    return LV_grid, MV_grid, secondary_substations, all_houses

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

def show():
    st.title("Optimization")

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
        "ss_data": "ss_data_evn",  # Example SS data
        "landcover_option": "ESACCI",  # Example land cover option
        "gisele_dir": "/mount/src/gisele",  # Example GISELE directory
        "roads_weight": 2,  # Example roads weight
        "run_genetic": True,  # Example genetic algorithm flag
        "max_length_segment": 1000,  # Example max length segment
        "simplify_coef": 0.05,  # Example simplify coefficient
        "crit_dist": 100,  # Example critical distance
        "LV_base_cost": 10000,  # Example LV base cost
        "population_dataset_type": "raster"  # Example population dataset type
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

    # Display the results
    if 'geometry' in LV_grid.columns:
        st.write(LV_grid.drop(columns='geometry').head())
    else:
        st.write(LV_grid.head())

    if 'geometry' in MV_grid.columns:
        st.write(MV_grid.drop(columns='geometry').head())
    else:
        st.write(MV_grid.head())

    if 'geometry' in secondary_substations.columns:
        st.write(secondary_substations.drop(columns='geometry').head())
    else:
        st.write(secondary_substations.head())

    if 'geometry' in all_houses.columns:
        st.write(all_houses.drop(columns='geometry').head())
    else:
        st.write(all_houses.head())
