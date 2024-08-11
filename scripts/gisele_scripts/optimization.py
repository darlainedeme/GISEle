import os
import time
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
import rasterio
import streamlit as st
from shapely.geometry import Point, LineString, MultiPoint
from geneticalgorithm_github import geneticalgorithm as ga
from Steiner_tree_code import steiner_tree, metric_closure
from sklearn.cluster import AgglomerativeClustering

def optimize(crs, country, resolution, load_capita, pop_per_household, road_coef, Clusters, case_study, LV_distance, ss_data,
             landcover_option, gisele_dir, roads_weight, run_genetic, max_length_segment, simplify_coef, crit_dist, LV_base_cost, population_dataset_type):
    
    dir_input_1 = os.path.join(gisele_dir, r'data', '2_downloaded_input_data')
    dir_input = os.path.join(gisele_dir, r'data', '4_intermediate_output')
    dir_output = os.path.join(gisele_dir, r'data', '5_final_output')
    grid_of_points_path = os.path.join(dir_input, 'grid_of_points', 'weighted_grid_of_points_with_roads.csv')
    population_points_path = os.path.join(gisele_dir, r'data', '2_downloaded_input_data', 'buildings', 'mit', 'points_clipped', 'points_clipped.shp')
    roads_points_path = os.path.join(gisele_dir, r'data', '2_downloaded_input_data', 'roads', 'roads_points.shp')
    roads_lines_path = os.path.join(gisele_dir, r'data', '2_downloaded_input_data', 'roads', 'roads_lines.shp')
    ss_data_path = os.path.join(gisele_dir, r'data', '0_configuration_files', ss_data)
    
    # Load initial grid of points with roads
    st.write("Loading grid of points with roads...")
    grid_of_points = pd.read_csv(grid_of_points_path)
    st.write(f"Columns in grid_of_points: {grid_of_points.columns.tolist()}")
    grid_of_points_GDF = gpd.GeoDataFrame(grid_of_points, geometry=gpd.points_from_xy(grid_of_points.X, grid_of_points.Y), crs=crs)
    
    st.write(f"Initial GeoDataFrame: {grid_of_points_GDF.head()}")
    
    Starting_node = int(grid_of_points_GDF['ID'].max() + 1)
    LV_resume = pd.DataFrame()
    LV_grid = gpd.GeoDataFrame()
    MV_grid = gpd.GeoDataFrame()
    secondary_substations = gpd.GeoDataFrame()
    all_houses = gpd.GeoDataFrame()

    # Load population data
    st.write("Loading population data...")
    Population = gpd.read_file(population_points_path)

    for index, row in Clusters.iterrows():
        dir_cluster = os.path.join(gisele_dir, r'data', '4_intermediate_output', 'optimization', str(row["cluster_ID"]))
        os.makedirs(dir_cluster, exist_ok=True)
        os.makedirs(os.path.join(dir_cluster, 'grids'), exist_ok=True)

        area = row['geometry']
        area_buffered = area.buffer(resolution)

        # Clip population points to the study area
        grid_of_points = gpd.clip(Population, area_buffered)
        grid_of_points['X'] = [point['geometry'].xy[0][0] for _, point in grid_of_points.iterrows()]
        grid_of_points['Y'] = [point['geometry'].xy[1][0] for _, point in grid_of_points.iterrows()]
        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        # Clip roads points and lines to the study area
        road_points = gpd.read_file(roads_points_path)
        road_points = gpd.clip(road_points, area_buffered)
        road_lines = gpd.read_file(roads_lines_path)
        road_lines = road_lines[(road_lines['ID1'].isin(road_points.ID.to_list()) & road_lines['ID2'].isin(road_points.ID.to_list()))]

        # Load rasters for the specific region
        Elevation = rasterio.open(os.path.join(dir_input_1, 'elevation', f'Elevation_{crs}.tif'))
        Slope = rasterio.open(os.path.join(dir_input_1, 'slope', f'Slope_{crs}.tif'))
        LandCover = rasterio.open(os.path.join(dir_input_1, 'landcover', f'LandCover_{crs}.tif'))

        # Populate the grid of points with raster data
        coords = [(x, y) for x, y in zip(grid_of_points.X, grid_of_points.Y)]
        grid_of_points['Population'] = [x[0] for x in Population.sample(coords)]
        grid_of_points['Elevation'] = [x[0] for x in Elevation.sample(coords)]
        grid_of_points['Slope'] = [x[0] for x in Slope.sample(coords)]
        grid_of_points['Land_cover'] = [x[0] for x in LandCover.sample(coords)]
        grid_of_points['Protected_area'] = ['FALSE' for _ in LandCover.sample(coords)]

        grid_of_points.to_file(os.path.join(dir_cluster, 'points.shp'))

        # Backbone finding and other processing...
        # Similar steps to the original code, using the new paths for saving files

    # Check columns present in grid_of_points_GDF
    st.write(f"Final columns in grid_of_points_GDF: {grid_of_points_GDF.columns.tolist()}")
    
    # Ensure GeoDataFrames are not empty and contain valid geometry before saving
    if not LV_grid.empty and LV_grid.geometry.notnull().all():
        LV_grid.to_file(os.path.join(dir_output, 'LV_grid'))
    else:
        st.error("LV_grid is empty or has invalid geometries.")

    if not secondary_substations.empty and secondary_substations.geometry.notnull().all():
        secondary_substations.to_file(os.path.join(dir_output, 'secondary_substations'))
    else:
        st.error("secondary_substations is empty or has invalid geometries.")

    if not all_houses.empty and all_houses.geometry.notnull().all():
        all_houses.to_file(os.path.join(dir_output, 'final_users'))
    else:
        st.error("all_houses is empty or has invalid geometries.")

    if not MV_grid.empty and MV_grid.geometry.notnull().all():
        MV_grid.to_file(os.path.join(dir_output, 'MV_grid'), index=False)
    else:
        st.warning("MV_grid is empty or has invalid geometries.")

    # Save the final grid with secondary substations and roads
    final_grid_path = os.path.join(dir_input, 'grid_of_points', 'weighted_grid_of_points_with_ss_and_roads.csv')
    if not grid_of_points_GDF.empty and grid_of_points_GDF.geometry.notnull().all():
        # Check and log missing columns before saving
        required_columns = ['X', 'Y', 'ID', 'Population', 'Elevation', 'Weight', 'geometry', 'Land_cover', 'Cluster', 'MV_Power', 'Substation', 'Type']
        missing_columns = [col for col in required_columns if col not in grid_of_points_GDF.columns]
        
        if missing_columns:
            st.error(f"Missing columns in grid_of_points_GDF: {missing_columns}")
        else:
            grid_of_points_GDF[required_columns].to_csv(final_grid_path, index=False)
    else:
        st.error("Final grid is empty or has invalid geometries.")

    return LV_grid, MV_grid, secondary_substations, all_houses

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
    st.write("LV_grid:")
    st.write(LV_grid.head())
    
    st.write("MV_grid:")
    st.write(MV_grid.head())
    
    st.write("Secondary Substations:")
    st.write(secondary_substations.head())
    
    st.write("All Houses:")
    st.write(all_houses.head())