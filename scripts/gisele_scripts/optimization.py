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
        st.write(f"Clipped grid of points for cluster {row['cluster_ID']} (without geometry):")
        st.write(grid_of_points.drop(columns='geometry').head())

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
    st.write(LV_grid.drop(columns='geometry').head())

    st.write("MV_grid:")
    st.write(MV_grid.drop(columns='geometry').head())

    st.write("Secondary Substations:")
    st.write(secondary_substations.drop(columns='geometry').head())

    st.write("All Houses:")
    st.write(all_houses.drop(columns='geometry').head())
