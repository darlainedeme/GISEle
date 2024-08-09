import os
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import MultiPoint, MultiLineString
import streamlit as st
from rasterio.enums import Resampling
import numpy as np
from shapely.geometry import Point
import math

def new_case_study(parameters, output_path_clusters):
    try:
        st.write("2. New case study creation")

        gisele_folder = parameters["gisele_folder"]
        crs = parameters["crs"]

        # Define paths
        database = gisele_folder
        study_area_folder = os.path.join(database, 'data', '3_user_uploaded_data')
        intermediate_output_folder = os.path.join(database, 'data', '4_intermediate_output')
        final_output_folder = os.path.join(database, 'data', '5_final_output')

        # Create necessary directories for the case study
        os.makedirs(os.path.join(intermediate_output_folder, 'Communities'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Microgrid'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Optimization', 'MILP_output'), exist_ok=True)
        os.makedirs(os.path.join(final_output_folder, 'MILP_processed'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Optimization', 'all_data', 'Lines_connections'), exist_ok=True)
        os.makedirs(os.path.join(intermediate_output_folder, 'Optimization', 'all_data', 'Lines_marked'), exist_ok=True)

        # Copy the configuration file to the destination directory
        configuration_path = os.path.join(database, 'data', '0_configuration_files', 'Configuration.csv')
        pd.read_csv(configuration_path).to_csv(os.path.join(intermediate_output_folder, 'Configuration.csv'), index=False)

        # Process and save substations data
        substations_path = os.path.join(database, 'data', '4_intermediate_output', 'connection_points', 'con_points.shp')
        Substations = gpd.read_file(substations_path)
        Substations = Substations.to_crs(crs)
        Substations['X'] = Substations.geometry.apply(lambda geom: geom.xy[0][0])
        Substations['Y'] = Substations.geometry.apply(lambda geom: geom.xy[1][0])
        Substations.to_file(os.path.join(database, 'data', '2_downloaded_input_data', 'substations', 'substations.shp'))

        # Save the study area file
        study_area = gpd.read_file(study_area_folder)
        st.write(study_area.drop(columns='geometry').head())
        study_area.to_file(os.path.join(study_area_folder, 'Study_area.geojson'))

        # Process and save the clusters data
        Clusters = gpd.read_file(output_path_clusters)
        Clusters = Clusters.to_crs(crs)
        Clusters['cluster_ID'] = range(1, Clusters.shape[0] + 1)
        for i, row in Clusters.iterrows():
            if row['geometry'].geom_type == 'MultiPolygon':
                Clusters.at[i, 'geometry'] = row['geometry'][0]
        Clusters.to_file(os.path.join(intermediate_output_folder, 'clustering', 'Communities_boundaries.shp'))

        st.write("New case study creation completed")
        return Clusters, study_area, Substations

    except Exception as e:
        st.error(f"An error occurred during case study creation: {e}")
        raise

def create_study():
    try:
        st.write("Initializing case study creation...")

        # Define the parameters required for the case study creation
        parameters = {
            "gisele_folder": "/mount/src/gisele",  # Base folder path
            "crs": "EPSG:4326"  # The coordinate reference system to be used
        }
        output_path_clusters = os.path.join(parameters["gisele_folder"], 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')  # Path to the clusters file

        # Call the new_case_study function
        Clusters, study_area, Substations = new_case_study(parameters, output_path_clusters)

        # Display the results excluding the geometry column
        st.write("Case study created successfully.")
        st.write("Clusters:", Clusters.drop(columns='geometry'))  # Exclude geometry column
        st.write("Study Area:", study_area.drop(columns='geometry'))  # Exclude geometry column
        st.write("Substations:", Substations.drop(columns='geometry'))  # Exclude geometry column

    except Exception as e:
        st.error(f"An error occurred during the case study creation process: {e}")
        raise

def create_input_csv(crs, resolution, resolution_population, landcover_option, database, study_area):
    """
    Create a weighted grid of points for the area of interest.
    """
    crs_str = 'epsg:' + str(crs)
    
    geospatial_data_path = os.path.join('data', '2_downloaded_input_data')
    
    protected_areas_file = os.path.join(geospatial_data_path, 'protected_areas', 'protected_areas.shp')
    roads_file = os.path.join(geospatial_data_path, 'roads', 'roads.shp')
    elevation_file = os.path.join(geospatial_data_path, 'elevation', 'Elevation.tif')
    slope_file = os.path.join(geospatial_data_path, 'slope', 'slope.tif')
    landcover_file = os.path.join(database, 'data', '2_downloaded_input_data', 'landcover', 'LandCover.tif')

    # Open the roads, protected areas, and rivers
    protected_areas = gpd.read_file(protected_areas_file).to_crs(crs)
    st.write("Protected areas loaded and reprojected:")
    st.write(protected_areas.drop(columns='geometry').head())

    streets = gpd.read_file(roads_file).to_crs(crs)
    st.write("Roads loaded and reprojected:")
    st.write(streets.drop(columns='geometry').head())
    
    study_area_crs = study_area.to_crs(crs)
    st.write("Study area reprojected:")
    st.write(study_area_crs.drop(columns='geometry').head())

    # Create a small buffer to avoid issues
    study_area_buffered = study_area.buffer((resolution * 0.1 / 11250) / 2)

    # Clip the protected areas and streets
    protected_areas_clipped = gpd.clip(protected_areas, study_area_crs)
    streets_clipped = gpd.clip(streets, study_area_crs)

    if not streets_clipped.empty:
        streets_clipped.to_file(os.path.join(geospatial_data_path, 'Roads.shp'))

    if not protected_areas_clipped.empty:
        protected_areas_clipped.to_file(os.path.join(geospatial_data_path, 'protected_area.shp'))

    # Clip the elevation and then change the CRS
    with rasterio.open(elevation_file, mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(os.path.join(geospatial_data_path, 'Elevation.tif'), "w", **out_meta) as dest:
        dest.write(out_image)

    # Reproject using rasterio
    input_raster = os.path.join(geospatial_data_path, 'Elevation.tif')
    output_raster = os.path.join(geospatial_data_path, f'Elevation_{crs}.tif')
    reproject_raster(input_raster, output_raster, crs_str)

    # Clip the slope and then change the CRS
    with rasterio.open(slope_file, mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(os.path.join(geospatial_data_path, 'Slope.tif'), "w", **out_meta) as dest:
        dest.write(out_image)

    # Reproject using rasterio
    input_raster = os.path.join(geospatial_data_path, 'Slope.tif')
    output_raster = os.path.join(geospatial_data_path, f'Slope_{crs}.tif')
    reproject_raster(input_raster, output_raster, crs_str)

    # Clip the land cover and then change the CRS
    with rasterio.open(landcover_file, mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(os.path.join(geospatial_data_path, 'LandCover.tif'), "w", **out_meta) as dest:
        dest.write(out_image)

    # Reproject using rasterio
    input_raster = os.path.join(geospatial_data_path, 'LandCover.tif')
    output_raster = os.path.join(geospatial_data_path, f'LandCover_{crs}.tif')
    reproject_raster(input_raster, output_raster, crs_str)

    # Convert streets from lines to multipoints
    streets_points = []
    for line in streets_clipped['geometry']:
        try:
            if line.geom_type == 'MultiLineString':
                for line1 in line.geoms:
                    for x in zip(line1.xy[0], line1.xy[1]):
                        streets_points.append(x)
            elif line.geom_type == 'LineString':
                for x in zip(line.xy[0], line.xy[1]):
                    streets_points.append(x)
            else:
                st.warning(f"Unexpected geometry type: {line.geom_type}")
        except AttributeError as e:
            st.error(f"AttributeError: {e} for geometry: {line}")
        except TypeError as e:
            st.error(f"TypeError: {e} for geometry: {line}")

    if streets_points:
        streets_multipoint = MultiPoint(streets_points)
    else:
        streets_multipoint = MultiPoint()
        st.warning("No points extracted from streets data")

    # Create and populate the grid of points
    df, geo_df = rasters_to_points(study_area_crs, crs, resolution, geospatial_data_path, protected_areas_clipped, streets_multipoint, resolution_population)
    geo_df.to_file(os.path.join(geospatial_data_path, 'grid_of_points.shp'))

    geo_df = geo_df.reset_index(drop=True)
    geo_df['ID'] = geo_df.index
    df = df.reset_index(drop=True)
    df['ID'] = df.index

    df.to_csv(os.path.join(geospatial_data_path, 'grid_of_points.csv'), index=False)

    # Check if the required columns exist
    required_columns = ['Elevation', 'Slope', 'Land_cover', 'Road_dist', 'Protected_area']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # Perform the weighting of the grid of points
    df_weighted = weighting(df, resolution, landcover_option)
    df_weighted.to_csv(os.path.join(geospatial_data_path, 'weighted_grid_of_points.csv'), index=False)

    return df_weighted

def create_roads_new(gisele_folder, Clusters, crs, accepted_road_types, resolution_MV, resolution_LV):
    geospatial_data_path = os.path.join(gisele_folder, 'data', '4_intermediate_output')

    weighted_grid_of_points = pd.read_csv(os.path.join(geospatial_data_path, 'weighted_grid_of_points.csv'))
    starting_ID = weighted_grid_of_points['ID'].max() + 1
    ROADS_unfiltered = gpd.read_file(os.path.join(geospatial_data_path, 'Roads.shp'))
    ROADS_unfiltered = ROADS_unfiltered.to_crs(crs)
    ROADS = MultiLine_to_Line(ROADS_unfiltered)
    all_points = gpd.GeoDataFrame()
    gdf_ROADS, ROADS_segments = create_roads2(ROADS, all_points, crs)
    gdf_ROADS.crs = crs
    ROADS_segments.crs = crs

    MP = MultiPolygon([p for p in Clusters['geometry']])
    nodes = ROADS_segments.ID1.to_list() + ROADS_segments.ID2.to_list()
    nodes = [int(i) for i in nodes]
    occurence = Counter(nodes)

    intersection_IDs = []
    terminal_IDs = []
    for i in occurence:
        if occurence[i] == 1:
            terminal_IDs.append(i)
        elif occurence[i] > 2:
            intersection_IDs.append(i)
    new_nodes = terminal_IDs + intersection_IDs

    Substations = new_nodes
    Nodes = gdf_ROADS.copy()
    Nodes.loc[Nodes['ID'].isin(new_nodes), 'Substation'] = 1

    Nodes['inside_clusters'] = [1 if MP.contains(row['geometry']) else 0 for i, row in Nodes.iterrows()]
    Lines = ROADS_segments.copy()
    Lines.ID1 = Lines.ID1.astype(int)
    Lines.ID2 = Lines.ID2.astype(int)

    Lines_marked = Lines.copy()
    conn_param = 0
    New_Lines = gpd.GeoDataFrame(columns=['ID1', 'ID2', 'Cost', 'length', 'geometry', 'Conn_param'], crs=crs, geometry='geometry')
    while not Lines.empty:
        nodes = Lines.ID1.to_list() + Lines.ID2.to_list()
        nodes = [int(i) for i in nodes]
        Substations = list(set(Substations) & set(nodes))
        current_node = int(Substations[0])
        no_lines = False
        tot_length = 0
        tot_cost = 0
        id1 = current_node
        while not no_lines:
            next_index = Lines.index[Lines['ID1'] == current_node].to_list()
            if next_index:
                next_index = next_index[0]
                next_node = Lines.loc[next_index, 'ID2']
                tot_length = tot_length + Lines.loc[next_index, 'length']
                tot_cost = tot_cost + Lines.loc[next_index, 'length']
                Lines.drop(index=next_index, inplace=True)
            else:
                next_index = Lines.index[Lines['ID2'] == current_node].to_list()
                if next_index:
                    next_index = next_index[0]
                    next_node = Lines.loc[next_index, 'ID1']
                    tot_length = tot_length + Lines.loc[next_index, 'length']
                    tot_cost = tot_cost + Lines.loc[next_index, 'length']
                    Lines.drop(index=next_index, inplace=True)
                else:
                    no_lines = True

            if not no_lines:
                is_substation = Nodes.loc[Nodes.ID == int(next_node), 'Substation'] == 1
                is_inside = int(Nodes.loc[Nodes.ID == int(next_node), 'inside_clusters'])
                if is_inside == 1:
                    max_tot_length = resolution_LV / 1000
                else:
                    max_tot_length = resolution_MV / 1000
                Lines_marked.loc[next_index, 'Conn_param'] = conn_param
                if is_substation.values[0]:
                    Point1 = Nodes.loc[Nodes['ID'] == int(id1), 'geometry'].values[0]
                    Point2 = Nodes.loc[Nodes['ID'] == int(next_node), 'geometry'].values[0]
                    geom = LineString([Point1, Point2])
                    Data = {'ID1': id1, 'ID2': next_node, 'Cost': tot_cost, 'length': tot_length, 'geometry': geom, 'Conn_param': conn_param}
                    New_Lines = pd.concat([New_Lines, gpd.GeoDataFrame([Data], crs=crs, geometry='geometry')], ignore_index=True)
                    current_node = next_node
                    tot_length = 0
                    tot_cost = 0
                    id1 = current_node
                    conn_param = conn_param + 1
                elif tot_length > max_tot_length:
                    Point1 = Nodes.loc[Nodes['ID'] == int(id1), 'geometry'].values[0]
                    Point2 = Nodes.loc[Nodes['ID'] == int(next_node), 'geometry'].values[0]
                    geom = LineString([Point1, Point2])
                    Data = {'ID1': id1, 'ID2': next_node, 'Cost': tot_cost, 'length': tot_length, 'geometry': geom, 'Conn_param': conn_param}
                    New_Lines = pd.concat([New_Lines, gpd.GeoDataFrame([Data], crs=crs, geometry='geometry')], ignore_index=True)
                    current_node = next_node
                    tot_length = 0
                    tot_cost = 0
                    id1 = current_node
                    conn_param = conn_param + 1
                else:
                    current_node = next_node

    new_lines = []
    for i, row in New_Lines.iterrows():
        actual_Lines = Lines_marked.loc[Lines_marked['Conn_param'] == row['Conn_param'], 'geometry']
        new_line = MultiLineString([actual_Lines.values[i] for i in range(len(actual_Lines))])
        new_lines.append(new_line)

    New_Lines.geometry = new_lines

    new_nodes = New_Lines.ID1.to_list() + New_Lines.ID2.to_list()
    New_Nodes = gdf_ROADS[gdf_ROADS['ID'].isin(new_nodes)]
    New_Nodes.reset_index(inplace=True)
    for i, row in New_Nodes.iterrows():
        id = int(i)
        New_Nodes.loc[i, 'ID'] = id
        New_Lines.loc[New_Lines['ID1'] == row['ID'], 'ID1'] = id
        New_Lines.loc[New_Lines['ID2'] == row['ID'], 'ID2'] = id

    New_Nodes['ID'] += starting_ID
    New_Lines['ID1'] += starting_ID
    New_Lines['ID2'] += starting_ID

    drop = New_Lines.loc[New_Lines['ID1'] == New_Lines['ID2'], :]
    if not len(drop) == 0:
        New_Lines.drop(index=drop.index, inplace=True)

    New_Lines.to_file(os.path.join(geospatial_data_path, 'Roads_lines', 'Roads_lines.shp'))
    New_Nodes.to_file(os.path.join(geospatial_data_path, 'Roads_points', 'Roads_points.shp'))

    return New_Nodes, New_Lines

def Merge_Roads_GridOfPoints(gisele_folder):
    geospatial_data_path = os.path.join(gisele_folder, 'data', '4_intermediate_output', 'Geospatial_Data')

    road_points = gpd.read_file(os.path.join(geospatial_data_path, 'Roads_points', 'Roads_points.shp'))
    weighted_grid_points = pd.read_csv(os.path.join(geospatial_data_path, 'weighted_grid_of_points.csv'))

    weighted_grid_points['Type'] = 'Standard'
    road_points['Type'] = 'Road'

    road_points.drop(columns=['geometry'], inplace=True)

    weighted_grid_points_with_roads = pd.concat([weighted_grid_points, road_points], ignore_index=True)
    weighted_grid_points_with_roads[['X', 'Y', 'ID', 'Elevation', 'Type', 'Weight', 'Elevation']].\
        to_csv(os.path.join(geospatial_data_path, 'weighted_grid_of_points_with_roads.csv'))

def reproject_raster(input_raster, output_raster, dst_crs):
    """
    Reproject a raster to a different CRS using rasterio.

    Parameters:
    - input_raster: Path to the input raster file.
    - output_raster: Path to the output raster file.
    - dst_crs: Desired CRS as an EPSG code (string).
    """
    with rasterio.open(input_raster) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def rasters_to_points(study_area_crs, crs, resolution, dir, protected_areas_clipped, streets_multipoint, resolution_population):
    """
    Convert rasters and other inputs into a grid of points.
    """
    # Create the grid
    pointData = create_grid(crs, resolution, study_area_crs)

    # Ensure pointData has 'X' and 'Y' columns
    if 'X' not in pointData.columns or 'Y' not in pointData.columns:
        raise ValueError("pointData does not have 'X' and 'Y' columns")

    coords = [(x, y) for x, y in zip(pointData['X'], pointData['Y'])]

    # Placeholder for reading elevation data
    elevation_data = []
    slope_data = []
    land_cover_data = []
    road_dist_data = []
    protected_area_data = []

    # Replace these placeholder values with actual logic to read data for each coordinate
    for x, y in coords:
        elevation_value = get_elevation_at_point(x, y)  # Actual logic needed
        slope_value = get_slope_at_point(x, y)  # Actual logic needed
        land_cover_value = get_land_cover_at_point(x, y)  # Actual logic needed
        road_dist_value = get_road_distance_at_point(x, y)  # Actual logic needed
        protected_area_value = is_point_in_protected_area(x, y)  # Actual logic needed

        elevation_data.append(elevation_value)
        slope_data.append(slope_value)
        land_cover_data.append(land_cover_value)
        road_dist_data.append(road_dist_value)
        protected_area_data.append(protected_area_value)

    # Create the DataFrame with necessary columns
    data = {
        'ID': range(len(coords)),
        'X': [coord[0] for coord in coords],
        'Y': [coord[1] for coord in coords],
        'Elevation': elevation_data,
        'Slope': slope_data,
        'Land_cover': land_cover_data,
        'Road_dist': road_dist_data,
        'Protected_area': protected_area_data
    }
    df = pd.DataFrame(data)
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']), crs=crs)

    # Check the generated data
    st.write(df.head())  # Debugging line to verify data

    return df, geo_df

def weighting(df, resolution, landcover_option):
    """
    Assign weights to all points of a dataframe according to the terrain
    characteristics and the distance to the nearest road.
    :param df: From which part of GISEle the user is starting
    :param resolution: resolution of the dataframe df
    :return df_weighted: Point dataframe with weight attributes assigned
    """
    df_weighted = df.dropna(subset=['Elevation'])
    df_weighted.reset_index(drop=True)
    df_weighted.Slope.fillna(value=0, inplace=True)
    df_weighted.Land_cover.fillna(method='bfill', inplace=True)
    df_weighted['Land_cover'] = df_weighted['Land_cover'].round(0)
    df_weighted['Weight'] = 0
    print('Weighting the Dataframe..')

    # Directly load the Landcover.csv file without changing directories
    landcover_csv_path = os.path.join('data', '0_configuration_files', 'Landcover.csv')
    landcover_csv = pd.read_csv(landcover_csv_path)

    del df

    # Weighting section
    # Slope conditions
    df_weighted['Weight'] = math.e**(0.01732867951 * df_weighted['Slope'])

    # Land cover using the column Other or GLC to compute the weight
    for i, row in landcover_csv.iterrows():
        if landcover_option == 'GLC':
            df_weighted.loc[df_weighted['Land_cover'] == row['GLC2000'], 'Weight'] += landcover_csv.loc[i, 'WeightGLC']
        elif landcover_option == 'ESACCI':
            df_weighted.loc[df_weighted['Land_cover'] == row['ESACCI'], 'Weight'] += landcover_csv.loc[i, 'WeightESACCI']
        else:
            df_weighted.loc[df_weighted['Land_cover'] == row['Other'], 'Weight'] += landcover_csv.loc[i, 'WeightOther']

    # Road distance conditions
    df_weighted.loc[df_weighted['Road_dist'] < 1000, 'Weight'] += \
        5 * df_weighted.loc[df_weighted['Road_dist'] < 1000, 'Road_dist'] / 1000

    df_weighted.loc[df_weighted['Road_dist'] >= 1000, 'Weight'] += 5
    df_weighted.loc[df_weighted['Road_dist'] < resolution / 2, 'Weight'] = 1.5

    # Protected areas condition
    df_weighted.loc[df_weighted['Protected_area'] == True, 'Weight'] += 5

    # Cleaning up the dataframe
    valid_fields = ['ID', 'X', 'Y', 'Elevation', 'Weight']
    blacklist = [x for x in df_weighted.columns if x not in valid_fields]
    df_weighted.drop(blacklist, axis=1, inplace=True)
    df_weighted.drop_duplicates(['ID'], keep='last', inplace=True)

    print("Cleaning and weighting process completed")
    return df_weighted

def create_grid(crs, resolution, study_area):
    """
    Create a grid of points for the study area with the specified resolution.
    
    Parameters:
    - crs: Coordinate Reference System (integer).
    - resolution: Resolution for the grid (in degrees for EPSG:4326).
    - study_area: GeoDataFrame representing the study area.
    
    Returns:
    - GeoDataFrame of grid points.
    """
    # Get the bounding box of the study area
    min_x, min_y, max_x, max_y = study_area.total_bounds
    st.write(f"Bounding box of study area: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

    # Ensure resolution is valid
    if resolution <= 0:
        raise ValueError("Resolution must be a positive number.")

    # Create the grid points
    lon = np.arange(min_x, max_x, resolution)
    lat = np.arange(min_y, max_y, resolution)
    grid_points = [Point(x, y) for x in lon for y in lat]
    st.write(f"Number of grid points generated: {len(grid_points)}")

    # Create a GeoDataFrame from the grid points
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=crs)

    # Add X and Y columns
    grid_gdf['X'] = grid_gdf.geometry.x
    grid_gdf['Y'] = grid_gdf.geometry.y
    
    # Clip the grid to the study area
    grid_gdf = gpd.clip(grid_gdf, study_area)
    st.write(f"Number of grid points after clipping to study area: {len(grid_gdf)}")

    return grid_gdf

def show():
    # Step 1: Create the Case Study
    with st.expander("Case Study Parameters", expanded=False):
        st.write("Initialize the parameters for creating a new case study.")
        
        # Parameters for case study creation
        gisele_folder = st.text_input("GISELE Folder Path", "/mount/src/gisele")
        crs = st.number_input("Coordinate Reference System (CRS)", value=4326)
        output_path_clusters = os.path.join(gisele_folder, 'data', '4_intermediate_output', 'clustering', 'Communities_boundaries.shp')

        if st.button("Create Case Study"):
            parameters = {
                "gisele_folder": gisele_folder,
                "crs": crs
            }
            Clusters, study_area, Substations = new_case_study(parameters, output_path_clusters)
            st.write("Case study created successfully.")
            st.write("Clusters:", Clusters.drop(columns='geometry'))
            st.write("Study Area:", study_area.drop(columns='geometry'))
            st.write("Substations:", Substations.drop(columns='geometry'))

    # Step 2: Create the Weighted Grid of Points
    with st.expander("Weighted Grid Parameters", expanded=False):
        st.write("Define the parameters for creating a weighted grid of points.")
        
        resolution = st.number_input("Grid Resolution (meters)", value=100)
        resolution_population = st.number_input("Population Resolution (meters)", value=100)
        landcover_option = st.selectbox(
            "Landcover Option",
            options=["GLC", "ESACCI"],  # The two options
            index=1  # Set "ESACCI" as the default option (index 1)
        )

        if st.button("Create Weighted Grid CSV"):
            study_area = gpd.read_file(os.path.join(gisele_folder, 'data', '3_user_uploaded_data', 'selected_area.geojson'))
            df_weighted = create_input_csv(crs, resolution, resolution_population, landcover_option, gisele_folder, study_area)
            st.write("Weighted grid of points CSV created successfully.")
            st.dataframe(df_weighted.head())

    # Optional: Additional steps for creating roads and merging grids
    with st.expander("Roads and Merging (Optional)", expanded=False):
        st.write("You can optionally create roads and merge them with the grid of points.")
        
        accepted_road_types = st.multiselect(
            "Accepted Road Types",
            options=[
                'living_street', 'pedestrian', 'primary', 'primary_link', 'secondary', 'secondary_link',
                'tertiary', 'tertiary_link', 'unclassified', 'residential'
            ],
            default=[
                'living_street', 'pedestrian', 'primary', 'primary_link', 'secondary', 'secondary_link',
                'tertiary', 'tertiary_link', 'unclassified', 'residential'
            ]
        )
        resolution_MV = st.number_input("MV Resolution (meters)", value=1000)
        resolution_LV = st.number_input("LV Resolution (meters)", value=100)

        if st.button("Create Roads and Merge with Grid"):
            Clusters, study_area, Substations = new_case_study(parameters, output_path_clusters)
            New_Nodes, New_Lines = create_roads_new(gisele_folder, Clusters, crs, accepted_road_types, resolution_MV, resolution_LV)
            Merge_Roads_GridOfPoints(gisele_folder)
            st.write("Roads created and merged with the grid of points successfully.")
