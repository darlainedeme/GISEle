import zipfile
import os
import rasterio.mask
# from osgeo import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points
from shapely import ops
from rasterio.plot import show
from rasterio.mask import mask
import json
import matplotlib.pyplot as plt
import fiona
from collections import Counter
from statistics import mean
from math import ceil
import initialization
import QGIS_processing_polygon
import pdb


# Define the function resample
def resample(raster, resolution, options):
    """
    Resample a given raster according to the selected parameters.

    Parameters:
    - raster: the existing raster file
    - resolution: preferred resolution of the new file
    - options: method to be used in the resampling

    Returns:
    - data: Resampled raster data
    - profile: Updated raster profile
    """
    affine = raster.transform
    pixel_x = affine[0]
    scale_factor = pixel_x / resolution

    # Resample data to target shape
    if options == 'average':  # Resample according to the average value
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * scale_factor),
                int(raster.width * scale_factor)
            ),
            resampling=Resampling.average
        )
    elif options == 'mode':  # Resample according to the most frequent element
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * scale_factor),
                int(raster.width * scale_factor)
            ),
            resampling=Resampling.mode
        )
    else:  # Default to bilinear resampling
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * scale_factor),
                int(raster.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )

    # Scale image transform
    transform = raster.transform * raster.transform.scale(
        (raster.width / data.shape[-1]),
        (raster.height / data.shape[-2])
    )

    profile = raster.profile
    profile.update(transform=transform, width=raster.width * scale_factor,
                   height=raster.height * scale_factor)

    return data, profile


def create_grid(crs, resolution, study_area):
    """
    Create a grid of points for the study area with the specified resolution.
    
    Parameters:
    - crs: Coordinate Reference System (integer).
    - resolution: Resolution for the grid (integer).
    - study_area: GeoDataFrame representing the study area.
    
    Returns:
    - GeoDataFrame of grid points.
    """
    # Get the bounding box of the study area
    min_x, min_y, max_x, max_y = study_area.total_bounds

    # Ensure resolution is valid
    if resolution <= 0:
        raise ValueError("Resolution must be a positive number.")

    # Check the bounding box values
    if min_x >= max_x or min_y >= max_y:
        raise ValueError(f"Invalid bounding box values: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")

    # Create the grid points
    try:
        lon = np.arange(min_x, max_x, resolution)
        lat = np.arange(min_y, max_y, resolution)
        grid_points = [Point(x, y) for x in lon for y in lat]
    except Exception as e:
        raise ValueError(f"Error creating grid points: {e}")

    # Create a GeoDataFrame from the grid points
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=crs)

    # Add X and Y columns
    grid_gdf['X'] = grid_gdf.geometry.x
    grid_gdf['Y'] = grid_gdf.geometry.y
    
    # Clip the grid to the study area
    grid_gdf = gpd.clip(grid_gdf, study_area)

    return grid_gdf


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

    for x, y in coords:
        # Placeholder value for elevation
        # Replace this with actual logic to read elevation for each coordinate
        elevation_value = 100  # Example static value
        slope_value = 10       # Example static value
        land_cover_value = 1   # Example static value
        elevation_data.append(elevation_value)
        slope_data.append(slope_value)
        land_cover_data.append(land_cover_value)

    # Create the DataFrame with necessary columns
    data = {
        'ID': range(len(coords)),
        'X': [coord[0] for coord in coords],
        'Y': [coord[1] for coord in coords],
        'Elevation': elevation_data,
        'Slope': slope_data,
        'Land_cover': land_cover_data,
        'OtherData': [0] * len(coords)  # Placeholder for other data columns
    }
    df = pd.DataFrame(data)
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']), crs=crs)

    return df, geo_df


def locate_file(database, folder, extension):
    """
    Locate a file's name based on an extension and data on the location of the file.

    Parameters:
    - database: Location of the main database.
    - folder: Name of the folder to look in.

    Returns:
    - path: Path to the located file.
    """
    root = database + '/' + folder
    for file in os.listdir(root):
        if extension in file:
            path = os.path.join(root, file)
            return path


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

def delete_leftover_files(dir, crs):
    """
    Delete leftover files from the resampling and reprojecting process.

    Parameters:
    - dir: Directory where data is stored.
    - crs: CRS used for the files.
    """
    folder = dir + '/Intermediate/Geospatial_Data/'
    os.remove(folder + 'Elevation.tif')
    os.remove(folder + 'LandCover.tif')
    os.remove(folder + 'Slope.tif')


def create_input_csv(crs, resolution, resolution_population, landcover_option, country, case_study, database, study_area):
    """
    Create a weighted grid of points for the area of interest.

    Parameters:
    - crs: Desired CRS (integer).
    - resolution: Desired resolution for the MV grid routing (integer).
    - resolution_population: Resolution of the population raster.
    - landcover_option: Preferred landcover option (string).
    - country: Name of the country being analyzed.
    - case_study: Name for the case study.
    - database: Location of the database where all the raster/shape files are stored.
    - study_area: Shapely polygon of the area of interest in the preferred CRS.

    Returns:
    - df_weighted: DataFrame with the weighted grid of points.
    """
    database = database + '/' + country
    crs_str = 'epsg:' + str(crs)

    # Open the roads, protected areas, and rivers
    protected_areas_file = locate_file(database, folder='Protected_areas', extension='.shp')
    protected_areas = gpd.read_file(protected_areas_file).to_crs(crs)

    roads_file = locate_file(database, folder='Roads', extension='.shp')
    streets = gpd.read_file(roads_file).to_crs(crs)

    study_area_crs = study_area.to_crs(crs)
    dir = 'Case studies/' + case_study

    # Create a small buffer to avoid issues
    study_area_buffered = study_area.buffer((resolution * 0.1 / 11250) / 2)

    # Clip the protected areas and streets
    protected_areas_clipped = gpd.clip(protected_areas, study_area_crs)
    streets_clipped = gpd.clip(streets, study_area_crs)

    if not streets_clipped.empty:
        streets_clipped.to_file(dir + '/Intermediate/Geospatial_Data/Roads.shp')

    if not protected_areas_clipped.empty:
        protected_areas_clipped.to_file(dir + '/Intermediate/Geospatial_Data/protected_area.shp')

    # Clip the elevation and then change the CRS
    elevation_file = locate_file(database, folder='Elevation', extension='.tif')
    with rasterio.open(elevation_file, mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir + '/Intermediate/Geospatial_Data/Elevation.tif', "w", **out_meta) as dest:
        dest.write(out_image)

    # Reproject using rasterio
    input_raster = dir + '/Intermediate/Geospatial_Data/Elevation.tif'
    output_raster = dir + '/Intermediate/Geospatial_Data/Elevation_' + str(crs) + '.tif'
    reproject_raster(input_raster, output_raster, crs_str)

    # Clip the slope and then change the CRS
    slope_file = locate_file(database, folder='Slope', extension='.tif')
    with rasterio.open(slope_file, mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir + '/Intermediate/Geospatial_Data/Slope.tif', "w", **out_meta) as dest:
        dest.write(out_image)

    # Reproject using rasterio
    input_raster = dir + '/Intermediate/Geospatial_Data/Slope.tif'
    output_raster = dir + '/Intermediate/Geospatial_Data/Slope_' + str(crs) + '.tif'
    reproject_raster(input_raster, output_raster, crs_str)

    # Clip the land cover and then change the CRS
    landcover_file = locate_file(database, folder='LandCover', extension='.tif')
    with rasterio.open(landcover_file, mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered.to_crs(src.crs), crop=True)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir + '/Intermediate/Geospatial_Data/LandCover.tif', "w", **out_meta) as dest:
        dest.write(out_image)

    # Reproject using rasterio
    input_raster = dir + '/Intermediate/Geospatial_Data/LandCover.tif'
    output_raster = dir + '/Intermediate/Geospatial_Data/LandCover_' + str(crs) + '.tif'
    reproject_raster(input_raster, output_raster, crs_str)

    # Convert streets from lines to multipoints
    streets_points = []
    for line in streets_clipped['geometry']:
        if line.geom_type == 'MultiLineString':
            for line1 in line.geoms:
                for x in zip(line1.xy[0], line1.xy[1]):
                    streets_points.append(x)
        elif line.geom_type == 'LineString':
            for x in zip(line.xy[0], line.xy[1]):
                streets_points.append(x)
        else:
            st.warning(f"Unexpected geometry type: {line.geom_type}")

    streets_multipoint = MultiPoint(streets_points)

    # Create and populate the grid of points
    df, geo_df = rasters_to_points(study_area_crs, crs, resolution, dir, protected_areas_clipped, streets_multipoint, resolution_population)
    
    # Check if the required columns exist
    required_columns = ['Elevation', 'Slope', 'Land_cover']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    df_weighted = initialization.weighting(df, resolution, landcover_option)

    # Delete leftover files
    delete_leftover_files(dir, crs)

    return df_weighted
