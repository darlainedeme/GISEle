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
    Create a grid of points within the study area.

    Parameters:
    - crs: The preferred CRS of the dataframe (integer).
    - resolution: The preferred resolution of the grid of points (integer).
    - study_area: The study area as a Shapely polygon in the preferred CRS.

    Returns:
    - geo_df_clipped: GeoDataFrame with the grid of points clipped to the study area.
    """
    df = pd.DataFrame(columns=['X', 'Y'])
    min_x, min_y, max_x, max_y = study_area.bounds

    # Create one-dimensional arrays for x and y
    lon = np.arange(min_x, max_x, resolution)
    lat = np.arange(min_y, max_y, resolution)
    lon, lat = np.meshgrid(lon, lat)
    df['X'] = lon.reshape((np.prod(lon.shape),))
    df['Y'] = lat.reshape((np.prod(lat.shape),))
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=crs)
    geo_df_clipped = gpd.clip(geo_df, study_area)

    return geo_df_clipped


def rasters_to_points(study_area, crs, resolution_points, dir, protected_areas, streets, resolution_population):
    """
    Create a grid of points and add attributes to it using raster data.

    Parameters:
    - study_area: Shapely polygon of the study area in the preferred CRS.
    - crs: Preferred CRS (integer).
    - resolution_points: Preferred resolution of the new grid of points (integer).
    - dir: Directory where data is stored.
    - protected_areas: Shapefile with the protected areas in the desired CRS.
    - streets: Shapefile with the roads in the desired CRS.
    - resolution_population: Resolution of the population raster.

    Returns:
    - pointData: DataFrame with the grid of points and their attributes.
    - geo_df: GeoDataFrame with the grid of points and their attributes.
    """
    pointData = create_grid(crs, resolution_points, study_area)
    pointData = pointData.to_crs(crs)

    # Read all the rasters
    elevation_Raster = rasterio.open(dir + '/Intermediate/Geospatial_Data/Elevation_' + str(crs) + '.tif')
    data_elevation, profile_elevation = resample(elevation_Raster, resolution_points, 'bilinear')
    with rasterio.open(dir + '/Intermediate/Geospatial_Data/Elevation_resampled.tif', 'w', **profile_elevation) as dst:
        dst.write(data_elevation)
    Elevation = rasterio.open(dir + '/Intermediate/Geospatial_Data/Elevation_resampled.tif')

    slope_Raster = rasterio.open(dir + '/Intermediate/Geospatial_Data/Slope_' + str(crs) + '.tif')
    data_slope, profile_slope = resample(slope_Raster, resolution_points, 'bilinear')
    with rasterio.open(dir + '/Intermediate/Geospatial_Data/Slope_resampled.tif', 'w', **profile_slope) as dst:
        dst.write(data_slope)
    Slope = rasterio.open(dir + '/Intermediate/Geospatial_Data/Slope_resampled.tif')

    landcover_Raster = rasterio.open(dir + '/Intermediate/Geospatial_Data/LandCover_' + str(crs) + '.tif')
    data_landcover, profile_landcover = resample(landcover_Raster, resolution_points, 'mode')
    with rasterio.open(dir + '/Intermediate/Geospatial_Data/Landcover_resampled.tif', 'w', **profile_landcover) as dst:
        dst.write(data_landcover)
    LandCover = rasterio.open(dir + '/Intermediate/Geospatial_Data/Landcover_resampled.tif')

    # Create a dataframe for the final grid of points
    df = pd.DataFrame(columns=['ID', 'X', 'Y', 'Elevation', 'Slope', 'Land_cover', 'Road_dist', 'River_flow', 'Protected_area'])

    # Sample the rasters
    coords = [(x, y) for x, y in zip(pointData.X, pointData.Y)]
    pointData = pointData.reset_index(drop=True)
    pointData['ID'] = pointData.index
    pointData['Elevation'] = [x[0] for x in Elevation.sample(coords)]
    pointData.loc[pointData.Elevation < 0, 'Elevation'] = 0
    print('Elevation finished')
    pointData['Slope'] = [x[0] for x in Slope.sample(coords)]
    print('Slope finished')
    pointData['Land_cover'] = [x[0] for x in LandCover.sample(coords)]
    print('Land cover finished')
    pointData['Protected_area'] = [protected_areas['geometry'].contains(Point(x, y)).any() for x, y in zip(pointData.X, pointData.Y)]
    print('Protected area finished')

    # Calculate road distances
    road_distances = []
    for index, point in pointData.iterrows():
        x, y = point['geometry'].xy[0][0], point['geometry'].xy[1][0]
        nearest_geoms = nearest_points(Point(x, y), streets)
        road_distance = nearest_geoms[0].distance(nearest_geoms[1])
        road_distances.append(road_distance)
        print('\r' + str(index) + '/' + str(pointData.index.__len__()), sep=' ', end='', flush=True)

    pointData['Road_dist'] = road_distances
    pointData['River_flow'] = ""

    geo_df = gpd.GeoDataFrame(pointData, geometry=gpd.points_from_xy(pointData.X, pointData.Y), crs=crs)

    return pointData, geo_df


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
    files_folder = os.path.join('scripts', 'routing_scripts', 'Case studies', 'awach555', 'Input', 'Geospatial_Data')
    protected_areas_file = locate_file(files_folder, folder='Protected_areas', extension='.shp')
    protected_areas = gpd.read_file(protected_areas_file).to_crs(crs)

    roads_file = locate_file(files_folder, folder='Roads', extension='.shp')
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
    elevation_file = locate_file(files_folder, folder='Elevation', extension='.tif')
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
    slope_file = locate_file(files_folder, folder='Slope', extension='.tif')
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
    landcover_file = locate_file(files_folder, folder='LandCover', extension='.tif')
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
            for line1 in line:
                for x in zip(line1.xy[0], line1.xy[1]):
                    streets_points.append(x)
        else:
            for x in zip(line.xy[0], line.xy[1]):
                streets_points.append(x)
    streets_multipoint = MultiPoint(streets_points)

    # Create and populate the grid of points
    df, geo_df = rasters_to_points(study_area_crs, crs, resolution, dir, protected_areas_clipped, streets_multipoint, resolution_population)
    geo_df.to_file(dir + '/Intermediate/Geospatial_Data/grid_of_points.shp')

    geo_df = geo_df.reset_index(drop=True)
    geo_df['ID'] = geo_df.index
    df = df.reset_index(drop=True)
    df['ID'] = df.index

    df.to_csv(dir + '/Intermediate/Geospatial_Data/grid_of_points.csv', index=False)

    # Perform the weighting of the grid of points
    df_weighted = initialization.weighting(df, resolution, landcover_option)
    df_weighted.to_csv(dir + '/Intermediate/Geospatial_Data/weighted_grid_of_points.csv', index=False)

    # Delete leftover files
    delete_leftover_files(dir, crs)

    return df_weighted


