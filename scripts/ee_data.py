import ee
import requests
import zipfile
import numpy as np
import rasterio
import os
import streamlit as st
from rasterio.mask import mask
from shapely.geometry import mapping
import rasterio as rio

def download_ee_image(dataset, bands, region, filename, scale=30, dateMin=None, dateMax=None, crs='EPSG:4326'):
    print(f'Downloading {dataset} dataset ... ')
    
    if not isinstance(region, ee.Geometry):
        region = ee.Geometry.Polygon(region.exterior.coords[:])

    collection = ee.ImageCollection(dataset).filterBounds(region)
    
    if dateMin and dateMax:
        collection = collection.filterDate(ee.Date(dateMin), ee.Date(dateMax))
    
    image = collection.mosaic().clip(region)
    image = image.addBands(ee.Image.pixelLonLat())
    
    for band in bands:
        task = ee.batch.Export.image.toDrive(image=image.select(band),
                                             description=band,
                                             scale=scale,
                                             region=region,
                                             fileNamePrefix=band,
                                             crs=crs,
                                             fileFormat='GeoTIFF')
        task.start()

        url = image.select(band).getDownloadURL({
            'scale': scale,
            'crs': crs,
            'fileFormat': 'GeoTIFF',
            'region': region})
        
        r = requests.get(url, stream=True)

        filenameZip = f'{band}.zip'
        filenameTif = f'{band}.tif'

        with open(filenameZip, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

        zipdata = zipfile.ZipFile(filenameZip)
        zipinfos = zipdata.infolist()

        for zipinfo in zipinfos:
            zipinfo.filename = filenameTif
            zipdata.extract(zipinfo)
        
        zipdata.close()
        
    print('Creating multi-band GeoTIFF image ... ')
    
    band_files = [rasterio.open(f'{band}.tif') for band in bands]

    image = np.array([band_file.read(1) for band_file in band_files]).transpose(1, 2, 0)
    p2, p98 = np.percentile(image, (2, 98))

    first_band_geo = band_files[0].profile
    first_band_geo.update({'count': len(bands)})

    with rasterio.open(filename, 'w', **first_band_geo) as dest:
        for i, band_file in enumerate(band_files):
            dest.write((np.clip(band_file.read(1), p2, p98) - p2) / (p98 - p2) * 255, i + 1)

    for band_file in band_files:
        band_file.close()
    
    for band in bands:
        os.remove(f'{band}.tif')
        os.remove(f'{band}.zip')


def download_url(url, out_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(out_path, 'wb') as out_file:
            for chunk in response.iter_content(1024):
                out_file.write(chunk)
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    extracted_files = zip_ref.namelist()
    return extracted_files

def download_and_extract_band(image, selection, scale, aoi, output_folder):
    url = image.select(selection).getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'fileFormat': 'GeoTIFF',
        'region': aoi
    })
    zip_path = os.path.join(output_folder, f"{selection}.zip")
    tif_path = os.path.join(output_folder, f"{selection}.tif")
    download_url(url, zip_path)
    extracted_files = extract_zip(zip_path, output_folder)
    if not any(tif_path in file for file in extracted_files):
        raise Exception(f"{tif_path} not found in the extracted files.")
    return tif_path

def combine_bands_to_rgb(band_paths, output_path):
    bands = [rio.open(band) for band in band_paths]
    profile = bands[0].profile
    profile.update(count=3)

    with rio.open(output_path, 'w', **profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.read(1), i)

    for band in bands:
        band.close()

    for band_path in band_paths:
        os.remove(band_path)
        os.remove(band_path.replace('.tif', '.zip'))

def download_elevation_data(polygon, zip_path, dem_path):
    try:
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        # Define the Earth Engine image for SRTM
        srtm_image = ee.Image('USGS/SRTMGL1_003')

        # Define the CRS and scale
        crs = 4326  # WGS84
        scale = 30  # SRTM resolution is approximately 30 meters

        # Get the coordinates for the area of interest
        min_x, min_y, max_x, max_y = polygon.bounds
        coords = [
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y],
            [min_x, min_y]
        ]
        aoi = ee.Geometry.Polygon(coords)

        # Download and extract each band
        bands = ['B4', 'B3', 'B2']
        band_paths = []
        for band in bands:
            band_path = download_and_extract_band(srtm_image, band, scale, aoi, os.path.dirname(zip_path))
            band_paths.append(band_path)

        # Combine bands into a single RGB GeoTIFF
        combine_bands_to_rgb(band_paths, dem_path)

        # Verify if the file has been extracted and is a valid raster file
        if os.path.isfile(dem_path):
            # Open and show the DEM using rasterio
            dem = rio.open(dem_path)
            show(dem)
            st.write("Elevation data downloaded.")
            return dem_path
        else:
            raise Exception("File extraction failed or file is not valid.")
    except Exception as e:
        st.error(f"Error downloading elevation data: {e}")
        return None
        
def clip_raster_to_polygon(raster_path, polygon, output_path):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
