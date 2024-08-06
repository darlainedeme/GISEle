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
    import requests
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

def download_tif(area, crs, scale, image, out_path):
    """
    Download data from Earth Engine
    :param area: GeoDataFrame or shapely Polygon with the polygon of interest area
    :param crs: str with crs of the project
    :param scale: int with pixel size in meters
    :param image: image from the wanted database in Earth Image
    :param out_path: str with output path
    :return:
    """
    min_x, min_y, max_x, max_y = area.bounds
    path = image.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:' + str(crs),
        'region': [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]
    })
    print("Download URL:", path)
    download_url(path, out_path)
    return

def download_elevation_data(polygon, zip_path, dem_path):
    try:
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        
        # Define the Earth Engine image for SRTM
        srtm_image = ee.Image('USGS/SRTMGL1_003')

        # Define the CRS and scale
        crs = 4326  # WGS84
        scale = 30  # SRTM resolution is approximately 30 meters

        # Download the zip using Earth Engine
        download_tif(polygon, crs, scale, srtm_image, zip_path)

        # Extract the zip file
        extracted_files = extract_zip(zip_path, os.path.dirname(dem_path))
        print("Extracted files:", extracted_files)


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
