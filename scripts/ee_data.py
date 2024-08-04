import ee
import requests
import zipfile
import numpy as np
import rasterio
import os
import streamlit as st
from rasterio.mask import mask
from shapely.geometry import mapping

def download_ee_image(dataset, bands, region, filename, scale=30, dateMin=None, dateMax=None, crs='EPSG:4326'):
    print(f'Downloading {dataset} dataset ... ')
    
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

def download_elevation_data(polygon, dem_path):
    try:
        # Ensure output directories exist
        os.makedirs(os.path.dirname(dem_path), exist_ok=True)
        
        # Extract DEM based on polygon bounds
        bounds_combined = polygon.bounds
        west_c, south_c, east_c, north_c = bounds_combined
        
        # Ensure absolute paths
        absolute_dem_path = os.path.abspath(dem_path)
        
        # Clip the DEM based on bounds
        elevation.clip(bounds=(west_c, south_c, east_c, north_c), output=absolute_dem_path, product='SRTM1')
        dem = rio.open(absolute_dem_path)
        show(dem)
        
        # Clean up temporary files
        elevation.clean()
        
        st.write("Elevation data downloaded.")
        return absolute_dem_path
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
