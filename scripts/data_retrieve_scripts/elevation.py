import ee
import geemap
import streamlit as st
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import ee

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()

def download_elevation_data(polygon):
    elevation_file = os.path.join('data', '2_downloaded_input_data', 'elevation', 'elevation_data.tif')
    os.makedirs(os.path.dirname(elevation_file), exist_ok=True)
    
    # Convert the polygon to an Earth Engine object
    geo_json = json.loads(polygon.to_json())
    ee_polygon = ee.Geometry.Polygon(geo_json['features'][0]['geometry']['coordinates'])
    
    # Load the elevation dataset
    srtm = ee.Image('CGIAR/SRTM90_V4')
    
    # Clip the dataset to the polygon
    elevation_clip = srtm.clip(ee_polygon)
    
    # Define the export task
    task = ee.batch.Export.image.toDrive(
        image=elevation_clip,
        description='elevation_export',
        folder='earthengine',
        fileNamePrefix='elevation_data',
        region=ee_polygon,
        scale=30,
        crs='EPSG:4326'
    )
    
    # Start the export task
    task.start()
    
    # Monitor the task status
    st.write("Exporting elevation data. This may take a few minutes...")
    while task.active():
        st.write('Polling for task (id: {}).'.format(task.id))
        time.sleep(30)
    
    # Check if the task completed successfully
    if task.status()['state'] == 'COMPLETED':
        st.write("Elevation data exported successfully.")
        st.write("Download the file from your Google Drive.")
    else:
        st.error("Error exporting elevation data: {}".format(task.status()))
    
# If necessary, add any additional setup or import statements here
