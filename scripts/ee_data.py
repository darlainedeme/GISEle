# Import necessary libraries
import ee
import geemap
import os
import requests

# Function to download an image from Google Earth Engine
def download_ee_image(collection_id, bands, polygon, file_path, scale=30, dateMin='2020-01-01', dateMax='2020-12-31'):
    try:
        # Initialize the Earth Engine API
        ee.Initialize()

        # Define the Area of Interest (AOI)
        aoi = ee.Geometry.Polygon(polygon)

        # Filter the image collection
        collection = ee.ImageCollection(collection_id) \
            .filterDate(dateMin, dateMax) \
            .filterBounds(aoi) \
            .select(bands)

        # Get the first image from the collection
        image = collection.first()

        # Get the download URL
        url = image.getDownloadURL({
            'scale': scale,
            'crs': 'EPSG:4326',
            'region': aoi,
            'format': 'GEO_TIFF'
        })

        # Download the image
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024):
                fd.write(chunk)

        print(f"Image downloaded successfully and saved to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import streamlit as st

    # Define the polygon (example coordinates, replace with your own)
    polygon = [[[85.3240, 27.7172], [85.3240, 27.7272], [85.3340, 27.7272], [85.3340, 27.7172], [85.3240, 27.7172]]]

    # Selected datasets (example condition, replace with your own logic)
    selected_datasets = ["Satellite"]

    # Streamlit status text and progress bar
    status_text = st.empty()
    progress = st.progress(0)

    if "Satellite" in selected_datasets:
        status_text.text("Downloading satellite data...")
        satellite_file = 'data/output/satellite/satellite_image.tif'
        os.makedirs('data/output/satellite', exist_ok=True)
        download_ee_image('COPERNICUS/S2_SR_HARMONIZED', ['B4', 'B3', 'B2'], polygon, satellite_file, scale=30, dateMin='2020-04-01', dateMax='2020-04-30')
        st.write("Satellite data downloaded for the selected area.")
        progress.progress(0.9)
