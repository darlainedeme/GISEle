# Import necessary libraries
import ee
import geemap
import os
import requests

# Function to download an image from Google Earth Engine
def download_ee_image(collection_id, bands, polygon, file_path, scale=30, dateMin='2020-01-01', dateMax='2020-01-02'):
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