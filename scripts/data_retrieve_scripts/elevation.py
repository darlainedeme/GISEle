import elevation
import streamlit as st
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.merge import merge
import os

def download_elevation_data(polygon):
    try:
        # Define the file path for saving elevation data
        file_path = os.path.join('data', '2_downloaded_input_data', 'elevation', 'elevation_data.tif')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Get the bounds of the polygon
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds

        # Clip the elevation data for the given bounds using elevation.clip
        elevation.clip(bounds=(minx, miny, maxx, maxy), output=file_path)
        
        # Reproject and save the clipped elevation data using rasterio
        with rasterio.open(file_path) as src:
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'height': src.height,
                'width': src.width,
                'transform': src.transform,
                'crs': 'EPSG:4326'
            })
            
            with rasterio.open(file_path, 'w', **meta) as dst:
                dst.write(src.read())
        
        st.write(f"Elevation data saved to {file_path}")
        return file_path
    except Exception as e:
        st.error(f"Error downloading elevation data: {e}")
        return None

def clean_elevation_cache():
    try:
        elevation.clean()
        st.write("Elevation cache cleaned.")
    except Exception as e:
        st.error(f"Error cleaning elevation cache: {e}")

# Example usage within Streamlit
if __name__ == "__main__":
    st.title("Download Elevation Data")

    # Assume `polygon` is provided as input; this is just an example
    example_polygon = gpd.GeoSeries([box(12.35, 41.8, 12.65, 42.0)], crs="EPSG:4326")

    if st.button("Download Elevation Data"):
        download_elevation_data(example_polygon.unary_union)

    if st.button("Clean Elevation Cache"):
        clean_elevation_cache()
