import elevation
import streamlit as st
import geopandas as gpd
from shapely.geometry import mapping
import os

def download_elevation_data(polygon):
    try:
        # Define the file path for saving elevation data
        file_path = os.path.join('data', '2_downloaded_input_data', 'elevation', 'elevation_data.tif')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Get the bounds of the polygon
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds

        # Clip the elevation data for the given bounds
        elevation.clip(bounds=(minx, miny, maxx, maxy), output=file_path)
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
    example_polygon = gpd.GeoSeries([{
        'type': 'Polygon',
        'coordinates': [[
            [12.35, 41.8],
            [12.65, 41.8],
            [12.65, 42.0],
            [12.35, 42.0],
            [12.35, 41.8]
        ]]
    }], crs="EPSG:4326")

    if st.button("Download Elevation Data"):
        download_elevation_data(example_polygon.unary_union)

    if st.button("Clean Elevation Cache"):
        clean_elevation_cache()
