import requests
import json
import geopandas as gpd
import streamlit as st
import os
import pandas as pd

def get_elevation_data(locations):
    url = "https://api.ellipsis-drive.com/v3/path/77239b78-ff95-4c30-a90e-0428964a0f00/raster/timestamp/83a6860b-3c34-4a53-9d3f-d123019eff7c/location"
    params = {
        'locations': json.dumps(locations),
        'epsg': '4326'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        elevations = response.json()
        return elevations
    except requests.RequestException as e:
        st.error(f"Error fetching elevation data: {e}")
        return None

def download_elevation_data(polygon):
    elevation_file = os.path.join('data', '2_downloaded_input_data', 'elevation', 'elevation_data.csv')
    os.makedirs(os.path.dirname(elevation_file), exist_ok=True)
    
    # Generate a list of locations (lon, lat) from the polygon's bounding box or centroid
    centroid = polygon.centroid
    locations = [[centroid.x, centroid.y]]

    st.write(f"Fetching elevation data for locations: {locations}")
    elevations = get_elevation_data(locations)

    if elevations:
        elevation_data = {
            'longitude': [loc[0] for loc in locations],
            'latitude': [loc[1] for loc in locations],
            'elevation': elevations
        }
        df = pd.DataFrame(elevation_data)
        df.to_csv(elevation_file, index=False)
        st.write(f"Elevation data saved to {elevation_file}")
    else:
        st.write("No elevation data available for the selected area.")
