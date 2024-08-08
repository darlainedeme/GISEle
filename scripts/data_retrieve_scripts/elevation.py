import requests
import json
import geopandas as gpd
import pandas as pd
import streamlit as st
import os
import numpy as np
from shapely.geometry import Point

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

def create_grid_within_polygon(polygon, spacing=0.00027):  # Approximately 30 meters in degrees
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)
    points = [Point(x, y) for x in x_coords for y in y_coords]
    points_within_polygon = [point for point in points if polygon.contains(point)]
    return points_within_polygon

def download_elevation_data(polygon):
    elevation_file = os.path.join('data', '2_downloaded_input_data', 'elevation', 'elevation_data.csv')
    os.makedirs(os.path.dirname(elevation_file), exist_ok=True)
    
    # Create a grid of points within the polygon
    points_within_polygon = create_grid_within_polygon(polygon)
    locations = [[point.x, point.y] for point in points_within_polygon]

    st.write(f"Fetching elevation data for {len(locations)} locations")
    elevations = get_elevation_data(locations)

    if elevations:
        # Ensure we get the first elevation value if multiple are returned
        elevation_values = [e[0] if isinstance(e, list) else e for e in elevations]
        elevation_data = {
            'longitude': [loc[0] for loc in locations],
            'latitude': [loc[1] for loc in locations],
            'elevation': elevation_values
        }
        df = pd.DataFrame(elevation_data)
        df.to_csv(elevation_file, index=False)
        st.write(f"Elevation data saved to {elevation_file}")
    else:
        st.write("No elevation data available for the selected area.")

# If necessary, add any additional setup or import statements here
