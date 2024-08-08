import osmnx as ox
import streamlit as st
import geopandas as gpd
import os

def download_osm_data(polygon, tags, file_path):
    try:
        data = ox.features_from_polygon(polygon, tags)
        if data.empty:
            st.write(f"No data found for tags: {tags} in the selected area.")
            return None
        data.to_file(file_path, driver='GeoJSON')
        
        tag_key = list(tags.keys())[0]
        if tag_key == 'power' and tags[tag_key] == 'line':
            st.write(f"{len(data)} power lines identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def download_power_lines_data(polygon):
    power_lines_file = os.path.join('data', '2_downloaded_input_data', 'power_lines', 'osm_power_lines.geojson')
    os.makedirs(os.path.dirname(power_lines_file), exist_ok=True)
    return download_osm_data(polygon, {'power': 'line'}, power_lines_file)
