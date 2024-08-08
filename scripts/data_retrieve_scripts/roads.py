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
        if tag_key == 'highway':
            st.write(f"{len(data)} roads identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def download_roads_data(polygon, buffer_polygon):
    roads_file = os.path.join('data', '2_downloaded_input_data', 'roads', 'osm_roads.geojson')
    roads_buffer_file = os.path.join('data', '2_downloaded_input_data', 'roads', 'osm_roads_buffer.geojson')
    os.makedirs(os.path.dirname(roads_file), exist_ok=True)
    os.makedirs(os.path.dirname(roads_buffer_file), exist_ok=True)
    
    # Download and save roads data
    download_osm_data(polygon, {'highway': True}, roads_file)
    
    # Download and save buffered roads data
    download_osm_data(buffer_polygon, {'highway': True}, roads_buffer_file)
