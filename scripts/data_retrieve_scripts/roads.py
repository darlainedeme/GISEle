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

def download_roads_data(polygon):
    roads_file = os.path.join('data', '2_downloaded_input_data', 'roads', 'osm_roads.geojson')
    os.makedirs(os.path.dirname(roads_file), exist_ok=True)
    return download_osm_data(polygon, {'highway': True}, roads_file)

def download_roads_buffer_data(buffer_polygon):
    roads_buffer_file = os.path.join('data', '2_downloaded_input_data', 'roads', 'osm_roads_buffer.geojson')
    os.makedirs(os.path.dirname(roads_buffer_file), exist_ok=True)
    return download_osm_data(buffer_polygon, {'highway': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']}, roads_buffer_file)
