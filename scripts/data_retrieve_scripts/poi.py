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
        if tag_key == 'amenity':
            st.write(f"{len(data)} points of interest identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def download_poi_data(polygon):
    poi_file = os.path.join('data', '2_downloaded_input_data', 'poi', 'osm_pois.geojson')
    os.makedirs(os.path.dirname(poi_file), exist_ok=True)
    return download_osm_data(polygon, {'amenity': True}, poi_file)
