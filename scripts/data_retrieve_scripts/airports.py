import osmnx as ox
import streamlit as st
import geopandas as gpd
import os

def download_osm_data(polygon, tags, file_path):
    try:
        data = ox.features_from_polygon(polygon, tags)
        if data.empty:
            if 'aeroway' in tags and tags['aeroway'] == 'aerodrome':
                st.write("No airports found in the selected area.")
            return None
        data.to_file(file_path, driver='GeoJSON')

        if 'aeroway' in tags and tags['aeroway'] == 'aerodrome':
            st.write(f"{len(data)} airports identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def download_airports_data(polygon):
    airports_file = os.path.join('data', '2_downloaded_input_data', 'airports', 'osm_airports.geojson')
    os.makedirs(os.path.dirname(airports_file), exist_ok=True)
    return download_osm_data(polygon, {'aeroway': 'aerodrome'}, airports_file)
