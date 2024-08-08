import osmnx as ox
import streamlit as st
import geopandas as gpd
import os

def download_osm_data(polygon, tags, file_path):
    try:
        data = ox.features_from_polygon(polygon, tags)
        if data.empty:
            if 'place' in tags and tags['place'] == 'city':
                st.write("No cities found in the selected area.")
            return None
        data.to_file(file_path, driver='GeoJSON')

        if 'place' in tags and tags['place'] == 'city':
            st.write(f"{len(data)} cities identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def download_cities_data(polygon):
    cities_file = os.path.join('data', '2_downloaded_input_data', 'cities', 'osm_cities.geojson')
    os.makedirs(os.path.dirname(cities_file), exist_ok=True)
    return download_osm_data(polygon, {'place': 'city'}, cities_file)
