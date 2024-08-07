import requests
import json
import streamlit as st
import geopandas as gpd
import ee
import osmnx as ox
import os

# Initialize Earth Engine
ee.Initialize()

def download_google_buildings(polygon, file_path):
    try:
        geom = ee.Geometry.Polygon(polygon.exterior.coords[:])
        buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
            .filter(ee.Filter.intersects('.geo', geom))
        
        download_url = buildings.getDownloadURL('geojson')
        response = requests.get(download_url)
        if response.status_code != 200:
            st.write("No Google buildings found in the selected area.")
            return None

        with open(file_path, 'w') as f:
            json.dump(response.json(), f)
        
        google_buildings = gpd.read_file(file_path)
        st.write(f"{len(google_buildings)} Google buildings identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading Google buildings: {e}")
        return None

def download_osm_data(polygon, tags, file_path):
    try:
        data = ox.features_from_polygon(polygon, tags)
        if data.empty:
            st.write(f"No data found for tags: {tags} in the selected area.")
            return None
        data.to_file(file_path, driver='GeoJSON')
        
        tag_key = list(tags.keys())[0]
        if tag_key == 'building':
            st.write(f"{len(data)} buildings identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def create_combined_buildings_layer(osm_buildings, google_buildings_geojson):
    google_buildings = gpd.GeoDataFrame.from_features(google_buildings_geojson['features'])
    combined_buildings = gpd.GeoDataFrame(pd.concat([osm_buildings, google_buildings], ignore_index=True))
    return combined_buildings

def download_buildings_data(polygon):
    osm_buildings_file = 'data/2_downloaded_input_data/buildings/osm_buildings.geojson'
    google_buildings_file = 'data/2_downloaded_input_data/buildings/google_buildings.geojson'
    combined_buildings_file = 'data/2_downloaded_input_data/buildings/combined_buildings.geojson'

    os.makedirs('data/2_downloaded_input_data/buildings', exist_ok=True)

    osm_buildings_path = download_osm_data(polygon, {'building': True}, osm_buildings_file)
    google_buildings_path = download_google_buildings(polygon, google_buildings_file)

    if osm_buildings_path and google_buildings_path:
        with open(google_buildings_path) as f:
            google_buildings_geojson = json.load(f)
        osm_buildings = gpd.read_file(osm_buildings_path)
        combined_buildings = create_combined_buildings_layer(osm_buildings, google_buildings_geojson)
        combined_buildings.to_file(combined_buildings_file, driver='GeoJSON')
        st.write(f"Combined buildings dataset saved to {combined_buildings_file}")
    else:
        st.write("Skipping buildings combination due to missing data.")
