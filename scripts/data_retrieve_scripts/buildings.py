import requests
import json
import streamlit as st
import geopandas as gpd
import ee
import os
import osmnx as ox
import pandas as pd

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
        google_buildings['source'] = 'google'
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
        data['source'] = 'osm'
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None

def create_combined_buildings_layer(osm_buildings, google_buildings_geojson):
    google_buildings = gpd.GeoDataFrame.from_features(google_buildings_geojson['features'])
    google_buildings['source'] = 'google'
    osm_buildings['source'] = 'osm'
    combined_buildings = gpd.GeoDataFrame(pd.concat([osm_buildings, google_buildings], ignore_index=True))
    return combined_buildings

def download_buildings_data(polygon):
    osm_buildings_file = os.path.join('data', '2_downloaded_input_data', 'buildings', 'osm_buildings.geojson')
    google_buildings_file = os.path.join('data', '2_downloaded_input_data', 'buildings', 'google_buildings.geojson')
    combined_buildings_file = os.path.join('data', '2_downloaded_input_data', 'buildings', 'combined_buildings.geojson')

    os.makedirs(os.path.dirname(osm_buildings_file), exist_ok=True)

    osm_buildings_path = download_osm_data(polygon, {'building': True}, osm_buildings_file)
    google_buildings_path = download_google_buildings(polygon, googlimport geopandas as gpd
import pandas as pd
import json
import os
import streamlit as st

def create_combined_buildings_layer(osm_buildings, google_buildings_geojson):
    # Ensure both GeoDataFrames are in the same CRS
    osm_buildings = osm_buildings.to_crs(epsg=4326)
    google_buildings = gpd.GeoDataFrame.from_features(google_buildings_geojson["features"]).set_crs(epsg=4326)

    # Label sources
    osm_buildings['source'] = 'osm'
    google_buildings['source'] = 'google'

    # Remove Google buildings that touch OSM buildings
    osm_dissolved = osm_buildings.geometry.unary_union

    # Filter Google buildings that do not intersect with OSM buildings
    filtered_google = google_buildings[~google_buildings.intersects(osm_dissolved)]

    # Combine OSM buildings and filtered Google buildings
    combined_buildings = gpd.GeoDataFrame(pd.concat([osm_buildings, filtered_google], ignore_index=True))

    return combined_buildings

def download_buildings_data(polygon):
    osm_buildings_file = os.path.join('data', '2_downloaded_input_data', 'buildings', 'osm_buildings.geojson')
    google_buildings_file = os.path.join('data', '2_downloaded_input_data', 'buildings', 'google_buildings.geojson')
    combined_buildings_file = os.path.join('data', '2_downloaded_input_data', 'buildings', 'combined_buildings.geojson')

    os.makedirs(os.path.dirname(osm_buildings_file), exist_ok=True)

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

# Ensure the download functions for OSM and Google buildings are defined
def download_osm_data(polygon, tags, file_path):
    import osmnx as ox
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

def download_google_buildings(polygon, file_path):
    import requests
    import ee
    ee.Initialize()

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
