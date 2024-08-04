import requests
import json
import streamlit as st
import geopandas as gpd
import ee

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
