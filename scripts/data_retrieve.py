import streamlit as st
import osmnx as ox
import requests
import geopandas as gpd
import json
from scripts.utils import initialize_earth_engine, create_combined_buildings_layer
import zipfile

def download_osm_data(polygon, tags, file_path):
    data = ox.geometries_from_polygon(polygon, tags)
    data.to_file(file_path, driver='GeoJSON')

def download_google_buildings(polygon, file_path):
    geom = ee.Geometry.Polygon(polygon.exterior.coords)
    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
        .filter(ee.Filter.intersects('.geo', geom))
    
    download_url = buildings.getDownloadURL('geojson')
    response = requests.get(download_url)
    with open(file_path, 'w') as f:
        json.dump(response.json(), f)

def zip_results(directory, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                zipf.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file), directory))

def show():
    st.title("Data Retrieve")
    st.write("Downloading data...")

    # Load the selected area
    with open('data/input/selected_area.geojson') as f:
        selected_area = json.load(f)
    
    polygon = gpd.GeoDataFrame.from_features([selected_area]).unary_union

    # Initialize Earth Engine
    initialize_earth_engine()

    # Define file paths
    buildings_file = 'data/output/buildings/combined_buildings.geojson'
    roads_file = 'data/output/roads/osm_roads.geojson'
    pois_file = 'data/output/poi/osm_pois.geojson'

    os.makedirs('data/output/buildings', exist_ok=True)
    os.makedirs('data/output/roads', exist_ok=True)
    os.makedirs('data/output/poi', exist_ok=True)

    # Download data
    progress = st.progress(0)
    status_text = st.empty()

    try:
        # Download buildings data
        status_text.text("Downloading OSM buildings data...")
        osm_buildings = ox.geometries_from_polygon(polygon, tags={'building': True})
        osm_buildings.to_file(buildings_file, driver='GeoJSON')
        progress.progress(0.3)

        status_text.text("Downloading Google buildings data...")
        download_google_buildings(polygon, 'data/output/buildings/google_buildings.geojson')
        progress.progress(0.6)

        # Combine OSM and Google buildings data
        status_text.text("Combining buildings data...")
        google_buildings = gpd.read_file('data/output/buildings/google_buildings.geojson')
        combined_buildings = create_combined_buildings_layer(osm_buildings, google_buildings)
        combined_buildings.to_file(buildings_file, driver='GeoJSON')
        progress.progress(0.7)

        # Download roads data
        status_text.text("Downloading OSM roads data...")
        download_osm_data(polygon, {'highway': True}, roads_file)
        progress.progress(0.8)

        # Download points of interest data
        status_text.text("Downloading OSM points of interest data...")
        download_osm_data(polygon, {'amenity': True}, pois_file)
        progress.progress(0.9)

        # Zip all results
        status_text.text("Zipping results...")
        zip_results('data/output', 'data/output/results.zip')
        progress.progress(1.0)

        st.success("Data download complete. You can now proceed to the next section.")
        st.download_button('Download All Results', 'data/output/results.zip')

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the data retrieve page
if __name__ == "__main__":
    show()
