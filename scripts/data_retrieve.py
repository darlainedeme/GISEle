import streamlit as st
import osmnx as ox
import requests
import geopandas as gpd
import json
import ee
from scripts.utils import initialize_earth_engine, create_combined_buildings_layer
import zipfile
import os 

def download_osm_data(polygon, tags, file_path):
    data = ox.features_from_polygon(polygon, tags)
    data.to_file(file_path, driver='GeoJSON')

    if 'building' in tags:
        st.write(f"{len(data)} buildings identified")
    elif 'highway' in tags:
        if data.crs.is_geographic:
            data = data.to_crs(epsg=3857)  # Reproject to a projected CRS for accurate length calculation
        total_km = data.geometry.length.sum() / 1000
        st.write(f"{total_km:.2f} km of roads identified")
    elif 'amenity' in tags:
        st.write(f"{len(data)} points of interest identified")
    elif 'natural' in tags and tags['natural'] == 'water':
        st.write(f"{len(data)} water bodies identified")

def download_google_buildings(polygon, file_path):
    geom = ee.Geometry.Polygon(polygon.exterior.coords[:])
    buildings = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons') \
        .filter(ee.Filter.intersects('.geo', geom))
    
    download_url = buildings.getDownloadURL('geojson')
    response = requests.get(download_url)
    with open(file_path, 'w') as f:
        json.dump(response.json(), f)
    
    google_buildings = gpd.read_file(file_path)
    st.write(f"{len(google_buildings)} Google buildings identified")

def download_dem(polygon, file_path):
    geom = ee.Geometry.Polygon(polygon.exterior.coords[:])
    dem = ee.Image('USGS/SRTMGL1_003').clip(geom)
    dem_url = dem.getDownloadURL({
        'scale': 30,
        'crs': 'EPSG:4326',
        'region': geom.toGeoJSONString()
    })
    response = requests.get(dem_url)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    st.write("Digital Elevation Model (DEM) downloaded")

def zip_results(files, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in files:
            zipf.write(file_path, os.path.basename(file_path))

def show():
    st.title("Data Retrieve")
    st.write("Downloading data...")

    # Load the selected area
    with open('data/input/selected_area.geojson') as f:
        selected_area = json.load(f)

    gdf = gpd.GeoDataFrame.from_features(selected_area["features"])
    polygon = gdf.geometry.union_all()

    # Initialize Earth Engine
    initialize_earth_engine()

    # Define file paths
    osm_buildings_file = 'data/output/buildings/osm_buildings.geojson'
    google_buildings_file = 'data/output/buildings/google_buildings.geojson'
    combined_buildings_file = 'data/output/buildings/combined_buildings.geojson'
    roads_file = 'data/output/roads/osm_roads.geojson'
    pois_file = 'data/output/poi/osm_pois.geojson'
    water_bodies_file = 'data/output/water_bodies/osm_water_bodies.geojson'
    dem_file = 'data/output/dem/dem.tif'

    os.makedirs('data/output/buildings', exist_ok=True)
    os.makedirs('data/output/roads', exist_ok=True)
    os.makedirs('data/output/poi', exist_ok=True)
    os.makedirs('data/output/water_bodies', exist_ok=True)
    os.makedirs('data/output/dem', exist_ok=True)

    # Download data
    progress = st.progress(0)
    status_text = st.empty()

    try:
        # Download buildings data
        status_text.text("Downloading OSM buildings data...")
        download_osm_data(polygon, {'building': True}, osm_buildings_file)
        progress.progress(0.2)

        status_text.text("Downloading Google buildings data...")
        download_google_buildings(polygon, google_buildings_file)
        progress.progress(0.4)

        # Combine OSM and Google buildings data
        status_text.text("Combining buildings data...")
        with open(google_buildings_file) as f:
            google_buildings_geojson = json.load(f)
        osm_buildings = gpd.read_file(osm_buildings_file)
        combined_buildings = create_combined_buildings_layer(osm_buildings, google_buildings_geojson)
        combined_buildings.to_file(combined_buildings_file, driver='GeoJSON')
        st.write(f"{len(combined_buildings)} buildings in the combined dataset")
        progress.progress(0.6)

        # Download roads data
        status_text.text("Downloading OSM roads data...")
        download_osm_data(polygon, {'highway': True}, roads_file)
        progress.progress(0.7)

        # Download points of interest data
        status_text.text("Downloading OSM points of interest data...")
        download_osm_data(polygon, {'amenity': True}, pois_file)
        progress.progress(0.8)

        # Download water bodies data
        status_text.text("Downloading OSM water bodies data...")
        download_osm_data(polygon, {'natural': 'water'}, water_bodies_file)
        progress.progress(0.9)

        # Download Digital Elevation Model (DEM) data
        status_text.text("Downloading Digital Elevation Model (DEM) data...")
        download_dem(polygon, dem_file)
        progress.progress(0.95)

        # Zip all results
        status_text.text("Zipping results...")
        zip_files = [osm_buildings_file, google_buildings_file, combined_buildings_file, roads_file, pois_file, water_bodies_file, dem_file]
        zip_results(zip_files, 'data/output/results.zip')
        progress.progress(1.0)

        st.success("Data download complete. You can now proceed to the next section.")
        with open('data/output/results.zip', 'rb') as f:
            st.download_button('Download All Results', f, file_name='results.zip')

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the data retrieve page
if __name__ == "__main__":
    show()
