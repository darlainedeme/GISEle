import streamlit as st
from geopy.geocoders import Nominatim
from scripts.utils import create_map, uploaded_file_to_gdf
import json
import os
import geopandas as gpd

def save_geojson(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_example_geojson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def reset_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def show():
    st.title("Area Selection")

    if 'mode' not in st.session_state:
        st.session_state.mode = 'By address'

    which_modes = ['By address', 'By coordinates', 'Upload file', 'Examples']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=which_modes.index(st.session_state.mode), key='mode')

    if st.sidebar.button('Reset'):
        reset_session_state()
        st.experimental_rerun()

    if which_mode == 'By address':
        if 'address' not in st.session_state:
            st.session_state.address = 'B12 Bovisa'
        
        geolocator = Nominatim(user_agent="example app")
        address = st.sidebar.text_input('Enter your address:', value=st.session_state.address, key='address')

        if address:
            try:
                if st.sidebar.button('Fetch location'):
                    with st.spinner('Fetching location...'):
                        location = geolocator.geocode(address)
                        if location:
                            st.session_state.latitude = location.latitude
                            st.session_state.longitude = location.longitude
                            st.session_state.geojson_data = None

                            # Save selected location as GeoJSON
                            selected_area = {
                                "type": "FeatureCollection",
                                "features": [
                                    {
                                        "type": "Feature",
                                        "geometry": {
                                            "type": "Point",
                                            "coordinates": [location.longitude, location.latitude]
                                        },
                                        "properties": {}
                                    }
                                ]
                            }
                            os.makedirs('data/input', exist_ok=True)
                            save_geojson(selected_area, 'data/input/selected_area.geojson')

                            create_map(location.latitude, location.longitude)
                        else:
                            st.error("Could not geocode the address.")
            except Exception as e:
                st.error(f"Error fetching location: {e}")

    elif which_mode == 'By coordinates':
        if 'latitude' not in st.session_state:
            st.session_state.latitude = '45.5065'
        if 'longitude' not in st.session_state:
            st.session_state.longitude = '9.1598'
        
        latitude = st.sidebar.text_input('Latitude:', value=st.session_state.latitude, key='latitude')
        longitude = st.sidebar.text_input('Longitude:', value=st.session_state.longitude, key='longitude')

        if latitude and longitude:
            try:
                if st.sidebar.button('Create map'):
                    with st.spinner('Creating map...'):
                        lat = float(latitude)
                        lon = float(longitude)
                        st.session_state.latitude = lat
                        st.session_state.longitude = lon
                        st.session_state.geojson_data = None

                        # Save selected location as GeoJSON
                        selected_area = {
                            "type": "FeatureCollection",
                            "features": [
                                {
                                    "type": "Feature",
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [lon, lat]
                                    },
                                    "properties": {}
                                }
                            ]
                        }
                        os.makedirs('data/input', exist_ok=True)
                        save_geojson(selected_area, 'data/input/selected_area.geojson')

                        create_map(lat, lon)
            except Exception as e:
                st.error(f"Error with coordinates: {e}")

    elif which_mode == 'Upload file':
        uploaded_file = st.file_uploader("Upload GeoJSON file", type="geojson")

        if uploaded_file:
            try:
                if st.sidebar.button('Process file'):
                    with st.spinner('Processing file...'):
                        geojson_data = json.load(uploaded_file)
                        gdf = uploaded_file_to_gdf(uploaded_file)
                        
                        if gdf.empty:
                            st.error("Uploaded file is empty or not valid GeoJSON.")
                        
                        centroid = gdf.geometry.union_all.centroid
                        st.session_state.latitude = centroid.y
                        st.session_state.longitude = centroid.x
                        st.session_state.geojson_data = geojson_data

                        # Save the uploaded GeoJSON to a file
                        os.makedirs('data/input', exist_ok=True)
                        with open('data/input/selected_area.geojson', 'w') as f:
                            json.dump(geojson_data, f)

                        create_map(centroid.y, centroid.x, geojson_data)
                        st.success("Map created successfully!")
            except KeyError as e:
                st.error(f"Error processing file: {e}")
            except IndexError as e:
                st.error(f"Error processing file: {e}")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    elif which_mode == 'Examples':
        examples_path = 'data/examples'
        example_files = [f.split('.')[0] for f in os.listdir(examples_path) if f.endswith('.geojson')]
        selected_example = st.sidebar.selectbox('Select example', example_files)

        if selected_example:
            try:
                if st.sidebar.button('Load example'):
                    with st.spinner('Loading example...'):
                        example_geojson = load_example_geojson(os.path.join(examples_path, selected_example + '.geojson'))
                        gdf = gpd.GeoDataFrame.from_features(example_geojson["features"])
                        centroid = gdf.geometry.union_all.centroid
                        st.session_state.latitude = centroid.y
                        st.session_state.longitude = centroid.x
                        st.session_state.geojson_data = example_geojson

                        create_map(centroid.y, centroid.x, example_geojson)
                        st.success("Example map created successfully!")
            except Exception as e:
                st.error(f"Error loading example: {e}")

if __name__ == "__main__":
    show()
