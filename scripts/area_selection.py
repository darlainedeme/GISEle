import streamlit as st
from geopy.geocoders import Nominatim
from scripts.utils import create_map, uploaded_file_to_gdf
import json
import os
import geopandas as gpd

def save_geojson(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_example_files():
    example_dir = 'data/examples'
    if not os.path.exists(example_dir):
        os.makedirs(example_dir)
    files = [f.split('.')[0] for f in os.listdir(example_dir) if f.endswith('.geojson')]
    return files

def show():
    # Ensure session state is set up
    if 'mode' not in st.session_state:
        st.session_state.mode = 'By address'
    if 'address' not in st.session_state:
        st.session_state.address = 'B12 Bovisa'
    if 'latitude' not in st.session_state:
        st.session_state.latitude = '45.5065'
    if 'longitude' not in st.session_state:
        st.session_state.longitude = '9.1598'
    if 'geojson_data' not in st.session_state:
        st.session_state.geojson_data = None
    if 'example_file' not in st.session_state:
        st.session_state.example_file = None

    which_modes = ['By address', 'By coordinates', 'Upload file', 'Examples']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=which_modes.index(st.session_state.mode), key='mode_select')

    if which_mode != st.session_state.mode:
        st.session_state.mode = which_mode

    if which_mode == 'By address':
        geolocator = Nominatim(user_agent="example app")
        address = st.sidebar.text_input('Enter your address:', value=st.session_state.address, key='address_input')

        if address != st.session_state.address:
            st.session_state.address = address

        if address:
            try:
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
        latitude = st.sidebar.text_input('Latitude:', value=st.session_state.latitude, key='latitude_input')
        longitude = st.sidebar.text_input('Longitude:', value=st.session_state.longitude, key='longitude_input')

        if latitude != st.session_state.latitude:
            st.session_state.latitude = latitude
        if longitude != st.session_state.longitude:
            st.session_state.longitude = longitude

        if latitude and longitude:
            try:
                with st.spinner('Creating map...'):
                    lat = float(latitude)
                    lon = float(longitude)
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
        uploaded_file = st.file_uploader("Upload GeoJSON file", type="geojson", key='geojson_uploader')

        if uploaded_file:
            try:
                with st.spinner('Processing file...'):
                    geojson_data = json.load(uploaded_file)
                    gdf = uploaded_file_to_gdf(uploaded_file)
                    
                    if gdf.empty:
                        st.error("Uploaded file is empty or not valid GeoJSON.")

                    centroid = gdf.geometry.unary_union.centroid
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
        examples = load_example_files()
        example_file = st.sidebar.selectbox('Select example', examples, index=examples.index(st.session_state.example_file) if st.session_state.example_file in examples else 0, key='example_select')

        if example_file != st.session_state.example_file:
            st.session_state.example_file = example_file

        if example_file:
            try:
                with st.spinner('Loading example...'):
                    example_path = f'data/examples/{example_file}.geojson'
                    with open(example_path) as f:
                        geojson_data = json.load(f)
                    gdf = gpd.read_file(example_path)
                    
                    if gdf.empty:
                        st.error("Example file is empty or not valid GeoJSON.")

                    centroid = gdf.geometry.unary_union.centroid
                    st.session_state.latitude = centroid.y
                    st.session_state.longitude = centroid.x
                    st.session_state.geojson_data = geojson_data

                    create_map(centroid.y, centroid.x, geojson_data)
                    st.success("Example loaded successfully!")
            except KeyError as e:
                st.error(f"Error loading example: {e}")
            except IndexError as e:
                st.error(f"Error loading example: {e}")
            except Exception as e:
                st.error(f"Error loading example: {e}")

# Display the area selection page
if __name__ == "__main__":
    show()
