import streamlit as st
from geopy.geocoders import Nominatim
from scripts.utils import create_map, uploaded_file_to_gdf
import json
import os

def save_geojson(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def show():
    st.title("Area Selection")
    which_modes = ['By address', 'By coordinates', 'Upload file']
    which_mode = st.sidebar.selectbox('Select mode', which_modes, index=0)

    if which_mode == 'By address':  
        geolocator = Nominatim(user_agent="example app")
        address = st.sidebar.text_input('Enter your address:', value='B12 Bovisa') 

        if address:
            try:
                with st.spinner('Fetching location...'):
                    location = geolocator.geocode(address)
                    if location:
                        st.session_state.latitude = location.latitude
                        st.session_state.longitude = location.longitude
                        st.session_state.geojson_data = None
                        st.session_state.combined_buildings = None
                        st.session_state.osm_roads = None
                        st.session_state.osm_pois = None
                        st.session_state.missing_layers = []
                        
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
        latitude = st.sidebar.text_input('Latitude:', value='45.5065') 
        longitude = st.sidebar.text_input('Longitude:', value='9.1598') 
        
        if latitude and longitude:
            try:
                with st.spinner('Creating map...'):
                    lat = float(latitude)
                    lon = float(longitude)
                    st.session_state.latitude = lat
                    st.session_state.longitude = lon
                    st.session_state.geojson_data = None
                    st.session_state.combined_buildings = None
                    st.session_state.osm_roads = None
                    st.session_state.osm_pois = None
                    st.session_state.missing_layers = []

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
                with st.spinner('Processing file...'):
                    geojson_data = json.load(uploaded_file)
                    gdf = uploaded_file_to_gdf(uploaded_file)
                    
                    if gdf.empty:
                        st.error("Uploaded file is empty or not valid GeoJSON.")
                    
                    centroid = gdf.geometry.unary_union.centroid
                    st.session_state.latitude = centroid.y
                    st.session_state.longitude = centroid.x
                    st.session_state.geojson_data = geojson_data
                    st.session_state.combined_buildings = None
                    st.session_state.osm_roads = None
                    st.session_state.osm_pois = None
                    st.session_state.missing_layers = []

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

# Display the area selection page
if __name__ == "__main__":
    show()
