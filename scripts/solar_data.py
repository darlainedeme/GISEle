import requests
import os
import streamlit as st

def download_solar_data(polygon, solar_path):
    try:
        bounds_combined = polygon.bounds
        west_c, south_c, east_c, north_c = bounds_combined
        
        url = f"https://globalsolaratlas.info/download/solar_resource_and_pv?latitude={south_c}&longitude={west_c}&maxLatitude={north_c}&maxLongitude={east_c}&dataset=solar_resource_and_pv&api_key=YOUR_API_KEY"
        
        response = requests.get(url)
        if response.status_code == 200:
            with open(solar_path, 'wb') as fd:
                fd.write(response.content)
            st.write("Solar data downloaded.")
            return solar_path
        else:
            st.write("No solar data found for the selected area.")
            return None
    except Exception as e:
        st.error(f"Error downloading solar data: {e}")
        return None
