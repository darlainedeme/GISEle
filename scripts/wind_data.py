import requests
import os
import streamlit as st

def download_wind_data(polygon, wind_path):
    try:
        bounds_combined = polygon.bounds
        west_c, south_c, east_c, north_c = bounds_combined
        
        url = f"https://globalwindatlas.info/api/area/download?lat1={south_c}&lon1={west_c}&lat2={north_c}&lon2={east_c}&dataset=global_wind&api_key=YOUR_API_KEY"
        
        response = requests.get(url)
        if response.status_code == 200:
            with open(wind_path, 'wb') as fd:
                fd.write(response.content)
            st.write("Wind data downloaded.")
            return wind_path
        else:
            st.write("No wind data found for the selected area.")
            return None
    except Exception as e:
        st.error(f"Error downloading wind data: {e}")
        return None
