import requests
import os
import streamlit as st
import requests
import json
import pandas as pd
import geopandas as gpd
import time
import os
import streamlit as st

def import_pv_data(lat, lon, tilt_angle):
    token = '556d9ea27f35f2e26ac9ce1552a3f702e35a8596'
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    s.headers = {'Authorization': 'Token ' + token}
    url = api_base + 'data/pv'

    args = {
        'lat': lat,
        'lon': lon,
        'date_from': '2023-01-01',
        'date_to': '2023-12-31',
        'dataset': 'merra2',
        'local_time': True,
        'capacity': 1.0,
        'system_loss': 0,
        'tracking': 0,
        'tilt': tilt_angle,
        'azim': 180,
        'format': 'json',
    }

    while True:
        try:
            r = s.get(url, params=args)
            parsed_response = json.loads(r.text)
            break
        except:
            st.write('Problem with importing PV data. Software is in sleep mode for 30 seconds.')
            time.sleep(30)
    
    data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
    st.write("Solar Data imported")

    return data

def download_solar_data(polygon):
    solar_file = os.path.join('data', '2_downloaded_input_data', 'solar', 'solar_data.csv')
    os.makedirs(os.path.dirname(solar_file), exist_ok=True)

    # Assuming the polygon centroid represents the location for solar data
    centroid = polygon.centroid
    lat = centroid.y
    lon = centroid.x
    tilt_angle = 30  # Default tilt angle, modify as needed

    solar_data = import_pv_data(lat, lon, tilt_angle)
    solar_data.to_csv(solar_file)
    st.write(f"Solar data saved to {solar_file}")

'''
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
'''