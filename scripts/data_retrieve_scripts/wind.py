import requests
import os
import streamlit as st
import requests
import json
import pandas as pd
import geopandas as gpd
import streamlit as st
import os
import time

def import_wind_data(lat, lon, wt):
    token = 'c511d32b578b4ec19c3d43c1a3fffb4cad5dc4d2'
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    s.headers = {'Authorization': 'Token ' + token}
    url = api_base + 'data/wind'

    args = {
        'lat': lat,
        'lon': lon,
        'date_from': '2023-01-01',
        'date_to': '2023-12-31',
        'capacity': 1.0,
        'height': 50,
        'turbine': str(wt),
        'format': 'json',
    }

    while True:
        try:
            r = s.get(url, params=args)
            parsed_response = json.loads(r.text)
            break
        except:
            st.write('Problem with importing Wind data. Software is in sleep mode for 30 seconds.')
            time.sleep(30)
    
    data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
    st.write("Wind Data imported")

    return data

def download_wind_data(polygon):
    wind_file = os.path.join('data', '2_downloaded_input_data', 'wind', 'wind_data.csv')
    os.makedirs(os.path.dirname(wind_file), exist_ok=True)

    # Assuming the polygon centroid represents the location for wind data
    centroid = polygon.centroid
    lat = centroid.y
    lon = centroid.x
    turbine_type = 'Vestas V112-3.0MW'  # Default turbine type, modify as needed

    wind_data = import_wind_data(lat, lon, turbine_type)
    wind_data.to_csv(wind_file)
    st.write(f"Wind data saved to {wind_file}")

'''
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
'''