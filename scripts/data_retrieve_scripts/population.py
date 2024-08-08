import requests
import json
import os
import geopandas as gpd
import pandas as pd
import streamlit as st

def download_worldpop_age_structure(geojson_path, year, output_csv):
    try:
        st.write("Starting download process...")
        
        # Read the GeoJSON file
        gdf = gpd.read_file(geojson_path)
        
        # Ensure the GeoDataFrame contains only one feature
        if len(gdf) > 1:
            st.write("GeoJSON contains more than one feature. Using the first feature.")
            gdf = gdf.iloc[[0]]
        
        # Convert the GeoDataFrame to a GeoJSON string
        geojson_str = gdf.to_json()
        
        st.write(f"GeoJSON constructed from file: {geojson_str}")

        # Define the WorldPop API endpoint for age structure
        api_url = f"https://api.worldpop.org/v1/services/stats"

        # Parameters for the API request
        params = {
            "dataset": "wpgpas",
            "year": year,
            "geojson": geojson_str,
            "runasync": "false"  # Synchronous execution
        }

        st.write(f"API URL: {api_url}")
        st.write(f"Parameters: {params}")

        # Send a GET request to the WorldPop API
        response = requests.get(api_url, params=params)
        st.write(f"API request sent. Status code: {response.status_code}")

        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()
            st.write(f"Response data received: {response_data}")
            taskid = response_data.get("taskid")
            st.write(f"Task ID: {taskid}")

            # Monitor the task status
            task_url = f"https://api.worldpop.org/v1/tasks/{taskid}"
            st.write(f"Task URL: {task_url}")
            task_response = requests.get(task_url)
            st.write(f"Task request sent. Status code: {task_response.status_code}")

            if task_response.status_code == 200:
                task_data = task_response.json()
                st.write(f"Task response data: {task_data}")
                
                if task_data.get("status") == "finished":
                    # The task is finished, get the results
                    data = task_data.get("data")
                    if data is not None:
                        agesexpyramid = data.get("agesexpyramid")
                        st.write(f"Age and sex pyramid data: {agesexpyramid}")

                        # Create a DataFrame from the data
                        df = pd.DataFrame(agesexpyramid)

                        # Save the DataFrame to a CSV file
                        df.to_csv(output_csv, index=False, columns=["age", "male", "female"])
                        st.write(f"File downloaded successfully and saved to {output_csv}")
                    else:
                        st.write("No data found in the task response.")
                else:
                    st.write("Task is not yet finished. Please check later.")
            else:
                st.write(f"Failed to monitor task: {task_response.status_code} - {task_response.text}")
        else:
            st.write(f"Failed to download file: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error downloading population data: {e}")

def download_population_data(polygon, year):
    population_file = os.path.join('data', '2_downloaded_input_data', 'population', 'age_structure_output.csv')
    os.makedirs(os.path.dirname(population_file), exist_ok=True)
    
    # Convert polygon to GeoDataFrame and save as GeoJSON
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    geojson_path = os.path.join('data', '2_downloaded_input_data', 'population', 'selected_area.geojson')
    gdf.to_file(geojson_path, driver="GeoJSON")
    
    # Validate the GeoJSON file
    try:
        with open(geojson_path, 'r') as file:
            geojson_data = json.load(file)
        st.write("GeoJSON file is valid.")
    except Exception as e:
        st.error(f"Invalid GeoJSON file: {e}")
        return
    
    # Call the function to download the WorldPop data
    download_worldpop_age_structure(geojson_path, year, population_file)
