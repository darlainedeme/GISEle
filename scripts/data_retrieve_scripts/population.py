import requests
import json
import os
import geopandas as gpd
import pandas as pd
import streamlit as st

def download_worldpop_age_structure(geojson_str, year, output_csv):
    try:
        st.write("Starting download process...")

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
    
    # Convert polygon to GeoDataFrame and save as GeoJSON string
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    geojson_str = gdf.to_json()
    
    # Validate the GeoJSON string
    try:
        geojson_data = json.loads(geojson_str)
        st.write("GeoJSON string is valid.")
    except Exception as e:
        st.error(f"Invalid GeoJSON string: {e}")
        return
    
    # Call the function to download the WorldPop data
    download_worldpop_age_structure(geojson_str, year, population_file)

# Example usage within Streamlit
if __name__ == "__main__":
    st.title("Download Population Data")

    # Assume `polygon` is provided as input; this is just an example
    example_polygon = gpd.GeoSeries([box(12.35, 41.8, 12.65, 42.0)], crs="EPSG:4326")

    if st.button("Download Population Data"):
        download_population_data(example_polygon.unary_union, 2020)
