import requests
import json
import os
import geopandas as gpd
import pandas as pd
import streamlit as st

def download_population_data(geojson_path, year, output_csv, output_raster):
    print("Starting download process...")
    # Read the GeoJSON file
    gdf = gpd.read_file(geojson_path)
    
    # Ensure the GeoDataFrame contains only one feature
    if len(gdf) > 1:
        print("GeoJSON contains more than one feature. Using the first feature.")
        gdf = gdf.iloc[[0]]
    
    # Convert the GeoDataFrame to a GeoJSON string
    geojson_str = gdf.to_json()
    
    print(f"GeoJSON constructed from file: {geojson_str}")

    # Define the WorldPop API endpoint for age structure
    api_url = f"https://api.worldpop.org/v1/services/stats"

    # Parameters for the API request
    params = {
        "dataset": "wpgpas",
        "year": year,
        "geojson": geojson_str,
        "runasync": "false"  # Synchronous execution
    }

    print(f"API URL: {api_url}")
    print(f"Parameters: {params}")

    # Send a GET request to the WorldPop API
    response = requests.get(api_url, params=params)
    print(f"API request sent. Status code: {response.status_code}")

    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        print(f"Response data received: {response_data}")
        taskid = response_data.get("taskid")
        print(f"Task ID: {taskid}")

        # Monitor the task status
        task_url = f"https://api.worldpop.org/v1/tasks/{taskid}"
        print(f"Task URL: {task_url}")
        task_response = requests.get(task_url)
        print(f"Task request sent. Status code: {task_response.status_code}")

        if task_response.status_code == 200:
            task_data = task_response.json()
            print(f"Task response data: {task_data}")
            
            if task_data.get("status") == "finished":
                # The task is finished, get the results
                data = task_data.get("data")
                if data is not None:
                    agesexpyramid = data.get("agesexpyramid")
                    print(f"Age and sex pyramid data: {agesexpyramid}")

                    # Create a DataFrame from the data
                    df = pd.DataFrame(agesexpyramid)

                    # Save the DataFrame to a CSV file
                    df.to_csv(output_csv, index=False, columns=["age", "male", "female"])
                    print(f"File downloaded successfully and saved to {output_csv}")
                else:
                    print("No data found in the task response.")
            else:
                print("Task is not yet finished. Please check later.")
        else:
            print(f"Failed to monitor task: {task_response.status_code} - {task_response.text}")
    else:
        print(f"Failed to download file: {response.status_code} - {response.text}")

    # Now download the population raster data
    raster_api_url = "https://www.worldpop.org/rest/data/pop/wpgp"
    raster_params = {
        "geojson": geojson_str,
        "runasync": "false",  # Synchronous execution
        "year": year,
        "project": "cic2020_UNadj_100m"  # Constrained Individual countries 2020 UN adjusted (100m resolution)
    }

    print(f"Raster API URL: {raster_api_url}")
    print(f"Raster Parameters: {raster_params}")

    raster_response = requests.get(raster_api_url, params=raster_params)
    print(f"Raster API request sent. Status code: {raster_response.status_code}")

    if raster_response.status_code == 200:
        # Parse the JSON response
        raster_response_data = raster_response.json()
        print(f"Raster response data received: {raster_response_data}")
        raster_taskid = raster_response_data.get("taskid")
        print(f"Raster Task ID: {raster_taskid}")

        # Monitor the raster task status
        raster_task_url = f"https://api.worldpop.org/v1/tasks/{raster_taskid}"
        print(f"Raster Task URL: {raster_task_url}")
        raster_task_response = requests.get(raster_task_url)
        print(f"Raster Task request sent. Status code: {raster_task_response.status_code}")

        if raster_task_response.status_code == 200:
            raster_task_data = raster_task_response.json()
            print(f"Raster Task response data: {raster_task_data}")

            if raster_task_data.get("status") == "finished":
                # The raster task is finished, get the results
                raster_data = raster_task_data.get("data")
                if raster_data is not None:
                    raster_download_url = raster_data.get("downloadUrl")
                    print(f"Raster Download URL: {raster_download_url}")

                    # Download the raster file
                    raster_download_response = requests.get(raster_download_url)
                    with open(output_raster, 'wb') as f:
                        f.write(raster_download_response.content)
                    print(f"Raster file downloaded successfully and saved to {output_raster}")
                else:
                    print("No raster data found in the task response.")
            else:
                print("Raster task is not yet finished. Please check later.")
        else:
            print(f"Failed to monitor raster task: {raster_task_response.status_code} - {raster_task_response.text}")
    else:
        print(f"Failed to download raster file: {raster_response.status_code} - {raster_response.text}")

