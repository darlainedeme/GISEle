import osmnx as ox
import streamlit as st
import geopandas as gpd

def download_osm_data(polygon, tags, file_path):
    try:
        data = ox.features_from_polygon(polygon, tags)
        if data.empty:
            if 'building' in tags:
                st.write("No buildings found in the selected area.")
            elif 'highway' in tags:
                st.write("No roads found in the selected area.")
            elif 'amenity' in tags:
                st.write("No points of interest found in the selected area.")
            elif 'natural' in tags and tags['natural'] == 'water':
                st.write("No water bodies found in the selected area.")
            elif 'place' in tags and tags['place'] == 'city':
                st.write("No major cities found within 200 km of the selected area.")
            elif 'aeroway' in tags and tags['aeroway'] == 'aerodrome':
                st.write("No airports found within 200 km of the selected area.")
            elif 'amenity' in tags and tags['amenity'] == 'port':
                st.write("No ports found within 200 km of the selected area.")
            elif 'power' in tags and tags['power'] == 'line':
                st.write("No power lines found within 200 km of the selected area.")
            elif 'power' in tags and tags['power'] in ['transformer', 'substation']:
                st.write("No transformers or substations found within 200 km of the selected area.")
            return None
        data.to_file(file_path, driver='GeoJSON')

        if 'building' in tags:
            st.write(f"{len(data)} buildings identified")
        elif 'highway' in tags:
            if data.crs.is_geographic:
                data = data.to_crs(epsg=3857)  # Reproject to a projected CRS for accurate length calculation
            total_km = data.geometry.length.sum() / 1000
            st.write(f"{total_km:.2f} km of roads identified")
        elif 'amenity' in tags:
            st.write(f"{len(data)} points of interest identified")
        elif 'natural' in tags and tags['natural'] == 'water':
            st.write(f"{len(data)} water bodies identified")
        elif 'place' in tags and tags['place'] == 'city':
            st.write(f"{len(data)} major cities identified")
        elif 'aeroway' in tags and tags['aeroway'] == 'aerodrome':
            st.write(f"{len(data)} airports identified")
        elif 'amenity' in tags and tags['amenity'] == 'port':
            st.write(f"{len(data)} ports identified")
        elif 'power' in tags and tags['power'] == 'line':
            st.write(f"{len(data)} power lines identified")
        elif 'power' in tags and tags['power'] in ['transformer', 'substation']:
            st.write(f"{len(data)} transformers or substations identified")
        return file_path
    except Exception as e:
        st.error(f"Error downloading OSM data: {e}")
        return None
