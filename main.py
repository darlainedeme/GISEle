import os
import streamlit as st
from scripts import home, area_selection, timelapse, data_retrieve, data_visualization, vania, modelling_parameters, demand_estimation, minigrid_sizing, routing, results
from scripts.gisele_scripts import clustering_modes, case_study_creation, optimization, geneticalgorithm_github

st.set_page_config(layout="wide")

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=100)  # Adjust the width as needed

# Define the main sections with emojis
main_sections = {
    "🏠 Home": ["Home"],
    "📍 Area Identification": ["Area Selection", "Satellite Timelapse"],
    "📊 VANIA": ["Data Retrieve", "Data Visualization and Enhancement", "VANIA Report"],
    "⚙️ GISELE": [
        "Modelling Parameters", 
        "Clustering", 
        "Case Study", 
        "Optimization", 
        "Demand Estimation", 
        "Mini-grid Sizing", 
        "Routing", 
        "Results"
    ]
}

# Select the main section
main_section = st.sidebar.radio("Navigation", list(main_sections.keys()))

# Visual separation and styling for subpages
if main_section == "🏠 Home":
    subpage = "Home"
    pages = {
        "Home": home.show
    }

elif main_section == "📍 Area Identification":
    st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)
    subpage = st.sidebar.radio("Area Identification", main_sections[main_section], index=0, key="area_id")
    pages = {
        "Area Selection": area_selection.show,
        "Satellite Timelapse": timelapse.show
    }

elif main_section == "📊 VANIA":
    st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)
    subpage = st.sidebar.radio("VANIA", main_sections[main_section], index=0, key="vania")
    pages = {
        "Data Retrieve": data_retrieve.show,
        "Data Visualization and Enhancement": data_visualization.show,
        "VANIA Report": vania.show
    }

elif main_section == "⚙️ GISELE":
    st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)
    subpage = st.sidebar.radio("GISELE", main_sections[main_section], index=0, key="gisele")
    pages = {
        "Modelling Parameters": modelling_parameters.show,
        "Clustering": clustering_modes.show,
        "Case Study": case_study_creation.show,
        "Optimization": optimization.show,
        "Demand Estimation": demand_estimation.show,
        "Mini-grid Sizing": minigrid_sizing.show,
        "Routing": routing.show,
        "Results": results.show
    }

# File Explorer Section
st.sidebar.markdown("<hr style='border: none; border-bottom: 2px solid #ccc;'>", unsafe_allow_html=True)

# Function to list and download files
def list_files(directory):
    files = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path):
            files.append((file, path))
        else:
            files.append((file + "/", path))
    return files

def file_explorer(directory):
    st.sidebar.write("**File Explorer**")
    files = list_files(directory)
    for filename, filepath in files:
        if os.path.isdir(filepath):
            if st.sidebar.button(f"Open {filename}"):
                st.session_state.current_dir = filepath
        else:
            st.sidebar.write(f"**{filename}**")
            st.sidebar.download_button("Download", open(filepath, 'rb'), file_name=filename)

# Initial directory to start from
if "current_dir" not in st.session_state:
    st.session_state.current_dir = os.getcwd()

file_explorer(st.session_state.current_dir)

# Display the selected page
if subpage in pages:
    pages[subpage]()
