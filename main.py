import os
import streamlit as st
from pathlib import Path
from scripts import home, area_selection, timelapse, data_retrieve, data_visualization, vania, modelling_parameters, demand_estimation, minigrid_sizing, routing, results
from scripts.gisele_scripts import clustering_modes, case_study_creation, optimization, geneticalgorithm_github

st.set_page_config(layout="wide")

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=100)  # Adjust the width as needed


# File Navigator Implementation
def file_navigator(current_dir):
    st.title("File Navigator")
    
    if current_dir.is_dir():
        # Display the current directory
        st.write(f"**Current Directory: {current_dir}**")

        # Parent Directory Link
        if current_dir != Path.cwd():
            if st.button("Go Up"):
                st.session_state.current_dir = current_dir.parent
                st.experimental_rerun()

        # List Files and Directories
        files = sorted(current_dir.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        for file in files:
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                if file.is_dir():
                    if st.button("📁", key=file.name):
                        st.session_state.current_dir = file
                        st.experimental_rerun()
                else:
                    st.text("📄")
            with col2:
                if file.is_dir():
                    st.write(f"**{file.name}/**")
                else:
                    st.write(file.name)
                    st.download_button("Download", open(file, 'rb'), file_name=file.name)

# Initialize session state
if "current_dir" not in st.session_state:
    st.session_state.current_dir = Path.cwd()

# Define the main sections with emojis
main_sections = {
    "🏠 Home": ["Home", "File Navigator"],  # Added File Navigator here
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
    subpage = st.sidebar.radio("Home", main_sections[main_section], index=0, key="home")
    pages = {
        "Home": home.show,
        "File Navigator": lambda: file_navigator(st.session_state.get("current_dir", Path.cwd()))  # Call the file navigator
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

# Display the selected page
if subpage in pages:
    pages[subpage]()
