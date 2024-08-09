import streamlit as st

# Import your other pages
from scripts import home, area_selection, timelapse, data_retrieve, data_visualization, vania, modelling_parameters, clustering_modes, demand_estimation, minigrid_sizing, routing, results

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=200)  # Adjust the width as needed

# Define the main sections
main_sections = {
    "Home": ["Home"],
    "Area Identification": ["Area Selection", "Satellite Timelapse"],
    "VANIA": ["Data Retrieve", "Data Visualization and Enhancement", "VANIA Report"],
    "GISELE": ["Modelling Parameters", "Clustering", "Demand Estimation", "Mini-grid Sizing", "Routing", "Results"]
}

# Select the main section
main_section = st.sidebar.radio("Navigation", list(main_sections.keys()))

# Subpage selection based on the main section
if main_section == "Home":
    pages = {
        "Home": home.show
    }
    subpage = "Home"

elif main_section == "Area Identification":
    subpage = st.sidebar.radio("Area Identification", main_sections[main_section])
    pages = {
        "Area Selection": area_selection.show,
        "Satellite Timelapse": timelapse.show
    }

elif main_section == "VANIA":
    subpage = st.sidebar.radio("VANIA", main_sections[main_section])
    pages = {
        "Data Retrieve": data_retrieve.show,
        "Data Visualization and Enhancement": data_visualization.show,
        "VANIA Report": vania.show
    }

elif main_section == "GISELE":
    subpage = st.sidebar.radio("GISELE", main_sections[main_section])
    pages = {
        "Modelling Parameters": modelling_parameters.show,
        "Clustering": clustering_modes.show,
        "Demand Estimation": demand_estimation.show,
        "Mini-grid Sizing": minigrid_sizing.show,
        "Routing": routing.show,
        "Results": results.show
    }

# Display the selected page
if subpage in pages:
    pages[subpage]()
