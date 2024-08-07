import streamlit as st

# Set page configuration at the top level
st.set_page_config(page_title="Local GISEle", page_icon="🗺️", layout="wide")

from scripts import home, area_selection, data_retrieve, buildings_mapping, clustering_modes, data_visualization, costs, summary_analysis, demand_estimation, routing, results

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=200)  # Adjust the width as needed

# Define navigation
main_nav = st.sidebar.radio("Navigation", [
    "Home", "Area Selection", "Data Retrieve", "Buildings", 
    "Clustering", "Data Visualization and Enhancement", 
    "Summary Analysis", "Costs", "Demand Estimation", 
    "Routing", "Results"
])

# Navigation dictionary for clean code
pages = {
    "Home": home.show,
    "Area Selection": area_selection.show,
    "Data Retrieve": data_retrieve.show,
    "Buildings": buildings.show,
    "Clustering": clustering_modes.show,
    "Data Visualization and Enhancement": data_visualization.show,
    "Summary Analysis": summary_analysis.show,
    "Costs": costs.show,
    "Demand Estimation": demand_estimation.show,
    "Routing": routing.show,
    "Results": results.show
}

# Display the selected page
if main_nav in pages:
    pages[main_nav]()
