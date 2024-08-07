import streamlit as st

# Set page configuration at the top level
st.set_page_config(page_title="Local GISEle", page_icon="🗺️", layout="wide")

from scripts import home, area_selection, data_retrieve, buildings, clustering_modes, data_visualization, costs, summary_analysis, demand_estimation, routing, results

# Display logo at the very top of the sidebar
st.sidebar.image("data/logo.png", width=200)  # Adjust the width as needed

# Define navigation
main_nav = st.sidebar.radio("Navigation", [
    "Home", "Area Selection", "Data Retrieve", "Buildings", 
    "Clustering", "Data Visualization and Enhancement", 
    "Summary Analysis", "Costs", "Demand Estimation", 
    "Routing", "Results"
])

# Display the selected page
if main_nav == "Home":
    home.show()
elif main_nav == "Area Selection":
    area_selection.show()
elif main_nav == "Data Retrieve":
    data_retrieve.show()
elif main_nav == "Buildings":
    buildings.show()
elif main_nav == "Clustering":
    clustering_modes.show()
elif main_nav == "Data Visualization and Enhancement":
    data_visualization.show()
elif main_nav == "Summary Analysis":
    summary_analysis.show()
elif main_nav == "Costs":
    costs.show()
elif main_nav == "Demand Estimation":
    demand_estimation.show()
elif main_nav == "Routing":
    routing.show()
elif main_nav == "Results":
    results.show()
