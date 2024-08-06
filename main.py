import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
st.set_page_config(layout="wide")

from scripts import home, area_selection, data_retrieve, buildings, clustering, data_visualization, costs, summary_analysis

# Define navigation
main_nav = st.sidebar.radio("Navigation", ["Home", "Area Selection", "Data Retrieve", "Buildings", "Clustering", "Data Visualization", "Costs", "Summary Analysis"])

# Navigation logic
if main_nav == "Home":
    home.show()
elif main_nav == "Area Selection":
    area_selection.show()
elif main_nav == "Data Retrieve":
    data_retrieve.show()
elif main_nav == "Buildings":
    buildings.show()
elif main_nav == "Clustering":
    clustering.show()
elif main_nav == "Data Visualization":
    data_visualization.show()
elif main_nav == "Costs":
    costs.show()
elif main_nav == "Summary Analysis":
    summary_analysis.show()
