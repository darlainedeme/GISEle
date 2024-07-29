import streamlit as st

def show():
    st.markdown("""
    # Welcome to Local GISEle
    
    Local GISEle is a comprehensive Geographic Information System (GIS) tool designed to facilitate local area analysis, data collection, visualization, and analysis for various geographical and infrastructural projects. This application integrates multiple data sources, including OpenStreetMap (OSM), Google, and potentially Microsoft in the future, to provide enriched datasets for your projects.

    ## Features
    - **Area Selection**: Choose an area by address, coordinates, or upload a GeoJSON file.
    - **Data Retrieval**: Download and visualize data such as buildings, roads, points of interest, and more.
    - **Buildings Analysis**: Enhance building datasets using multiple sources.
    - **Clustering**: Identify and evaluate clusters within the target area.
    - **Data Visualization and Enhancement**: Visualize and manually enhance various datasets.
    - **Cost Inputs**: Define various cost inputs for your analysis.
    - **Summary Analysis**: Combine all information into a comprehensive summary with maps and charts.
    - **Demand Estimation**: Estimate energy demand using different methodologies.
    - **Mini-grid Sizing**: Optimize mini-grid designs.
    - **Grid Extension**: Estimate the extension of the national grid to the study area.
    - **Results**: View and download comprehensive results.

    ## How to Use
    1. **Home**: This is the welcome page where you can get an overview of the application.
    2. **Area Selection**: Navigate to the 'Area Selection' page to select your area of interest by address, coordinates, or by uploading a GeoJSON file.
    3. **Data Retrieve**: Use this page to download all necessary datasets for your selected area. Visualize the progress and download the results as a zip file.
    4. **Buildings**: Analyze and enhance building data using OSM as the reference dataset and Google data for enhancements.
    5. **Clustering**: Perform clustering analysis to identify clusters within the target area.
    6. **Data Visualization and Enhancement**: Visualize and manually enhance the downloaded datasets.
    7. **Costs**: Define various cost inputs required for your analysis.
    8. **Summary Analysis**: View a comprehensive summary of all data with maps and charts.
    9. **Demand Estimation**: Estimate the energy demand using different methodologies.
    10. **Mini-grid Sizing**: Optimize the mini-grid design based on user inputs.
    11. **Grid**: Estimate the grid extension needed to connect the study area to the national grid.
    12. **Results**: View and download the final results of your analysis.

    ## Resources
    - **Web App URL**: [https://gisele.streamlit.app/](https://gisele.streamlit.app/)
    - **GitHub Repository**: [https://github.com/darlainedeme/GISEle](https://github.com/darlainedeme/GISEle)
    - **Documentation**: Detailed documentation and usage instructions can be found in the GitHub repository.
    
    ## Contact
    For any questions or support, please contact:
    - **Darlain Edeme**: [http://www.e4g.polimi.it/](http://www.e4g.polimi.it/)
    - **GitHub**: [https://github.com/darlainedeme](https://github.com/darlainedeme)
    - **Twitter**: [https://twitter.com/darlainedeme](https://twitter.com/darlainedeme)
    - **LinkedIn**: [https://www.linkedin.com/in/darlain-edeme](https://www.linkedin.com/in/darlain-edeme)
    
    We hope you find Local GISEle helpful for your GIS projects. Happy analyzing!
    """)

# Display the home page
if __name__ == "__main__":
    show()
