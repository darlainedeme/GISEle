import streamlit as st

def show():
    st.markdown("""
    # Welcome to Local GISEle 🗺️

    Local GISEle is a state-of-the-art Geographic Information System (GIS) tool tailored for in-depth local area analysis. It empowers users with the ability to gather, visualize, and analyze a wide range of geographical and infrastructural data, making it an indispensable resource for planners, developers, and researchers. By integrating data from various sources such as OpenStreetMap (OSM) and Google, Local GISEle provides comprehensive datasets that are vital for any spatial analysis project. Future expansions may even include data from Microsoft, further enhancing its capabilities.
    """)

    st.markdown("## Features")

    with st.expander("📍 **Area Identification**"):
        st.markdown("""
        **Area Selection**
        
        The foundation of any spatial analysis is the selection of the area of interest. Local GISEle offers you multiple ways to define this area: 
        - **Address Input**: Simply enter a physical address, and the system will pinpoint the location on the map.
        - **Coordinate Input**: If you have the exact latitude and longitude, you can input these coordinates to define the area.
        - **GeoJSON Upload**: For more complex areas, you can upload a GeoJSON file that delineates the boundaries of your study area. This flexibility ensures that you can start your analysis with the most precise area definition possible.
        
        **Satellite Timelapse**
        
        Once your area of interest is selected, you can create a satellite timelapse that visualizes changes over time. This feature is invaluable for understanding trends, such as urban expansion, deforestation, or seasonal changes. By observing how the area evolves, you can gain critical insights that inform your analysis and decision-making processes.
        """)

    with st.expander("📊 **VANIA**"):
        st.markdown("""
        **Data Retrieve**
        
        In this phase, Local GISEle allows you to pull in a wealth of data for your selected area. The datasets available for retrieval include:
        - **Buildings**: Information on the number, size, and distribution of buildings within the area.
        - **Roads**: Comprehensive data on road networks, including major highways, local roads, and even paths.
        - **Points of Interest (POI)**: Data on significant locations such as schools, hospitals, and businesses.
        - **Water Bodies**: Information about lakes, rivers, and other bodies of water.
        This process ensures that you have all the necessary data at your disposal to conduct a thorough analysis.

        **Data Visualization and Enhancement**
        
        After data retrieval, the next step is visualization. Local GISEle provides powerful tools to display the datasets on an interactive map. Not only can you view the data, but you can also manually enhance it by adding or modifying features. This is particularly useful for refining the accuracy of your analysis, as you can incorporate on-the-ground knowledge or additional data layers to better represent the reality on the ground.

        **VANIA Report**
        
        Once your data is visualized and enhanced, Local GISEle enables you to generate comprehensive reports. These reports bring together all the visualized data and enhancements, providing a detailed overview of the study area. They are essential for presenting findings to stakeholders, supporting decision-making processes, and documenting your analysis for future reference.
        """)

    with st.expander("⚙️ **GISELE**"):
        st.markdown("""
        **Modelling Parameters**
        
        Before diving into advanced analysis, it's crucial to set up the parameters that will guide your study. This includes defining cost inputs, assumptions about energy demand, and other factors that will influence the results. These parameters form the backbone of your analysis, ensuring that the results are tailored to your specific needs and conditions.

        **Clustering**
        
        Clustering is a powerful analytical tool that helps you identify patterns within your data. By grouping similar data points, you can uncover significant trends and relationships that might not be visible at first glance. Whether you're looking to optimize resource allocation or understand population distributions, clustering is an essential step in making data-driven decisions.

        **Demand Estimation**
        
        Estimating energy demand is a critical step in planning for infrastructure development. Local GISEle provides several methodologies for calculating energy needs within your area of interest. This analysis is vital for ensuring that energy supplies can meet future demands, supporting sustainable development, and minimizing environmental impacts.

        **Mini-grid Sizing**
        
        Based on the estimated demand, Local GISEle assists in designing and optimizing mini-grid systems. These systems are crucial for providing reliable electricity in remote or underserved areas. By tailoring the grid size to the specific needs of the area, you can ensure efficient energy distribution, reduce costs, and enhance the sustainability of the energy supply.

        **Routing**
        
        Efficient infrastructure planning requires careful route optimization. This feature helps you plan the best routes for roads, power lines, and pipelines by considering geographical features and existing infrastructure. By finding the optimal paths, you can minimize construction costs, reduce environmental impacts, and improve service delivery.

        **Results**
        
        After completing your analysis, the final step is to view and download the results. Local GISEle presents your findings in a comprehensive format, including maps, charts, and detailed reports. These outputs summarize all the steps you've taken, providing a clear and concise overview of your project. You can download these results for further use, whether it's for reporting, presentations, or future reference.
        """)

    st.markdown("## How to Use")

    with st.expander("📘 **User Guide**"):
        st.markdown("""
        - **Home**: Begin here to understand what Local GISEle offers and how to navigate the tool.
        - **Area Identification**: Start your project by selecting your area of interest and creating a satellite timelapse to understand temporal changes.
        - **VANIA**: Retrieve necessary data, visualize it on an interactive map, enhance it, and generate comprehensive reports.
        - **GISELE**: Move on to advanced analysis by setting up parameters, performing clustering, estimating energy demand, designing mini-grids, planning routes, and finally reviewing the results of your analysis.
        """)

    st.markdown("## Resources 🌐")
    st.markdown("""
    - **Web App**: [Local GISEle](https://gisele.streamlit.app/)
    - **GitHub Repository**: [Local GISEle GitHub](https://github.com/darlainedeme/GISEle)
    - **Documentation**: Find detailed documentation and step-by-step instructions in the GitHub repository.
    """)

    st.markdown("## Contact 📬")
    st.markdown("""
    For any questions, support, or collaboration inquiries, please reach out to:
    - **Darlain Edeme**: [E4G Polimi](http://www.e4g.polimi.it/)
    - **GitHub**: [Darlain Edeme](https://github.com/darlainedeme)
    - **Twitter**: [@darlainedeme](https://twitter.com/darlainedeme)
    - **LinkedIn**: [Darlain Edeme](https://www.linkedin.com/in/darlain-edeme)
    
    We're here to help you make the most out of Local GISEle for your GIS projects. Happy analyzing!
    """)

if __name__ == "__main__":
    show()
