## Local GISEle Project

Welcome to the Local GISEle Project! This project is designed to provide a comprehensive Geographic Information System (GIS) tool for analyzing and visualizing various spatial datasets. The application leverages a combination of open-source tools and APIs to offer functionalities for area selection, data retrieval, clustering, data visualization, and energy demand estimation. This README file will guide you through the project's structure, setup, and usage.

### Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
    - [Home](#home)
    - [Area Selection](#area-selection)
    - [Data Retrieve](#data-retrieve)
    - [Buildings](#buildings)
    - [Clustering](#clustering)
    - [Data Visualization and Enhancement](#data-visualization-and-enhancement)
    - [Costs](#costs)
    - [Summary Analysis](#summary-analysis)
    - [Demand Estimation](#demand-estimation)
    - [Mini-grid Sizing](#mini-grid-sizing)
    - [Grid](#grid)
    - [Results](#results)
5. [Contributing](#contributing)
6. [License](#license)

### Introduction

Local GISEle is a GIS-based web application developed using Streamlit, Folium, GeoPandas, and Earth Engine. It aims to provide users with a powerful tool to select areas, retrieve and visualize spatial data, perform clustering, and estimate energy demand for selected regions. The tool is particularly useful for energy analysts, urban planners, and researchers working in the field of spatial data analysis and energy planning.

### Project Structure

The project is organized into several directories and files to maintain modularity and ease of maintenance:

```
local_gisele/
│
├── data/
│   ├── input/
│   │   ├── buildings/
│   │   ├── roads/
│   │   ├── poi/
│   │   ├── national_grid/
│   │   ├── airports/
│   │   ├── ports/
│   │   ├── night_time_lights/
│   │   └── ...
│   │
│   ├── output/
│   │   ├── buildings/
│   │   ├── clustering/
│   │   ├── visualizations/
│   │   ├── demand_estimation/
│   │   ├── minigrid_sizing/
│   │   ├── grid_extension/
│   │   └── ...
│
├── scripts/
│   ├── main.py
│   ├── home.py
│   ├── area_selection.py
│   ├── data_retrieve.py
│   ├── buildings.py
│   ├── clustering.py
│   ├── data_visualization.py
│   ├── costs.py
│   ├── summary_analysis.py
│   ├── demand_estimation.py
│   ├── minigrid_sizing.py
│   ├── grid.py
│   ├── results.py
│   ├── utils.py
│   └── ...
│
├── results/
│   ├── combined_buildings.geojson
│   ├── osm_buildings.geojson
│   ├── google_buildings.geojson
│   ├── osm_roads.geojson
│   ├── osm_pois.geojson
│   └── ...
│
├── .streamlit/
│   └── secrets.toml
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Setup and Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/darlainedeme/GISEle.git
   cd GISEle
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the Streamlit secrets:**

   Create a `.streamlit` directory in the project root and add a `secrets.toml` file.

5. **Run the Streamlit application:**

   ```bash
   streamlit run scripts/main.py
   ```

### Usage

Once the application is running, you can navigate through different sections using the sidebar. Here's a brief overview of each section:

#### Home

This is the landing page of the application. It provides an introduction and navigation guidance to the user.

#### Area Selection

In this section, you can select an area of interest using three modes:
- By address: Enter an address to geocode and select the area.
- By coordinates: Enter latitude and longitude to specify the area.
- Upload file: Upload a GeoJSON file to define the area.

#### Data Retrieve

This page visualizes the progress of downloading data. All required data files are downloaded together and saved in a specific directory structure. Users can download all the results in a zip file from this page.

#### Buildings

In this section, building data is downloaded using OpenStreetMap (OSM) as the reference dataset and Google data for enhancement. Future enhancements might include Microsoft data.

#### Clustering

This section performs clustering within the target area to identify clusters for evaluation in subsequent phases. It combines already downloaded OSM files, clustering algorithms, and manual drawing by the user.

#### Data Visualization and Enhancement

This section allows users to visualize and manually enhance the previously downloaded data. It includes subpages for different data types:
- Out of the study area: Focuses on data related to the surrounding area, including major cities, main roads, airports, ports, national grid, substations, and nighttime lights.
- Within the study area: Focuses on data within the study area, including buildings, points of interest, access status, relative wealth index, roads, elevation, crops and biomass potential, water bodies and hydro potential, solar potential, wind potential, land cover, and available land for infrastructure.

#### Costs

This page allows users to input various cost parameters relevant to the analysis. It includes inputs for infrastructure costs, energy costs, and other relevant expenses.

#### Summary Analysis

This section provides a summary analysis combining maps and charts with all the information collected and processed in previous sections.

#### Demand Estimation

This section estimates the energy demand for the selected area using one of four methodologies based on user selection: MTF standard, MTF Polimi, RAMP, or MIT.

#### Mini-grid Sizing

This section optimizes the mini-grid based on one of two methodologies selected by the user: Michele or MicrogridsPy.

#### Grid

This part estimates the grid extension needed to connect the national grid from the closest substation to the study area and also estimates the internal distribution grid within the study area.

#### Results

This section displays the final results, including charts and maps, summarizing all the analyses and visualizations performed in the application.

### Contributing

We welcome contributions to the Local GISEle Project! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request on GitHub. Please ensure that your contributions are well-documented and tested.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Thank you for using Local GISEle! We hope this tool helps you in your spatial data analysis and energy planning projects. If you have any questions or need further assistance, please feel free to reach out through our contact information provided in the application.