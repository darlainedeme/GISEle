import os

# Define the project structure
project_structure = {
    'local_gisele': {
        'data': {
            'input': {
                'buildings': {},
                'roads': {},
                'poi': {},
                'national_grid': {},
                'airports': {},
                'ports': {},
                'night_time_lights': {},
            },
            'output': {
                'buildings': {},
                'clustering': {},
                'visualizations': {},
                'demand_estimation': {},
                'minigrid_sizing': {},
                'grid_extension': {},
            }
        },
        'scripts': {
            'main.py': '',
            'home.py': '',
            'area_selection.py': '',
            'data_retrieve.py': '',
            'buildings.py': '',
            'clustering.py': '',
            'data_visualization.py': '',
            'costs.py': '',
            'summary_analysis.py': '',
            'demand_estimation.py': '',
            'minigrid_sizing.py': '',
            'grid.py': '',
            'results.py': '',
            'utils.py': '',
        },
        'results': {
            'combined_buildings.geojson': '',
            'osm_buildings.geojson': '',
            'google_buildings.geojson': '',
            'osm_roads.geojson': '',
            'osm_pois.geojson': '',
        },
        '.streamlit': {
            'secrets.toml': '',
        },
        'requirements.txt': '',
        'README.md': '',
        '.gitignore': '',
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w') as f:
                f.write(content)

# Create the folder structure
base_path = os.getcwd()  # Change this if you want to create it in a different location
create_structure(base_path, project_structure)

print("Project structure created successfully.")
