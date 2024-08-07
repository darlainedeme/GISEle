import os
import shutil

# Define the new structure
structure = {
    "project_root/": [
        "docs/", "scripts/", "main.py", "README.md", "requirements.txt", "utils/"
    ],
    "data/": [
        "predefined/", "downloaded/", "user_created/", "intermediate/", "output/"
    ],
    "data/predefined/": [
        "general_input/"
    ],
    "scripts/": [
        "clustering/", "visualization/", "summary/", "routing/"
    ],
    "scripts/clustering/": ["__init__.py"],
    "scripts/visualization/": ["__init__.py"],
    "scripts/summary/": ["__init__.py"],
    "scripts/routing/": ["__init__.py", "routing_scripts/"],
    "utils/": ["__init__.py"]
}

# Create the new directories
for base, dirs in structure.items():
    for dir in dirs:
        os.makedirs(os.path.join(base, dir), exist_ok=True)

# Move predefined files
predefined_files = [
    "Configuration.csv", "config_param.csv", "COUNTRIES.csv",
    "Landcover.csv", "Load Profile.csv", "ss_data_evn.csv", "TiltAngles.csv"
]

for file in predefined_files:
    shutil.move(os.path.join("general_input", file), os.path.join("data/predefined/general_input", file))

# Move other predefined files (assuming they are in the root or other known locations)
# Add any other predefined files here

# Move routing data to data folder
routing_data_files = [
    # List all routing data files here
]

for file in routing_data_files:
    shutil.move(os.path.join("scripts/routing", file), os.path.join("data/predefined", file))

# Move scripts to their new locations
script_files = {
    "home.py": "scripts/",
    "area_selection.py": "scripts/",
    "data_retrieve.py": "scripts/",
    "buildings.py": "scripts/",
    "clustering_modes.py": "scripts/clustering/",
    "data_visualization.py": "scripts/visualization/",
    "summary_analysis.py": "scripts/summary/",
    "costs.py": "scripts/",
    "demand_estimation.py": "scripts/",
    "routing.py": "scripts/routing/",
    "results.py": "scripts/"
}

for src, dst in script_files.items():
    shutil.move(os.path.join("scripts", src), os.path.join(dst, src))

# Move routing scripts to their new location
if os.path.exists("scripts/routing_scripts"):
    shutil.move("scripts/routing_scripts", "scripts/routing/")

# Move utility scripts to utils
utility_files = [
    "data_import.py", "dijkstra.py"
    # Add any other utility scripts here
]

for file in utility_files:
    shutil.move(os.path.join("scripts", file), os.path.join("utils", file))

print("Project files have been rearranged according to the new structure.")
