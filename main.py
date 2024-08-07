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
    src = os.path.join("general_input", file)
    dst = os.path.join("data/predefined/general_input", file)
    if os.path.exists(src):
        shutil.move(src, dst)

# Move other predefined files (assuming they are in the root or other known locations)
# Add any other predefined files here

# Move routing data to data folder
routing_data_files = [
    # Add any specific routing data files here if needed
]

for file in routing_data_files:
    src = os.path.join("scripts/routing", file)
    dst = os.path.join("data/predefined", file)
    if os.path.exists(src):
        shutil.move(src, dst)

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
    if os.path.exists(os.path.join("scripts", src)):
        shutil.move(os.path.join("scripts", src), os.path.join(dst, src))

# Move routing scripts to their new location
routing_scripts_src = "scripts/routing_scripts"
routing_scripts_dst = "scripts/routing/routing_scripts"
if os.path.exists(routing_scripts_src):
    for item in os.listdir(routing_scripts_src):
        s = os.path.join(routing_scripts_src, item)
        d = os.path.join(routing_scripts_dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    shutil.rmtree(routing_scripts_src)

# Move utility scripts to utils
utility_files = [
    "data_import.py", "dijkstra.py"
    # Add any other utility scripts here
]

for file in utility_files:
    src = os.path.join("scripts", file)
    dst = os.path.join("utils", file)
    if os.path.exists(src):
        shutil.move(src, dst)

print("Project files have been rearranged according to the new structure.")
