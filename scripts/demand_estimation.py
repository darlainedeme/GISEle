import streamlit as st
from ramp import UseCase, User
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

# Define initial values for user categories and appliances
initial_values = {
    "High-Income Household": {
        "num_users": 25,
        "appliances": [
            {"name": "Indoor Bulb", "number": 6, "power": 7, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(1170, 1440), (0, 30)]},
            {"name": "Television", "number": 2, "power": 60, "num_windows": 3, "func_time": 180, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(720, 900), (1170, 1440), (0, 60)]},
            {"name": "Refrigerator", "number": 1, "power": 150, "num_windows": 1, "func_time": 1440, "time_fraction_random_variability": 0.0, "func_cycle": 30, "fixed": "yes", "fixed_cycle": 3, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(0, 1440)]}
        ]
    },
    "Middle-Income Household": {
        "num_users": 75,
        "appliances": [
            {"name": "Indoor Bulb", "number": 3, "power": 7, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(1170, 1440), (0, 30)]},
            {"name": "Television", "number": 1, "power": 60, "num_windows": 3, "func_time": 90, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(450, 660), (720, 840), (1170, 1440)]},
            {"name": "Phone Charger", "number": 4, "power": 2, "num_windows": 1, "func_time": 300, "time_fraction_random_variability": 0.2, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(1020, 1440)]}
        ]
    },
    "Low-Income Household": {
        "num_users": 50,
        "appliances": [
            {"name": "Indoor Bulb", "number": 2, "power": 7, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(1170, 1440), (0, 30)]},
            {"name": "Outdoor Bulb", "number": 1, "power": 13, "num_windows": 2, "func_time": 600, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(0, 330), (1170, 1440)]},
            {"name": "Phone Charger", "number": 2, "power": 2, "num_windows": 1, "func_time": 300, "time_fraction_random_variability": 0.2, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(1080, 1440)]}
        ]
    },
    "Public Lighting": {
        "num_users": 1,
        "appliances": [
            {"name": "Street Light", "number": 20, "power": 40, "num_windows": 2, "func_time": 310, "time_fraction_random_variability": 0.1, "func_cycle": 300, "fixed": "yes", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "yes", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(0, 336), (1110, 1440)]},
            {"name": "Community Light", "number": 10, "power": 150, "num_windows": 2, "func_time": 310, "time_fraction_random_variability": 0.1, "func_cycle": 300, "fixed": "yes", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "yes", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(0, 336), (1110, 1440)]}
        ]
    },
    "Rural School": {
        "num_users": 1,
        "appliances": [
            {"name": "Indoor Bulb", "number": 8, "power": 7, "num_windows": 1, "func_time": 60, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(1020, 1080)]},
            {"name": "Laptop", "number": 18, "power": 50, "num_windows": 2, "func_time": 210, "time_fraction_random_variability": 0.1, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(510, 750), (810, 1080)]},
            {"name": "Printer", "number": 1, "power": 20, "num_windows": 2, "func_time": 30, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(510, 750), (810, 1080)]}
        ]
    },
    "Rural Hospital": {
        "num_users": 1,
        "appliances": [
            {"name": "Indoor Bulb", "number": 12, "power": 7, "num_windows": 2, "func_time": 690, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(480, 720), (870, 1440)]},
            {"name": "Fridge", "number": 2, "power": 150, "num_windows": 1, "func_time": 1440, "time_fraction_random_variability": 0.0, "func_cycle": 30, "fixed": "yes", "fixed_cycle": 3, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(0, 1440)]},
            {"name": "Laptop", "number": 2, "power": 50, "num_windows": 2, "func_time": 300, "time_fraction_random_variability": 0.1, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(480, 720), (1050, 1440)]}
        ]
    },
    "Commercial Shops": {
        "num_users": 10,
        "appliances": [
            {"name": "Lighting", "number": 10, "power": 20, "num_windows": 2, "func_time": 300, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(600, 900), (1000, 1300)]},
            {"name": "Refrigerator", "number": 1, "power": 150, "num_windows": 1, "func_time": 1440, "time_fraction_random_variability": 0.0, "func_cycle": 30, "fixed": "yes", "fixed_cycle": 3, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(0, 1440)]},
            {"name": "Cash Register", "number": 1, "power": 50, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.1, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(600, 900), (1000, 1300)]}
        ]
    },
    "Agricultural Processing": {
        "num_users": 5,
        "appliances": [
            {"name": "Grinder", "number": 2, "power": 500, "num_windows": 1, "func_time": 240, "time_fraction_random_variability": 0.1, "func_cycle": 20, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(600, 840)]},
            {"name": "Dryer", "number": 1, "power": 1000, "num_windows": 1, "func_time": 480, "time_fraction_random_variability": 0.1, "func_cycle": 30, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(600, 1080)]},
            {"name": "Lighting", "number": 10, "power": 20, "num_windows": 2, "func_time": 300, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1.0, "flat": "no", "thermal_P_var": 0.0, "pref_index": 0, "wd_we_type": 2, "windows": [(600, 900), (1000, 1300)]}
        ]
    }
}
# Function to display and edit user categories and appliances
def display_user_category(category_name, category_data):
    st.subheader(category_name)
    num_users = st.number_input(f"Number of {category_name}", min_value=1, value=category_data["num_users"])
    appliances = category_data["appliances"]

    for appliance in appliances:
        with st.expander(f"{appliance['name']}"):
            appliance["number"] = st.number_input(f"Number of {appliance['name']}", min_value=1, value=appliance["number"])
            appliance["power"] = st.number_input(f"Power of {appliance['name']} (W)", min_value=1, value=appliance["power"])
            appliance["num_windows"] = st.number_input(f"Number of Windows for {appliance['name']}", min_value=1, value=appliance["num_windows"])
            appliance["func_time"] = st.number_input(f"Functioning Time for {appliance['name']} (minutes)", min_value=1, value=appliance["func_time"])
            appliance["time_fraction_random_variability"] = st.number_input(f"Time Fraction Random Variability for {appliance['name']}", min_value=0.0, max_value=1.0, value=appliance["time_fraction_random_variability"])
            appliance["func_cycle"] = st.number_input(f"Functioning Cycle for {appliance['name']} (minutes)", min_value=1, value=appliance["func_cycle"])
            appliance["fixed"] = st.selectbox(f"Fixed for {appliance['name']}", options=["yes", "no"], index=0 if appliance["fixed"] == "no" else 1)
            appliance["fixed_cycle"] = st.number_input(f"Fixed Cycle for {appliance['name']}", min_value=0, value=appliance["fixed_cycle"])
            appliance["occasional_use"] = st.number_input(f"Occasional Use for {appliance['name']}", min_value=0.0, max_value=1.0, value=float(appliance["occasional_use"]))
            appliance["flat"] = st.selectbox(f"Flat for {appliance['name']}", options=["yes", "no"], index=0 if appliance["flat"] == "no" else 1)
            appliance["thermal_P_var"] = st.number_input(f"Thermal Power Variability for {appliance['name']}", min_value=0.0, max_value=1.0, value=float(appliance["thermal_P_var"]))
            appliance["pref_index"] = st.number_input(f"Preference Index for {appliance['name']}", min_value=0, value=appliance["pref_index"])
            appliance["wd_we_type"] = st.selectbox(f"Weekday/Weekend Type for {appliance['name']}", options=[0, 1, 2], index=appliance["wd_we_type"])

            for i in range(appliance["num_windows"]):
                start, end = appliance["windows"][i]
                appliance["windows"][i] = st.slider(f"Window {i+1} for {appliance['name']} (minutes from midnight)", 0, 1440, (start, end))
    return {"num_users": num_users, "appliances": appliances}

# Streamlit interface
def show():
    st.title("Demand Estimation with RAMP")
    st.write("Estimate the load profile for different user categories.")

    user_data = {}
    for category_name, category_data in initial_values.items():
        user_data[category_name] = display_user_category(category_name, category_data)

    if st.button("Estimate Demand"):
        st.write("Generating load profiles...")
        progress = st.progress(0)

        users = []
        for category_name, category_data in user_data.items():
            user = User(user_name=category_name, num_users=category_data["num_users"])
            for appliance in category_data["appliances"]:
                app = user.Appliance(
                    number=appliance["number"], power=appliance["power"], num_windows=appliance["num_windows"], 
                    func_time=appliance["func_time"], time_fraction_random_variability=appliance["time_fraction_random_variability"], 
                    func_cycle=appliance["func_cycle"], fixed=appliance["fixed"], fixed_cycle=appliance["fixed_cycle"], 
                    occasional_use=appliance["occasional_use"], flat=appliance["flat"], thermal_P_var=appliance["thermal_P_var"], 
                    pref_index=appliance["pref_index"], wd_we_type=appliance["wd_we_type"]
                )

                # Ensure total window time is greater than or equal to func_time
                total_window_time = sum([end - start for start, end in appliance["windows"]])
                if total_window_time < appliance["func_time"]:
                    st.error(f"The sum of all windows time intervals for the appliance '{appliance['name']}' of user '{category_name}' is smaller than the time the appliance is supposed to be on ({total_window_time} < {appliance['func_time']}). Please adjust the time windows.")
                    return
                
                for i, window in enumerate(appliance["windows"], start=1):
                    start, end = window
                    setattr(app, f"window_{i}", (start, end))
                users.append(user)
        
        today = datetime.today().strftime('%Y-%m-%d')
        use_case = UseCase(users=users, date_start=today, date_end=today)
        load_profile = use_case.generate_daily_load_profiles()

        progress.progress(100)
        st.write("Load profile generation complete.")

        # Plot results
        st.write("Plotting results...")
        plt.figure(figsize=(10, 5))
        plt.plot(load_profile, label='Load Profile')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Load (W)')
        plt.title('Generated Load Profile')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Export to CSV
        csv = pd.DataFrame(load_profile).to_csv(index=False)
        st.download_button(label="Download Load Profile as CSV", data=csv, file_name="load_profile.csv", mime='text/csv')

if __name__ == "__main__":
    show()
