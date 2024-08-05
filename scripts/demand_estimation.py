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
        "number_of_users": 25,
        "appliances": [
            {"name": "Indoor Bulb", "number": 6, "power": 7, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Television", "number": 2, "power": 60, "num_windows": 3, "func_time": 180, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Freezer", "number": 1, "power": 200, "num_windows": 1, "func_time": 1440, "time_fraction_random_variability": 0.0, "func_cycle": 30, "fixed": "yes", "fixed_cycle": 3, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2}
        ]
    },
    "Middle-Income Household": {
        "number_of_users": 75,
        "appliances": [
            {"name": "Indoor Bulb", "number": 3, "power": 7, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Television", "number": 1, "power": 60, "num_windows": 3, "func_time": 90, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Mixer", "number": 1, "power": 50, "num_windows": 2, "func_time": 30, "time_fraction_random_variability": 0.1, "func_cycle": 1, "fixed": "no", "fixed_cycle": 0, "occasional_use": 0.33, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2}
        ]
    },
    "Low-Income Household": {
        "number_of_users": 50,
        "appliances": [
            {"name": "Indoor Bulb", "number": 2, "power": 7, "num_windows": 2, "func_time": 120, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Television", "number": 1, "power": 60, "num_windows": 3, "func_time": 90, "time_fraction_random_variability": 0.1, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Phone Charger", "number": 2, "power": 2, "num_windows": 1, "func_time": 300, "time_fraction_random_variability": 0.2, "func_cycle": 5, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2}
        ]
    },
    "Public Lighting": {
        "number_of_users": 1,
        "appliances": [
            {"name": "Public Lighting Type 1", "number": 12, "power": 40, "num_windows": 2, "func_time": 310, "time_fraction_random_variability": 0.1, "func_cycle": 300, "fixed": "yes", "fixed_cycle": 0, "occasional_use": 1, "flat": "yes", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Public Lighting Type 2", "number": 25, "power": 150, "num_windows": 2, "func_time": 310, "time_fraction_random_variability": 0.1, "func_cycle": 300, "fixed": "yes", "fixed_cycle": 0, "occasional_use": 1, "flat": "yes", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2}
        ]
    },
    "Rural School": {
        "number_of_users": 1,
        "appliances": [
            {"name": "Indoor Bulb", "number": 8, "power": 7, "num_windows": 1, "func_time": 60, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Laptop", "number": 18, "power": 50, "num_windows": 2, "func_time": 210, "time_fraction_random_variability": 0.1, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Freezer", "number": 1, "power": 200, "num_windows": 1, "func_time": 1440, "time_fraction_random_variability": 0.0, "func_cycle": 30, "fixed": "yes", "fixed_cycle": 3, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2}
        ]
    },
    "Rural Hospital": {
        "number_of_users": 1,
        "appliances": [
            {"name": "Indoor Bulb", "number": 12, "power": 7, "num_windows": 2, "func_time": 690, "time_fraction_random_variability": 0.2, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Fridge Type 1", "number": 1, "power": 150, "num_windows": 1, "func_time": 1440, "time_fraction_random_variability": 0.0, "func_cycle": 30, "fixed": "yes", "fixed_cycle": 3, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2},
            {"name": "Laptop", "number": 2, "power": 50, "num_windows": 2, "func_time": 300, "time_fraction_random_variability": 0.1, "func_cycle": 10, "fixed": "no", "fixed_cycle": 0, "occasional_use": 1, "flat": "no", "thermal_P_var": 0, "pref_index": 0, "wd_we_type": 2}
        ]
    }
}

# Function to display and edit user categories and appliances
def display_user_category(category_name, category_data):
    st.subheader(category_name)
    number_of_users = st.number_input(f"Number of Users for {category_name}", min_value=1, value=category_data["number_of_users"])
    appliances = []
    for appliance in category_data["appliances"]:
        st.markdown(f"**{appliance['name']}**")

        # Debugging prints
        st.write(f"Debug: {appliance}")

        appliance["number"] = st.number_input(f"Number of {appliance['name']} ({category_name})", min_value=1, value=appliance["number"])
        appliance["power"] = st.number_input(f"Power of {appliance['name']} ({category_name}) (W)", min_value=1, value=appliance["power"])
        appliance["num_windows"] = st.number_input(f"Number of Windows for {appliance['name']} ({category_name})", min_value=1, value=appliance["num_windows"])
        appliance["func_time"] = st.number_input(f"Functioning Time of {appliance['name']} ({category_name}) (minutes)", min_value=1, value=appliance["func_time"])
        appliance["time_fraction_random_variability"] = st.number_input(f"Time Fraction Random Variability for {appliance['name']} ({category_name})", min_value=0.0, max_value=1.0, value=float(appliance["time_fraction_random_variability"]))
        appliance["func_cycle"] = st.number_input(f"Functioning Cycle of {appliance['name']} ({category_name}) (minutes)", min_value=1, value=appliance["func_cycle"])
        appliance["fixed"] = st.selectbox(f"Fixed Operation for {appliance['name']} ({category_name})", options=["yes", "no"], index=["yes", "no"].index(appliance["fixed"]))
        appliance["fixed_cycle"] = st.number_input(f"Fixed Cycle for {appliance['name']} ({category_name})", min_value=0, value=appliance["fixed_cycle"])
        appliance["occasional_use"] = st.number_input(f"Occasional Use for {appliance['name']} ({category_name})", min_value=0.0, max_value=1.0, value=float(appliance["occasional_use"]))
        appliance["flat"] = st.selectbox(f"Flat Usage for {appliance['name']} ({category_name})", options=["yes", "no"], index=["yes", "no"].index(appliance["flat"]))
        appliance["thermal_P_var"] = st.number_input(f"Thermal Power Variability for {appliance['name']} ({category_name})", min_value=0.0, max_value=1.0, value=float(appliance["thermal_P_var"]))
        appliance["pref_index"] = st.number_input(f"Preference Index for {appliance['name']} ({category_name})", min_value=0, value=appliance["pref_index"])
        appliance["wd_we_type"] = st.number_input(f"Weekday/Weekend Type for {appliance['name']} ({category_name})", min_value=0, max_value=2, value=appliance["wd_we_type"])
        appliances.append(appliance)
    return {
        "number_of_users": number_of_users,
        "appliances": appliances
    }

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
