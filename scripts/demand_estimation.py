import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from ramp import UseCase, User
from ramp.post_process.post_process import Profile_formatting

# Predefined appliance data
appliance_data_dict = {
    "Refrigerator": {"power": 150, "num": 1, "start": datetime.time(0, 0), "end": datetime.time(23, 59), "coincidence": 1.0, "min_time_on": 24.0},
    "Television": {"power": 100, "num": 1, "start": datetime.time(18, 0), "end": datetime.time(23, 0), "coincidence": 0.7, "min_time_on": 1.0},
    "Air Conditioner": {"power": 2000, "num": 1, "start": datetime.time(18, 0), "end": datetime.time(6, 0), "coincidence": 0.5, "min_time_on": 1.0},
    "Washing Machine": {"power": 500, "num": 1, "start": datetime.time(8, 0), "end": datetime.time(20, 0), "coincidence": 0.3, "min_time_on": 1.0},
    "Microwave": {"power": 1200, "num": 1, "start": datetime.time(6, 0), "end": datetime.time(22, 0), "coincidence": 0.2, "min_time_on": 0.5},
    "Electric Kettle": {"power": 2000, "num": 1, "start": datetime.time(6, 0), "end": datetime.time(22, 0), "coincidence": 0.2, "min_time_on": 0.2},
    "Computer": {"power": 150, "num": 1, "start": datetime.time(8, 0), "end": datetime.time(18, 0), "coincidence": 0.5, "min_time_on": 1.0},
    "Heater": {"power": 1500, "num": 1, "start": datetime.time(18, 0), "end": datetime.time(6, 0), "coincidence": 0.4, "min_time_on": 1.0},
    "Fan": {"power": 75, "num": 1, "start": datetime.time(0, 0), "end": datetime.time(23, 59), "coincidence": 0.8, "min_time_on": 24.0},
    "Light Bulb": {"power": 60, "num": 5, "start": datetime.time(18, 0), "end": datetime.time(6, 0), "coincidence": 0.9, "min_time_on": 1.0}
}

appliance_options = list(appliance_data_dict.keys())

# Function to collect appliance info
def collect_appliance_info(category, appliance, idx):
    st.subheader(f"Category: {category}, Appliance: {appliance}")
    num_appliances = st.number_input(f"Number of {appliance}(s) in {category}", min_value=0, step=1, value=appliance_data_dict[appliance]["num"], key=f"num_{category}_{appliance}_{idx}")
    power_rating = st.number_input(f"Power rating of each {appliance} (in watts)", min_value=0, step=1, value=appliance_data_dict[appliance]["power"], key=f"power_{category}_{appliance}_{idx}")
    start_time = st.time_input(f"Start time of use for {appliance}", value=appliance_data_dict[appliance]["start"], key=f"start_{category}_{appliance}_{idx}")
    end_time = st.time_input(f"End time of use for {appliance}", value=appliance_data_dict[appliance]["end"], key=f"end_{category}_{appliance}_{idx}")
    coincidence_factor = st.number_input(f"Coincidence factor for {appliance}", min_value=0.0, max_value=1.0, step=0.01, value=appliance_data_dict[appliance]["coincidence"], key=f"coincidence_{category}_{appliance}_{idx}")
    min_time_on = st.number_input(f"Minimum time the {appliance} is on (in hours)", min_value=0.0, max_value=24.0, step=0.5, value=appliance_data_dict[appliance]["min_time_on"], key=f"min_time_{category}_{appliance}_{idx}")
    return num_appliances, power_rating, start_time, end_time, coincidence_factor, min_time_on

# Function to model appliance usage with stochasticity
def model_usage(start_hour, end_hour, num_appliances, power, coincidence_factor, min_time_on):
    usage = np.zeros(24)
    if end_hour > start_hour:
        hours_of_use = range(start_hour, end_hour)
    else:
        hours_of_use = list(range(start_hour, 24)) + list(range(0, end_hour))

    active_hours = np.random.choice(hours_of_use, size=int(len(hours_of_use) * coincidence_factor), replace=False)
    for hour in active_hours:
        usage[hour] = power * num_appliances * np.random.uniform(0.5, 1.0)

    return usage

# Function to create user classes and appliances
def create_user_classes(appliance_data):
    users = []
    for category, appliance_info in appliance_data.items():
        user = User(user_name=category, num_users=appliance_info['num_users'])
        for appliance in appliance_info['appliances']:
            app = user.Appliance(
                appliance['num_appliances'],
                appliance['power_rating'],
                1,
                appliance['min_time_on']*60,
                appliance['coincidence_factor'],
                5,
                fixed="no"
            )
            app.windows(
                [int(appliance['start_time'].split(":")[0]) * 60 + int(appliance['start_time'].split(":")[1]), int(appliance['end_time'].split(":")[0]) * 60 + int(appliance['end_time'].split(":")[1])],
                [0, 0]
            )
        users.append(user)
    return users

def generate_load_profiles(users, start_date, end_date):
    use_case = UseCase(users=users, date_start=start_date, date_end=end_date)
    load_profile = use_case.generate_daily_load_profiles()
    return load_profile

def plot_load_profile(Profiles_series):
    Profiles_avg, Profiles_list_kW, Profiles_series = Profile_formatting(Profiles_series)
    minutes_per_day = 1440
    num_days = len(Profiles_series) // minutes_per_day
    Profiles_reshaped = Profiles_series.reshape((num_days, minutes_per_day))
    Profiles_daily_avg = Profiles_reshaped.mean(axis=0)

    plt.figure(figsize=(15, 10))
    for day in Profiles_reshaped:
        plt.plot(day / 1000, color='lightgrey', linewidth=0.5, alpha=0.3)
    Profiles_min = Profiles_reshaped.min(axis=0)
    Profiles_max = Profiles_reshaped.max(axis=0)
    plt.fill_between(range(minutes_per_day), Profiles_min / 1000, Profiles_max / 1000, color='grey', alpha=0.3, label='Variability Range')
    plt.plot(Profiles_daily_avg / 1000, color='red', linewidth=2, label='Average Daily Profile')

    plt.title('Daily Average Load Curve')
    plt.xlabel('Time of Day')
    plt.ylabel('Power [kW]')
    plt.xticks(np.linspace(0, minutes_per_day, 24), [f'{hour}:00' for hour in range(24)], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    return Profiles_daily_avg

def show():
    st.title("Demand Estimation with RAMP")
    
    # Predefined categories
    categories_predefined = ["High Income", "Middle Income", "Low Income"]
    categories = []

    for i in range(3):
        category_name = categories_predefined[i]
        category = st.text_input(f"Category {i+1} Name", value=category_name, key=f"category_{i}")
        num_users = st.number_input(f"Number of users in {category}", min_value=0, step=1, value=50, key=f"num_users_{i}")
        appliance_selections = st.multiselect(f"Select appliances for {category}", appliance_options, default=np.random.choice(appliance_options, 3, replace=False).tolist(), key=f"appliances_{i}")
        appliance_data = {}
        if appliance_selections:
            appliance_data[category] = {"num_users": num_users, "appliances": []}
            for idx, appliance in enumerate(appliance_selections):
                num_appliances, power_rating, start_time, end_time, coincidence_factor, min_time_on = collect_appliance_info(category, appliance, idx)
                appliance_data[category]["appliances"].append({
                    "num_appliances": num_appliances,
                    "power_rating": power_rating,
                    "start_time": start_time.strftime("%H:%M"),
                    "end_time": end_time.strftime("%H:%M"),
                    "coincidence_factor": coincidence_factor,
                    "min_time_on": min_time_on
                })
            categories.append(appliance_data)
    
    if st.button("Estimate Demand"):
        if not categories:
            st.warning("Please select appliances for at least one category.")
        else:
            appliance_data_combined = {k: v for d in categories for k, v in d.items()}
            users = create_user_classes(appliance_data_combined)
            progress_bar = st.progress(0)
            progress_bar.progress(10)
            today_date = datetime.date.today().strftime('%Y-%m-%d')
            load_profile = generate_load_profiles(users, today_date, today_date)
            progress_bar.progress(70)
            st.write("**Load Profile Generated:**")
            Profiles_daily_avg = plot_load_profile(load_profile)
            progress_bar.progress(100)
            df = pd.DataFrame(load_profile, columns=["Power (W)"])
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Load Profile as CSV",
                data=csv,
                file_name='load_profile.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
