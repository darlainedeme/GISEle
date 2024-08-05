import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ramp import UseCase, User
from ramp.post_process.post_process import Profile_formatting
import time

# Sample data for identified buildings and estimated population
# Replace these with actual data in your application
identified_buildings = 150  # Example number of combined buildings identified
estimated_population = 600  # Example total estimated population

# Define a function to create user classes and appliances
def create_user_classes():
    users = []

    HI = User(user_name="High-Income Household", num_users=25)
    HI.Appliance(6, 7, 2, 120, 0.1, 5).windows([1170, 1440], [0, 30])
    HI.Appliance(2, 60, 3, 180, 0.1, 5).windows([720, 900], [1170, 1440], 0.35, [0, 60])
    HI.Appliance(1, 8, 3, 60, 0.1, 5).windows([720, 900], [1170, 1440], 0.35, [0, 60])
    HI.Appliance(1, 8, 3, 120, 0.1, 5).windows([720, 900], [1170, 1440], 0.35, [0, 60])
    HI.Appliance(5, 2, 2, 300, 0.2, 5).windows([1110, 1440], [0, 30], 0.35)
    HI.Appliance(1, 200, 1, 1440, 0, 30, "yes", 3).windows([0, 1440], [0, 0])
    HI.Appliance(1, 50, 3, 30, 0.1, 1, occasional_use=0.33).windows([420, 480], [660, 750], 0.35, [1140, 1200])
    users.append(HI)

    MI = User(user_name="Middle-Income Household", num_users=75)
    MI.Appliance(3, 7, 2, 120, 0.2, 10).windows([1170, 1440], [0, 30], 0.35)
    MI.Appliance(2, 13, 2, 600, 0.2, 10).windows([0, 330], [1170, 1440], 0.35)
    MI.Appliance(1, 60, 3, 90, 0.1, 5).windows([450, 660], [720, 840], 0.35, [1170, 1440])
    MI.Appliance(1, 8, 3, 30, 0.1, 5).windows([450, 660], [720, 840], 0.35, [1170, 1440])
    MI.Appliance(1, 8, 3, 60, 0.1, 5).windows([450, 660], [720, 840], 0.35, [1170, 1440])
    MI.Appliance(4, 2, 1, 300, 0.2, 5).windows([1020, 1440], [0, 0], 0.35)
    MI.Appliance(1, 50, 2, 30, 0.1, 1, occasional_use=0.33).windows([660, 750], [1110, 1200], 0.35)
    users.append(MI)

    LI = User(user_name="Low-Income Household", num_users=50)
    LI.Appliance(2, 7, 2, 120, 0.2, 10).windows([1170, 1440], [0, 30], 0.35)
    LI.Appliance(1, 13, 2, 600, 0.2, 10).windows([0, 330], [1170, 1440], 0.35)
    LI.Appliance(1, 60, 3, 90, 0.1, 5).windows([750, 840], [1170, 1440], 0.35, [0, 30])
    LI.Appliance(1, 8, 3, 30, 0.1, 5).windows([750, 840], [1170, 1440], 0.35, [0, 30])
    LI.Appliance(1, 8, 3, 60, 0.1, 5).windows([750, 840], [1170, 1440], 0.35, [0, 30])
    LI.Appliance(2, 2, 1, 300, 0.2, 5).windows([1080, 1440], [0, 0], 0.35)
    users.append(LI)

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
    st.write(f"**Combined Buildings Identified:** {identified_buildings}")
    st.write(f"**Total Estimated Population:** {estimated_population}")

    # Inputs for the RAMP simulation
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    # Button to generate load profiles
    if st.button("Estimate Demand"):
        users = create_user_classes()
        
        # Show progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        load_profile = generate_load_profiles(users, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        progress_bar.progress(70)

        st.write("**Load Profile Generated:**")
        Profiles_daily_avg = plot_load_profile(load_profile)
        progress_bar.progress(100)

        # Convert Profiles_series to a DataFrame for CSV export
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
