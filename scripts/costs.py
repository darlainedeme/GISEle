import streamlit as st
import pandas as pd
from openpyxl import Workbook
import os

# Function to create an Excel file with the given data
def create_excel(data, file_path):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Define initial cost values based on general knowledge
initial_costs = {
    "Grid Extension": {
        "Transmission Lines ($/km)": 50000,
        "Distribution Lines ($/km)": 30000,
        "Poles and Towers ($/unit)": 2000,
        "Transformers ($/unit)": 10000,
        "Substations ($/unit)": 50000,
        "Right-of-Way Acquisition ($/km)": 5000,
        "Household Connections ($/connection)": 300,
        "Service Drops ($/connection)": 100,
        "O&M - Maintenance ($/year/km)": 500,
        "O&M - Labor ($/year/person)": 20000,
        "Regulatory - Permitting and Licensing ($)": 10000,
        "Regulatory - Compliance ($/year)": 5000,
    },
    "Off-Grid Systems": {
        "Solar PV Systems ($/kW)": 1000,
        "Battery Storage ($/kWh)": 500,
        "Wind Turbines ($/kW)": 1500,
        "Micro-Hydro Systems ($/kW)": 2000,
        "Diesel Generators ($/kW)": 300,
        "Hybrid Systems ($/kW)": 1200,
        "Installation - Site Preparation ($)": 5000,
        "Installation - Labor ($/kW)": 100,
        "O&M - Maintenance ($/year/kW)": 50,
        "O&M - Fuel ($/liter)": 1,
        "O&M - Battery Replacement ($/kWh)": 300,
        "Regulatory - Permitting and Licensing ($)": 5000,
        "Regulatory - Compliance ($/year)": 2000,
        "Monitoring Systems ($)": 1000,
        "Management Costs ($/year)": 10000,
    },
    "Economic and Social Costs": {
        "Backup Systems ($)": 5000,
        "Resilience Costs ($)": 10000,
        "Environmental Mitigation ($)": 5000,
        "Community Engagement ($)": 2000,
        "Health and Safety ($)": 3000,
    },
    "Financing Costs": {
        "Interest and Loan Repayments (%)": 5,
        "Grants and Subsidies ($)": 20000,
    },
}

# Function to display a section and collect user inputs
def display_section(section_name, section_data):
    st.header(section_name)
    user_inputs = {}
    for cost_name, cost_value in section_data.items():
        user_inputs[cost_name] = st.number_input(cost_name, value=cost_value)
    return user_inputs

def main():
    st.title("Technoeconomic Analysis Cost Inputs")

    # Display each section and collect user inputs
    user_data = {}
    for section_name, section_data in initial_costs.items():
        user_data[section_name] = display_section(section_name, section_data)

    # Button to confirm and save the inputs as an Excel file
    if st.button("Confirm and Save"):
        data_to_save = {section: pd.DataFrame(list(data.items()), columns=["Cost Component", "Value"]) for section, data in user_data.items()}
        create_excel(data_to_save, "cost_inputs.xlsx")
        st.success("Cost inputs saved as cost_inputs.xlsx in the working directory.")

    # Button to export the inputs as an Excel file
    if st.button("Export as Excel"):
        data_to_save = {section: pd.DataFrame(list(data.items()), columns=["Cost Component", "Value"]) for section, data in user_data.items()}
        file_path = "cost_inputs_export.xlsx"
        create_excel(data_to_save, file_path)
        with open(file_path, "rb") as f:
            st.download_button("Download Excel File", f, file_name="cost_inputs_export.xlsx")
        st.success("Cost inputs exported as an Excel file.")

if __name__ == "__main__":
    main()

