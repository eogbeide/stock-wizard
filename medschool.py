import streamlit as st
import pandas as pd
import openpyxl

# Load the Excel file
df = pd.read_csv("Medical_School_Requirements.csv")

# Create a list of medical school options
school_options = df['Medical School'].tolist()

# Create a selectbox to choose a medical school
selected_school = st.sidebar.selectbox("Select a Medical School", school_options)

# Filter the DataFrame based on the selected school
filtered_df = df[df['Medical School'] == selected_school]

# Display the filtered DataFrame
st.dataframe(filtered_df)
