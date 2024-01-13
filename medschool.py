import streamlit as st
import pandas as pd

# Load the CSV file
df = pd.read_csv("Medical_School_Requirements.csv")

# Create a list of distinct medical school names
school_options = df['Medical School'].unique().tolist()

# Create a selectbox to choose a medical school
selected_school = st.sidebar.selectbox("Select a Medical School", school_options)

# Filter the DataFrame based on the selected school and exclude State and Medical School columns
filtered_df = df[df['Medical School'] == selected_school].drop(columns=['State', 'Medical School'])

# Display the filtered DataFrame without the index column
st.dataframe(filtered_df.reset_index(drop=True))
