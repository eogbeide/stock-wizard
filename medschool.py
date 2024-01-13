import streamlit as st
import pandas as pd

# Load the CSV file
df = pd.read_csv("Medical_School_Requirements.csv")

# Create a list of distinct medical school names
school_options = df['Medical School'].unique().tolist()

# Create a selectbox to choose a state
selected_state = st.sidebar.selectbox("Select a State", df['State'].unique())

# Filter the DataFrame based on the selected state
state_filtered_df = df[df['State'] == selected_state]

# Create a selectbox to choose a medical school within the selected state
selected_school = st.sidebar.selectbox("Select a Medical School", state_filtered_df['Medical School'].unique())

# Filter the DataFrame based on the selected school and exclude State and Medical School columns
filtered_df = state_filtered_df[state_filtered_df['Medical School'] == selected_school].drop(columns=['State', 'Medical School'])

# Check if 'Credit Hours' column exists in the filtered DataFrame
if 'Credit' in filtered_df.columns:
    # Format the 'CreditHours' column to display one significant figure
    filtered_df['Credit'] = filtered_df['Credit'].apply(lambda x: format(x, ".1g"))

    # Display the filtered DataFrame without the index column, with wrapped text in the 'Additional Info' column
    st.dataframe(filtered_df.reset_index(drop=True).style.set_properties(**{'white-space': 'pre-wrap'}))
else:
    st.write("No 'Credit Hours' column found in the filtered DataFrame.")
