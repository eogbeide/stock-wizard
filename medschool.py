import streamlit as st
import pandas as pd

# Load the CSV file
df = pd.read_csv("Medical_School_Requirements2.csv")

# Remove leading and trailing spaces from all string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Create a list of distinct medical school names
school_options = df['Medical School'].unique().tolist()

# Page 1: Filter by state and medical school
st.sidebar.title("Page 1: Filter")
selected_state = st.sidebar.selectbox("Select a State", df['State'].unique())
state_filtered_df = df[df['State'] == selected_state]
selected_school = st.sidebar.selectbox("Select a Medical School", state_filtered_df['Medical School'].unique())
filtered_df = state_filtered_df[state_filtered_df['Medical School'] == selected_school].drop(columns=['State', 'Medical School'])

# Check if 'Credit Hours' column exists in the filtered DataFrame
if 'Credit Hours' in filtered_df.columns:
    # Convert 'Credit Hours' column to numeric values
    filtered_df['Credit Hours'] = pd.to_numeric(filtered_df['Credit Hours'], errors='coerce')

    # Format the 'Credit Hours' column to display one significant figure
    filtered_df['Credit Hours'] = filtered_df['Credit Hours'].apply(lambda x: format(x, ".1f") if pd.notnull(x) else "")

    # Display the filtered DataFrame without the index column, with wrapped text in the 'Additional Info' column
    st.write("Page 1: Filtered Data")
    st.dataframe(filtered_df.reset_index(drop=True).style.set_properties(**{'white-space': 'pre-wrap'}))
else:
    st.write("No 'Credit Hours' column found in the filtered DataFrame.")

# Page 2: Display filtered data with specific columns
st.title("Page 2: Filtered Data")
columns_to_display = ['Medical School', 'Credit Hours', 'Lab?', 'Additional Info']
st.dataframe(filtered_df[columns_to_display])
