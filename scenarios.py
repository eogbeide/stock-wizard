import streamlit as st
import pandas as pd
import requests
from io import StringIO  # Import StringIO from the io module


# Function to load data from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/scenarios.xls'  # Update with your raw URL
    response = requests.get(url)
    if response.status_code == 200:
        with open('scenarios.xlsx', 'wb') as f:
            f.write(response.content)
        df = pd.read_excel('scenarios.xlsx')
        return df
    else:
        st.error("Failed to load data from GitHub.")
        return None

# Load the data
data = load_data()

if data is not None:
    # Streamlit app layout
    st.title("Scenario Questions")

    # Dropdowns for scenario, category, and section
    scenario = st.selectbox("Select a Scenario", data['scenario'].unique())
    category = st.selectbox("Select a Category", data['category'].unique())
    section = st.selectbox("Select a Section", data['section'].unique())

    # Display questions based on selections
    st.subheader("Questions")
    filtered_data = data[(data['scenario'] == scenario) & (data['category'] == category) & (data['section'] == section)]
    
    for index, row in filtered_data.iterrows():
        question = row['question']
        st.write(question)
        
        # Clickable option to show solution and source
        if st.button(f"Show solution for: {question}"):
            st.write(f"**Solution:** {row['solution']}")
            st.write(f"**Source:** {row['source']}")
