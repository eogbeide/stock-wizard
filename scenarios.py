import streamlit as st
import pandas as pd
import requests

# Function to load data from GitHub
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/scenarios.xls'  # Update with your raw URL
    response = requests.get(url)
    if response.status_code == 200:
        with open('scenarios.xls', 'wb') as f:
            f.write(response.content)
        df = pd.read_excel('scenarios.xls')
        return df
    else:
        st.error("Failed to load data from GitHub.")
        return None

# Load the data
data = load_data()

if data is not None:
    # Debugging: Show the first few rows and the columns of the DataFrame
    #st.write(data.head())  # Check the DataFrame structure
    #st.write("Columns available:", data.columns.tolist())  # Show column names

    # Streamlit app layout
    st.title("Scenario Questions")

    # Display scenario as formatted text
    scenario = data['scenario'].unique()[0]  # Assuming you want to display the first scenario
    formatted_scenario = f"### Scenario Overview\n\n**Scenario:** {scenario}\n\nThis scenario covers various aspects related to the topic. Please select the category and section to explore specific questions."
    st.markdown(formatted_scenario)

    # Dropdowns for category and section
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
