import streamlit as st
import pandas as pd
import requests

# Function to load data from GitHub
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/scenarios.xls'
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
    # Display scenario as formatted text with single line spacing
    scenario = data['scenario'].unique()[0]
    st.markdown(f"<h3 style='color:#4CAF50;'>Scenario Overview</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='line-height: 1.5;'><strong>Scenario:</strong> {scenario}</p>", unsafe_allow_html=True)
    st.markdown("<p>This scenario covers various aspects related to the topic. Please select the category and section to explore specific questions.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line for separation

    # Dropdown for category selection
    category = st.selectbox("Select a Category", data['category'].unique())

    # Filter sections based on selected scenario and category
    filtered_sections = data[data['category'] == category]['section'].unique()
    section = st.selectbox("Select a Section", filtered_sections)

    # Display questions based on selections
    st.subheader("Questions")
    st.markdown("<hr>", unsafe_allow_html=True)  # Another horizontal line for separation
    filtered_data = data[(data['scenario'] == scenario) & (data['category'] == category) & (data['section'] == section)]
    
    for index, row in filtered_data.iterrows():
        question = row['question']
        st.markdown(f"### Question {index + 1}: {question}")  # Display questions with a header style
        st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line for each question
        
        # Create a Google search link for the source
        source_link = row['source']
        google_search_link = f"https://www.google.com/search?q={source_link.replace(' ', '+')}"

        # Clickable option to show solution and source
        if st.button(f"Show Solution for Question {index + 1}"):
            st.write(f"**Solution:** {row['solution']}")
            if pd.notna(source_link):
                st.markdown(f"**Source:** [{source_link}]({source_link})", unsafe_allow_html=True)
                st.markdown(f"[Refer to source]({google_search_link})", unsafe_allow_html=True)
            else:
                st.write("**Source:** No link provided.")
        st.markdown("<hr>", unsafe_allow_html=True)  # Separator after each solution
