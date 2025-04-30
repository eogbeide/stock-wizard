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
    # Display scenario in an expandable box
    scenario = data['scenario'].unique()[0]
    with st.expander("Scenario Overview: Click to Expand for Details", expanded=True):
        st.markdown(f"<div style='padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; background-color: black; color: white;'>"
                    f"<strong style='color:#4CAF50;'>Scenario:</strong> {scenario}<br>"
                    f"This scenario covers various aspects related to the topic. Please select the category and section to explore specific questions."
                    "</div>", unsafe_allow_html=True)

    # Dropdown for category selection
    category = st.selectbox("Select a Category", data['category'].unique())

    # Filter sections based on selected scenario and category
    filtered_sections = data[data['category'] == category]['section'].unique()
    section = st.selectbox("Select a Section", filtered_sections)

    # Display questions based on selections
    st.markdown("<h4 style='font-size: 16px; margin: 0;'>Questions</h4>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)  # Another horizontal line for separation
    filtered_data = data[(data['scenario'] == scenario) & (data['category'] == category) & (data['section'] == section)]

    # Initialize a variable to hold the source link
    source_link = None
    google_search_link = None

    # Display questions and get the source link from the first question
    for index, row in filtered_data.iterrows():
        question = row['question']
        st.markdown(f"<h5 style='font-size: 14px; margin: 0;'>Question {index + 1}: {question}</h5>", unsafe_allow_html=True)  # Reduced font size for questions
        
        # Get source link only from the first question
        if index == 0:
            source_link = row['source']
            google_search_link = f"https://www.google.com/search?q={source_link.replace(' ', '+')}"

        # Clickable option to show solution
        if st.button(f"Show Solution for Question {index + 1}"):
            st.write(f"**Solution:** {row['solution']}")

    # Display source link once after all questions
    if pd.notna(source_link):
        st.markdown(f"**Source:** [{source_link}]({source_link})", unsafe_allow_html=True)
        st.markdown(f"[Refer to source]({google_search_link})", unsafe_allow_html=True)
    else:
        st.write("**Source:** No link provided.")

    st.markdown("<hr>", unsafe_allow_html=True)  # Separator after the source
