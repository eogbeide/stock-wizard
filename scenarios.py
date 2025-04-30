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
    # Welcome box at the start with adjusted size
    st.markdown("<div style='padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; background-color: #f9f9f9;'>"
                "<h2 style='color: #4CAF50; margin: 0;'>Welcome to the ABA ORAL EXAM PRACTICE</h2>"
                "<p style='margin: 0;'>This application allows you to explore various question scenarios. "
                "You can select different categories and sections to answer specific questions related to the scenario given.</p>"
                "<p style='margin: 0;'>Please select a category and section from the dropdowns below to get started.</p>"
                "</div>", unsafe_allow_html=True)

    # Display scenario in an expandable box (not expanded by default)
    scenario = data['scenario'].unique()[0]
    with st.expander("Scenario Overview", expanded=False):  # Changed to expanded=False
        st.markdown(f"<div style='padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; background-color: black; color: white;'>"
                    f"<strong style='color:#4CAF50;'>Scenario:</strong> {scenario}<br>"
                    f"This scenario covers various aspects related to the topic. Please select the category and section to explore specific questions."
                    "</div>", unsafe_allow_html=True)

    # Dropdown for category selection with bold and green label
    st.markdown("<div style='margin: 0;'><strong style='color:#4CAF50;'>Select a Category:</strong></div>", unsafe_allow_html=True)
    category = st.selectbox("", data['category'].unique())

    # Filter sections based on selected scenario and category
    filtered_sections = data[data['category'] == category]['section'].unique()
    st.markdown("<div style='margin: 0;'><strong style='color:#4CAF50;'>Select a Section:</strong></div>", unsafe_allow_html=True)
    section = st.selectbox("", filtered_sections)

    # Display questions based on selections
    st.markdown("<h4 style='font-size: 16px; margin: 0;'>Questions</h4>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)  # Another horizontal line for separation
    filtered_data = data[(data['scenario'] == scenario) & (data['category'] == category) & (data['section'] == section)]
    
    for index, row in filtered_data.iterrows():
        question = row['question']
        st.markdown(f"<h5 style='font-size: 14px; margin: 0;'>Question {index + 1}: {question}</h5>", unsafe_allow_html=True)  # Reduced font size for questions
        
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
