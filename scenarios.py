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
    # Initialize session state for current scenario index
    if 'current_scenario_index' not in st.session_state:
        st.session_state.current_scenario_index = 0

    # Sidebar for selecting unique scenario numbers
    st.sidebar.markdown("<h3 style='color:#4CAF50;'>Select Scenario Number:</h3>", unsafe_allow_html=True)
    scenario_options = data['scenario#'].unique()  # Use the existing scenario# column
    selected_scenario_number = st.sidebar.selectbox("", scenario_options)

    # Update current scenario index based on selection
    st.session_state.current_scenario_index = selected_scenario_number - 1  # Adjust for zero-based index

    # Welcome box at the start with black background
    st.markdown("<div style='padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; background-color: black; color: white;'>"
                "<h2 style='color: #4CAF50; margin: 0;'>Welcome to the ABA ORAL EXAM PRACTICE</h2>"
                "<p style='margin: 0;'>This application allows you to explore various oral exam question scenarios. "
                "You can select different topics and topic questions to answer specific questions related to the scenario given.</p>"
                "<p style='margin: 0;'>Please select a scenario# from the side bar and topic and topic question from the dropdowns below to get started.</p>"
                "</div>", unsafe_allow_html=True)

    # Get the current scenario
    current_index = st.session_state.current_scenario_index
    scenario = data.loc[data['scenario#'] == selected_scenario_number, 'scenario'].values[0]

    # Display scenario in an expandable box (not expanded by default)
    with st.expander("Scenario Overview", expanded=False):
        st.markdown(f"<div style='padding: 10px; border: 1px solid #4CAF50; border-radius: 5px; background-color: black; color: white;'>"
                    f"<strong style='color:#4CAF50;'>Scenario {selected_scenario_number}:</strong> {scenario}<br>"
                    f"This scenario covers various aspects related to the topic. Please select the category and section to explore specific questions."
                    "</div>", unsafe_allow_html=True)

    # Dropdown for category selection with bold and green label
    st.markdown("<div style='margin: 0;'><strong style='color:#4CAF50;'>Select a Topic:</strong></div>", unsafe_allow_html=True)
    category = st.selectbox("", data['category'].unique())

    # Filter sections based on selected scenario and category
    filtered_sections = data[data['category'] == category]['section'].unique()
    st.markdown("<div style='margin: 0;'><strong style='color:#4CAF50;'>Select a Topic Question:</strong></div>", unsafe_allow_html=True)
    section = st.selectbox("", filtered_sections)

    # Display questions based on selections
    st.markdown("<h4 style='font-size: 16px; margin: 0;'>Questions</h4>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)  # Another horizontal line for separation
    filtered_data = data[(data['scenario#'] == selected_scenario_number) & (data['category'] == category) & (data['section'] == section)]
    
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
                st.markdown(f"**Source:** ({source_link})", unsafe_allow_html=True)
                st.markdown(f"[Refer to source]({google_search_link})", unsafe_allow_html=True)
            else:
                st.write("**Source:** No link provided.")
        st.markdown("<hr>", unsafe_allow_html=True)  # Separator after each solution
