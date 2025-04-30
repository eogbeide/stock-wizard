import streamlit as sts
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
    # Display scenario as formatted text with single line spacing
    scenario = data['scenario'].unique()[0]  # Assuming you want to display the first scenario
    formatted_scenario = f"""
    <h3>Scenario Overview</h3>
    <p style="line-height: 1.0;"><strong>Scenario:</strong> {scenario}</p>
    <p style="line-height: 1.0;">This scenario covers various aspects related to the topic. Please select the category and section to explore specific questions.</p>
    """
    st.markdown(formatted_scenario, unsafe_allow_html=True)

    # Dropdowns for category and section
    category = st.selectbox("Select a Category", data['category'].unique())
    section = st.selectbox("Select a Section", data['section'].unique())

    # Display questions based on selections
    st.subheader("Questions")
    filtered_data = data[(data['scenario'] == scenario) & (data['category'] == category) & (data['section'] == section)]
    
    for index, row in filtered_data.iterrows():
        question = row['question']
        st.write(question)
        
        # Create a Google search link for the source
        source_link = row['source']  # This should contain the URL from the Excel file
        google_search_link = f"https://www.google.com/search?q={source_link.replace(' ', '+')}"

        # Clickable option to show solution and source
        if st.button("Show Solution"):
            st.write(f"**Solution:** {row['solution']}")
            # Ensure that the source is a valid URL
            if pd.notna(source_link):  # Check if the link is not NaN
                st.markdown(f"**Source:** ({source_link})", unsafe_allow_html=True)
                # Display the Google search link
                st.markdown(f"[Refer to source]({google_search_link})", unsafe_allow_html=True)
            else:
                st.write("**Source:** No link provided.")
