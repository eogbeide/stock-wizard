import streamlit as st
import pandas as pd

# Load data from the CSV file on GitHub
url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/science.csv'  # Update with the actual path
data = pd.read_csv(url)

# Sidebar for subject selection
subjects = data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subjects)

# Filter data based on selected subject
filtered_data = data[data['Subject'] == selected_subject]

# Display scenario in a box
scenario = filtered_data['Scenario'].iloc[0]  # Assuming one scenario per subject
st.subheader('Scenario')
st.write(scenario)

# Dropdown for Question and Answer
questions_answers = [(row['Question and Answer'], row['Question and Answer']) for index, row in filtered_data.iterrows()]
selected_qa = st.selectbox('Select Question and Answer', questions_answers)

# Display selected Question and Answer
st.subheader('Question and Answer')
st.write(selected_qa[0])  # Display the selected Q&A

# Optional: Add additional formatting or functionality as needed
