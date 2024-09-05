import streamlit as st
import pandas as pd

# Load data from the CSV file on GitHub with explicit encoding
url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/science.csv'
data = pd.read_csv(url, encoding='ISO-8859-1')  # Try 'utf-8' or 'ISO-8859-1'

# Sidebar for subject selection
subjects = data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subjects)

# Filter data based on selected subject
filtered_data = data[data['Subject'] == selected_subject]

# Display scenario in a box
scenario = filtered_data['Scenario'].iloc[0]  # Assuming one scenario per subject
st.subheader('Scenario')
st.write(scenario)

# Initialize session state to track the selected question index
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 0

# Dropdown for Question and Answer
questions_answers = filtered_data['Question and Answer'].tolist()
selected_qa = questions_answers[st.session_state.selected_index]

# Display selected Question and Answer
st.subheader('Question and Answer')
st.write(selected_qa)

# Navigation buttons
col1, col2 = st.columns(2)

with col1:
    if st.button('Back'):
        if st.session_state.selected_index > 0:
            st.session_state.selected_index -= 1

with col2:
    if st.button('Next'):
        if st.session_state.selected_index < len(questions_answers) - 1:
            st.session_state.selected_index += 1

# Display the current question index
st.write(f"Question {st.session_state.selected_index + 1} of {len(questions_answers)}")
