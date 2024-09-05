import streamlit as st
import pandas as pd

# Load data from the CSV file on GitHub with explicit encoding
url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/science.csv'
data = pd.read_csv(url, encoding='ISO-8859-1')  # Use 'utf-8' or 'ISO-8859-1'

# Sidebar for subject selection
subjects = data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subjects)

# Filter data based on selected subject
filtered_data = data[data['Subject'] == selected_subject]

# Display scenario in a box and show S/N
scenario_index = filtered_data.index[st.session_state.selected_index]
scenario = filtered_data['Scenario'].iloc[0]  # Assuming one scenario per subject
serial_number = filtered_data['S/N'].iloc[scenario_index]  # Assuming S/N column exists

st.subheader(f'Scenario (S/N: {serial_number})')
st.write(scenario)

# Initialize session state to track the selected question index
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 0

# Get the questions and answers
questions_answers = filtered_data['Question and Answer'].tolist()

# Display selected Question and Answer
selected_qa = questions_answers[st.session_state.selected_index]
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
