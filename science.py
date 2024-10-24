import streamlit as st
import pandas as pd

# Create a timestamp to force a refresh
#today = datetime.datetime.now().date()
#st.write(f"Last updated: {today}")

# Load data from the CSV file on GitHub with explicit encoding
url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/science.csv'
data = pd.read_csv(url, encoding='ISO-8859-1')

# Sidebar for subject selection
subjects = data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subjects)

# Filter data based on the selected subject
filtered_data = data[data['Subject'] == selected_subject]

# Sidebar for topic selection
topics = filtered_data['Topic'].unique()  # Assuming the 'Topic' column exists
selected_topic = st.sidebar.selectbox('Select Topic', topics)

# Further filter data based on the selected topic
filtered_data = filtered_data[filtered_data['Topic'] == selected_topic]

# Initialize session state to track the selected index
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 0

# Create a list of scenarios for easy navigation
if len(filtered_data) > 0:
    scenarios = filtered_data['Scenario'].tolist()

    # Navigation buttons at the top
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Back'):
            if st.session_state.selected_index > 0:
                st.session_state.selected_index -= 1

    with col2:
        if st.button('Next'):
            if st.session_state.selected_index < len(scenarios) - 1:
                st.session_state.selected_index += 1

    # Get the current scenario and its serial number
    current_index = st.session_state.selected_index
    scenario = scenarios[current_index]
    serial_number = filtered_data['S/N'].iloc[current_index]  # Assuming S/N column exists

    # Display scenario in a box with S/N
    st.subheader(f'Scenario (S/N: {serial_number})')
    st.write(scenario)

    # Get the questions and answers related to the current scenario
    questions_answers = filtered_data['Question and Answer'].tolist()

    # Expandable section for Question and Answer
    with st.expander('Question and Answer'):
        selected_qa = questions_answers[current_index]
        st.write(selected_qa)

    # Display the current scenario index
    st.write(f"Scenario {current_index + 1} of {len(filtered_data)}")
else:
    st.write("No scenarios available for the selected Subject and Topic.")
