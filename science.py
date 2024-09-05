import streamlit as st
import pandas as pd

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

# Initialize session state to track the selected S/N
if 'selected_sn' not in st.session_state:
    st.session_state.selected_sn = None

# Create a mapping of S/N to index
if len(filtered_data) > 0:
    sn_to_index = {sn: index for index, sn in enumerate(filtered_data['S/N'])}

    # Sidebar for S/N selection
    selected_sn = st.sidebar.selectbox('Select S/N', filtered_data['S/N'].tolist(), index=0)
    st.session_state.selected_sn = selected_sn

    # Get the current index based on the selected S/N
    current_index = sn_to_index[st.session_state.selected_sn]

    # Navigation buttons at the top
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Back'):
            current_serials = filtered_data['S/N'].tolist()
            current_index = current_serials.index(st.session_state.selected_sn)
            if current_index > 0:
                st.session_state.selected_sn = current_serials[current_index - 1]

    with col2:
        if st.button('Next'):
            current_serials = filtered_data['S/N'].tolist()
            current_index = current_serials.index(st.session_state.selected_sn)
            if current_index < len(current_serials) - 1:
                st.session_state.selected_sn = current_serials[current_index + 1]

    # Get the current scenario and its serial number
    if st.session_state.selected_sn is not None:  # Ensure there's data available
        scenario = filtered_data['Scenario'].iloc[current_index]
        serial_number = st.session_state.selected_sn

        # Display scenario in a box with S/N
        st.subheader(f'Scenario (S/N: {serial_number})')
        st.write(scenario)

        # Get the questions and answers related to the current scenario
        questions_answers = filtered_data['Question and Answer'].tolist()

        # Display selected Question and Answer
        selected_qa = questions_answers[current_index]
        st.subheader('Question and Answer')
        st.write(selected_qa)

        # Display the current scenario index
        st.write(f"Scenario {current_index + 1} of {len(filtered_data)}")
else:
    st.write("No scenarios available for the selected Subject and Topic.")
