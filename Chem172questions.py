import streamlit as st
import pandas as pd
import requests
from io import StringIO  # Import StringIO from the io module

# Load the CSV file from GitHub
#@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Chem172_questions.csv'  # Raw URL for the CSV
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad requests
    return pd.read_csv(StringIO(response.text))  # Use StringIO from io

# Main function
def main():
    # Load data
    data = load_data()
    
    # Sidebar for chapter selection
    st.sidebar.title("Select Chapter")
    chapters = data['Chapter'].unique()
    selected_chapter = st.sidebar.selectbox("Choose a Chapter", chapters)

    # Filter data based on selected chapter
    chapter_data = data[data['Chapter'] == selected_chapter]

    # Initialize session state for question index
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    # Display current question
    current_question = chapter_data.iloc[st.session_state.question_index]
    
    st.title(f"Chapter: {selected_chapter}")
    st.subheader(f"Question {st.session_state.question_index + 1}: {current_question['Question']}")
    
    if st.button("Show Answer"):
        st.write(current_question['Answer and Explanation'])

    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.question_index > 0:
            if st.button("Back"):
                st.session_state.question_index -= 1
    
    with col2:
        if st.session_state.question_index < len(chapter_data) - 1:
            if st.button("Next"):
                st.session_state.question_index += 1

# Run the app
if __name__ == "__main__":
    main()
