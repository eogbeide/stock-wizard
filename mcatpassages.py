import streamlit as st
import pandas as pd
import requests
from io import StringIO
from gtts import gTTS, gTTSError
import os
import tempfile

# Load the CSV file from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcat_passages.csv'
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad requests
    return pd.read_csv(StringIO(response.text))  # Use StringIO to load CSV data

# Main function
def main():
    # Load data
    data = load_data()
    
    # Clean column names
    data.columns = data.columns.str.strip()

    # Sidebar for subject selection
    st.sidebar.title("Select Subject")
    subjects = data['Subject'].unique()
    selected_subject = st.sidebar.selectbox("Choose a Subject", subjects)

    # Sidebar for chapter selection based on selected subject
    st.sidebar.title("Select Chapter")
    chapters = data[data['Subject'] == selected_subject]['Chapter'].unique()
    selected_chapter = st.sidebar.selectbox("Choose a Chapter", chapters)

    # Filter data based on selected subject and chapter
    chapter_data = data[(data['Subject'] == selected_subject) & (data['Chapter'] == selected_chapter)]

    # Initialize session state for topic index
    if 'topic_index' not in st.session_state:
        st.session_state.topic_index = 0

    # Navigation buttons at the top
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.topic_index > 0:
            if st.button("Back"):
                st.session_state.topic_index -= 1
    
    with col2:
        if st.session_state.topic_index < len(chapter_data) - 1:
            if st.button("Next"):
                st.session_state.topic_index += 1

    # Display current topic
    if not chapter_data.empty:
        current_topic = chapter_data.iloc[st.session_state.topic_index]
        
        st.title(f"Subject: {selected_subject} - Chapter: {selected_chapter}")
        st.subheader(f"Topic {st.session_state.topic_index + 1}: {current_topic['Topic']}")
        
        if st.button("Show Answer"):
            answer_text = current_topic['Answer and Explanation']
            st.write(answer_text)

            try:
                # Convert text to speech
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                    tts = gTTS(text=answer_text, lang='en')
                    tts.save(temp_audio_file.name)
                    st.audio(temp_audio_file.name, format='audio/mp3')
                    
            except gTTSError as e:
                st.error("Error generating audio: " + str(e))
                
            except Exception as e:
                st.error("An unexpected error occurred: " + str(e))
                
    else:
        st.write("No topic available for this chapter.")

# Run the app
if __name__ == "__main__":
    main()
