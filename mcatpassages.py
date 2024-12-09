import streamlit as st
import pandas as pd
import requests
from io import StringIO
from gtts import gTTS
import os
import time

# Load the CSV file from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcat_passages.csv'
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad requests
    return pd.read_csv(StringIO(response.text))  # Use StringIO to load CSV data

# Function to convert text to speech
def text_to_speech(text):
    if not text or not isinstance(text, str):
        st.error("No valid text provided for speech.")
        return

    audio_file = "answer.mp3"
    
    # Check if audio file already exists
    if os.path.exists(audio_file):
        os.remove(audio_file)  # Remove the old file to avoid confusion
    
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        os.system("start " + audio_file)  # Change command based on OS
    except Exception as e:
        if "429" in str(e):
            st.error("Too many requests to the TTS API. Please try again later.")
        else:
            st.error(f"Error in text-to-speech conversion: {e}")
        time.sleep(5)  # Wait before retrying

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
            st.write(current_topic['Answer and Explanation'])
            text_to_speech(current_topic['Answer and Explanation'])  # Read the answer out loud
    else:
        st.write("No topic available for this chapter.")

# Run the app
if __name__ == "__main__":
    main()
