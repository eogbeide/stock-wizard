import streamlit as st
import pandas as pds
import requests
from io import StringIO


import streamlit as st
from gtts import gTTS
from io import BytesIO

st.title("Simple Text to Speech Converter")

# Text input for user to enter content
text_input = st.text_area("Enter text to convert to speech:")

# Language selection
language = st.selectbox("Select language", ["en", "fr", "ru", "hi", "es"])

# Button to generate speech
if st.button("Generate my speech"):
    if text_input:
        # Create an audio stream from the text
        tts = gTTS(text_input, lang=language)
        audio_stream = BytesIO()
        tts.save(audio_stream)
        audio_stream.seek(0)  # Move to the start of the stream
        
        # Play the audio
        st.audio(audio_stream)
    else:
        st.warning("Please enter some text")

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
    
    # Print the columns for debugging
    #st.write("Available columns in the DataFrame:", data.columns.tolist())
    
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
    else:
        st.write("No topic available for this chapter.")

# Run the app
if __name__ == "__main__":
    main()
