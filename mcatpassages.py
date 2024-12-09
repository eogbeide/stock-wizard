import streamlit as st
import pandas as pd
import requests
from io import StringIO
from gtts import gTTS
import time
import os

# Load the CSV file from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcat_passages.csv'
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

# Function to convert text to speech
def text_to_speech(text):
    try:
        # Clean the text by removing special characters
        clean_text = ''.join(e for e in text if e.isalnum() or e.isspace() or e in ['.', ',', '!', '?'])
        
        # Split the text into manageable chunks if necessary
        max_length = 200
        chunks = [clean_text[i:i + max_length] for i in range(0, len(clean_text), max_length)]

        audio_files = []
        
        for idx, chunk in enumerate(chunks):
            audio_file = f"answer_{idx}.mp3"

            # Check if the audio file already exists
            if not os.path.exists(audio_file):
                for attempt in range(5):  # Retry up to 5 times
                    try:
                        tts = gTTS(chunk, lang='en')
                        tts.save(audio_file)
                        break  # Break if successful
                    except Exception as e:
                        if "429" in str(e):
                            wait_time = 5 * (2 ** attempt)  # Exponential backoff
                            st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)  # Wait before retrying
                        else:
                            st.error(f"An error occurred: {e}")
                            return []
            audio_files.append(audio_file)

        return audio_files

    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {e}")
        return []

# Main function
def main():
    data = load_data()
    data.columns = data.columns.str.strip()

    st.sidebar.title("Select Subject")
    subjects = data['Subject'].unique()
    selected_subject = st.sidebar.selectbox("Choose a Subject", subjects)

    st.sidebar.title("Select Chapter")
    chapters = data[data['Subject'] == selected_subject]['Chapter'].unique()
    selected_chapter = st.sidebar.selectbox("Choose a Chapter", chapters)

    chapter_data = data[(data['Subject'] == selected_subject) & (data['Chapter'] == selected_chapter)]

    if 'topic_index' not in st.session_state:
        st.session_state.topic_index = 0

    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.topic_index > 0:
            if st.button("Back"):
                st.session_state.topic_index -= 1
    
    with col2:
        if st.session_state.topic_index < len(chapter_data) - 1:
            if st.button("Next"):
                st.session_state.topic_index += 1

    if not chapter_data.empty:
        current_topic = chapter_data.iloc[st.session_state.topic_index]
        
        st.title(f"Subject: {selected_subject} - Chapter: {selected_chapter}")
        st.subheader(f"Topic {st.session_state.topic_index + 1}: {current_topic['Topic']}")
        
        if st.button("Show Answer"):
            answer_text = current_topic['Answer and Explanation']
            st.write(answer_text)
            audio_files = text_to_speech(answer_text)
            
            for audio_file in audio_files:
                st.audio(audio_file, format='audio/mp3')

    else:
        st.write("No topic available for this chapter.")

# Run the app
if __name__ == "__main__":
    main()
