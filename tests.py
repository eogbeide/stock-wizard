import streamlit as stsd
import pandas as pd
from gtts import gTTS
import io
import re

# URL of the Excel file
URL = "https://github.com/eogbeide/stock-wizard/raw/main/tests.xlsx"

@st.cache_data
def load_data():
    """Load test explanations from GitHub."""
    try:
        df = pd.read_excel(URL)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_passage(text):
    """Convert plain text into safe HTML paragraph format with line breaks."""
    text = str(text).strip()
    # Replace multiple newlines with double <br> and wrap in <p> tags
    paragraphs = re.split(r'\n\s*\n', text)
    html_passage = ''.join(f"<p>{p.strip().replace('\n', '<br>')}</p>" for p in paragraphs)
    return html_passage

def play_text(text: str):
    """Convert text to speech and play via in-memory buffer (uses default gTTS female voice)."""
    if not text:
        st.warning("No explanation to read.")
        return
    buffer = io.BytesIO()
    tts = gTTS(text=text, lang='en')  # gTTS does not support voice gender selection
    tts.write_to_fp(buffer)
    buffer.seek(0)
    st.audio(buffer, format="audio/mp3")

# Optional: use Google Cloud TTS for male voice (commented out)
# def play_text_male(text: str):
#     from google.cloud import texttospeech
#     client = texttospeech.TextToSpeechClient()
#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US", name="en-US-Wavenet-D", ssml_gender=texttospeech.SsmlVoiceGender.MALE
#     )
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
#     response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
#     st.audio(response.audio_content, format='audio/mp3')

def main():
    st.set_page_config(layout="wide")
    st.title("Test Explanations with TTS")

    # Load data
    df = load_data()
    if df.empty:
        st.stop()

    # Sidebar: Subject selection
    st.sidebar.title("Select Subject")
    subjects = df['Subject'].unique()
    selected_subject = st.sidebar.selectbox("Subject", subjects)

    # Filter explanations for the selected subject
    filtered = df[df['Subject'] == selected_subject].reset_index(drop=True)
    if filtered.empty:
        st.warning("No explanations found for this subject.")
        st.stop()

    # Initialize or reset index when subject changes
    if 'idx' not in st.session_state or st.session_state.get('last_subject') != selected_subject:
        st.session_state.idx = 0
        st.session_state.last_subject = selected_subject

    max_idx = len(filtered) - 1
    idx = st.session_state.idx

    # Display current explanation with HTML formatting
    explanation = str(filtered.loc[idx, 'Explanation']).strip()
    formatted_explanation = format_passage(explanation)
    st.subheader(f"{selected_subject} (Explanation {idx+1} of {max_idx+1})")
    st.markdown(formatted_explanation, unsafe_allow_html=True)

    # Play aloud buttons
    if st.button("🔊 Play Explanation"):
        play_text(explanation)
    if st.sidebar.button("🔊 Play Explanation (Sidebar)"):
        play_text(explanation)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back") and idx > 0:
            st.session_state.idx = idx - 1
    with col2:
        if st.button("Next ▶") and idx < max_idx:
            st.session_state.idx = idx + 1

if __name__ == "__main__":
    main()
