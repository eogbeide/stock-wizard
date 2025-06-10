import os
import re
import tempfile

import pandas as pds
import streamlit as st
from google.cloud import texttospeech

# Make sure your service account JSON key file path is in this env var
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your-key.json"

# URL of the Excel file
URL = "https://github.com/eogbeide/stock-wizard/raw/main/tests.xlsx"

@st.cache_data
def load_data():
    """Load test explanations from GitHub."""
    try:
        return pd.read_excel(URL)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_passage(text: str) -> str:
    """Convert raw text into HTML-formatted paragraphs with line breaks."""
    paragraphs = re.split(r'\n\s*\n', str(text).strip())
    return ''.join(f"<p>{p.replace('\n', '<br>')}</p>" for p in paragraphs)

def play_text(text: str, voice_name="en-US-Wavenet-D"):
    """
    Convert text to speech via Google Cloud TTS and play the MP3.
    Default uses a high-quality male Wavenet voice.
    """
    if not text:
        st.warning("No text to read.")
        return

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Synthesize speech
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write to temp file and play
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        fp.write(response.audio_content)
        tmp_path = fp.name

    st.audio(tmp_path, format="audio/mp3")

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Test Explanations with Google Cloud TTS")

    df = load_data()
    if df.empty:
        st.stop()

    # Sidebar: Subject selection + Playback
    st.sidebar.title("Select Subject")
    subjects = df['Subject'].unique()
    subject = st.sidebar.selectbox("Subject", subjects)

    filtered = df[df['Subject'] == subject].reset_index(drop=True)
    if filtered.empty:
        st.sidebar.warning("No explanations for this subject.")
        st.stop()

    # Index state
    if 'idx' not in st.session_state or st.session_state.get('last') != subject:
        st.session_state.idx = 0
        st.session_state.last = subject

    idx = st.session_state.idx
    max_idx = len(filtered) - 1

    # Display explanation
    explanation = str(filtered.loc[idx, 'Explanation']).strip()
    st.subheader(f"{subject} ({idx+1} of {max_idx+1})")
    st.markdown(format_passage(explanation), unsafe_allow_html=True)

    # Sidebar playback
    st.sidebar.markdown("### 🔊 Audio Controls")
    if st.sidebar.button("▶ Play Explanation"):
        play_text(explanation)

    # Main playback
    if st.button("▶ Play Explanation"):
        play_text(explanation)

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back") and idx > 0:
            st.session_state.idx -= 1
    with col2:
        if st.button("Next ▶") and idx < max_idx:
            st.session_state.idx += 1

if __name__ == "__main__":
    main()
