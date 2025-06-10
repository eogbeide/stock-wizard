import streamlit as sts
import pandas as pd
from gtts import gTTS
import tempfile
import re
import os

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
    """Convert plain text into HTML-formatted paragraphs with line breaks."""
    text = str(text).strip()
    paragraphs = re.split(r'\n\s*\n', text)
    html_passage = ''.join(f"<p>{p.strip().replace('\n', '<br>')}</p>" for p in paragraphs)
    return html_passage

def play_text(text: str):
    """Convert text to speech and serve audio file (iOS/mobile compatible)."""
    if not text:
        st.warning("No explanation to read.")
        return
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name
    st.audio(audio_path, format="audio/mp3")

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Test Explanations with TTS")

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

    # Reset index if subject changes
    if 'idx' not in st.session_state or st.session_state.get('last_subject') != selected_subject:
        st.session_state.idx = 0
        st.session_state.last_subject = selected_subject

    max_idx = len(filtered) - 1
    idx = st.session_state.idx

    # Current explanation
    explanation = str(filtered.loc[idx, 'Explanation']).strip()
    formatted_explanation = format_passage(explanation)

    st.subheader(f"{selected_subject} (Explanation {idx+1} of {max_idx+1})")
    st.markdown(formatted_explanation, unsafe_allow_html=True)

    # TTS Buttons
    st.markdown("#### 🔊 Listen")
    if st.button("▶ Play Explanation"):
        play_text(explanation)

    st.sidebar.markdown("### 🔊 Listen")
    if st.sidebar.button("▶ Play Explanation (Sidebar)"):
        play_text(explanation)

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back") and idx > 0:
            st.session_state.idx = idx - 1
    with col2:
        if st.button("Next ▶") and idx < max_idx:
            st.session_state.idx = idx + 1

if __name__ == "__main__":
    main()
