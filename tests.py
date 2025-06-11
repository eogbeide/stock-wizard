import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
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

def play_text(text: str):
    """Convert text to speech, play it, then delete the temp file."""
    if not text:
        st.warning("No explanation to read.")
        return
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
    os.remove(fp.name)

def main():
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

    # Display current explanation
    explanation = str(filtered.loc[idx, 'Explanation']).strip()
    st.subheader(f"{selected_subject} (Explanation {idx+1} of {max_idx+1})")
    st.write(explanation)

    # Play aloud button
    if st.button("🔊 Play Explanation"):
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
