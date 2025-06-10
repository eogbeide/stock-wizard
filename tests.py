import streamlit as st
import pandas as pd
from gtts import gTTS
import io

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
    """Convert text to speech and play via an in-memory buffer."""
    if not text:
        st.warning("No explanation to read.")
        return
    buffer = io.BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(buffer)
    buffer.seek(0)
    st.audio(buffer, format="audio/mp3")

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

    # Get current explanation
    explanation = str(filtered.loc[idx, 'Explanation']).strip()

    # Sidebar play aloud button
    if st.sidebar.button("🔊 Play Explanation (Sidebar)"):
        play_text(explanation)

    # Main content
    st.subheader(f"{selected_subject} (Explanation {idx+1} of {max_idx+1})")
    st.write(explanation)

    # Main play aloud button
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
