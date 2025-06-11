import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os

# Load data from the Excel file URL
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/tests.xlsx"
    try:
        df = pd.read_excel(url)
        df = df.dropna(subset=['Subject', 'Explanation'])
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Convert explanation to audio
def play_text(text: str):
    if not text:
        st.warning("No explanation to read.")
        return
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
        os.remove(fp.name)

def main():
    st.title("Subject Explanations (Text to Voice Enabled)")

    df = load_data()
    if df.empty:
        st.warning("No data available.")
        return

    subjects = df['Subject'].unique().tolist()
    selected_subject = st.sidebar.selectbox("Choose a Subject", subjects)

    subject_df = df[df['Subject'] == selected_subject].reset_index(drop=True)

    if 'index' not in st.session_state:
        st.session_state.index = 0

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Back"):
            st.session_state.index = max(0, st.session_state.index - 1)
    with col3:
        if st.button("Next"):
            st.session_state.index = min(len(subject_df) - 1, st.session_state.index + 1)

    current_row = subject_df.iloc[st.session_state.index]
    st.markdown(f"### Explanation {st.session_state.index + 1} of {len(subject_df)}")
    st.write(current_row['Explanation'])

    if st.button("🔊 Read Aloud"):
        play_text(current_row['Explanation'])

if __name__ == "__main__":
    main()
