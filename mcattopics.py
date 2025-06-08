import streamlit as st
import pandas as pd
from urllib.error import URLError
from gtts import gTTS
import re
import tempfile
import os

@st.cache_data
def read_questions_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        return df
    except UnicodeDecodeError:
        st.error("Error decoding the CSV file. Trying a different encoding.")
        return pd.DataFrame()
    except URLError as e:
        st.error(f"Error fetching data: {e.reason}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

def clean_text(text: str) -> str:
    text = re.sub(r'[*#]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    return text.strip()

def play_text(text: str):
    """Generate TTS, play it, then delete the temp file."""
    t = clean_text(text)
    if not t:
        st.warning("Nothing to read.")
        return

    tts = gTTS(text=t, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
    os.remove(fp.name)

def main():
    st.title("MCAT Topics Explanation with TTS")

    FILE_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcattopics.csv"
    df = read_questions_from_csv(FILE_URL)
    if df.empty:
        st.write("No data available.")
        return

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        's/n': 'serial_number',
        'subject': 'subject',
        'topic': 'topic',
        'passage': 'passage',           # assuming it exists
        'question': 'question',
        'answer': 'answer',             # assuming it exists
        'explanation': 'explanation'
    }, inplace=True)

    required = ['subject','topic','passage','question','answer','explanation']
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return

    # Sidebar filters
    subjects = df['subject'].unique()
    subj = st.sidebar.selectbox("Subject", ["All"] + list(subjects))
    if subj != "All":
        df = df[df['subject']==subj]

    topics = df['topic'].unique()
    top = st.sidebar.selectbox("Topic", ["All"] + list(topics))
    if top != "All":
        df = df[df['topic']==top]

    if df.empty:
        st.write("No questions for that selection.")
        return

    # session state index
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    n = len(df)
    st.session_state.idx = st.session_state.idx % n  # wrap around if you like

    row = df.iloc[st.session_state.idx]

    # --- Passage ---
    st.subheader("Passage")
    st.write(row['passage'])
    if st.button("🔊 Read Passage Aloud", key="read_passage"):
        play_text(row['passage'])

    # --- Q & A ---
    qa_block = f"Q: {row['question']}\nA: {row['answer']}"
    st.subheader("Question & Answer")
    st.write(qa_block.replace("\n", "  \n"))  # preserve line break
    if st.button("🔊 Read Q&A Aloud", key="read_qa"):
        play_text(qa_block)

    # --- Explanation ---
    st.subheader("Explanation")
    st.write(row['explanation'])
    if st.button("🔊 Read Explanation Aloud", key="read_exp"):
        play_text(row['explanation'])

    # --- Navigation ---
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("◀ Back"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
    with col3:
        if st.button("Next ▶"):
            st.session_state.idx = min(n-1, st.session_state.idx + 1)

if __name__ == "__main__":
    main()
