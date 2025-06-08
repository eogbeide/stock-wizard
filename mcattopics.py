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
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        st.error("Error decoding the CSV file. Trying a different encoding.")
    except URLError as e:
        st.error(f"Error fetching data: {e.reason}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return pd.DataFrame()

def clean_text(text: str) -> str:
    text = re.sub(r'[*#]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    return text.strip()

def play_text(text: str):
    tts = gTTS(text=clean_text(text), lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
    os.remove(fp.name)

def main():
    st.title("MCAT Topics Explanation")

    FILE_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcattopics.csv"
    df = read_questions_from_csv(FILE_URL)
    if df.empty:
        st.write("No data available.")
        return

    # Normalize & rename
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        's/n':'serial_number',
        'subject':'subject',
        'topic':'topic',
        'question':'question',
        'explanation':'explanation'
    }, inplace=True)

    # Validate
    for col in ['subject','topic','question','explanation']:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return

    # Sidebar
    subjects = df['subject'].unique()
    selected_subject = st.sidebar.selectbox("Select Subject:", ["All"] + list(subjects))
    if selected_subject != "All":
        df = df[df['subject']==selected_subject]

    topics = df['topic'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", ["All"] + list(topics))
    if selected_topic != "All":
        df = df[df['topic']==selected_topic]

    if df.empty:
        st.write("No questions for that filter.")
        return

    # Question index state
    if 'idx' not in st.session_state:
        st.session_state.idx = 0
    n = len(df)
    st.session_state.idx = min(st.session_state.idx, n-1)

    row = df.iloc[st.session_state.idx]

    # Render
    st.markdown(f"**Subject:** {selected_subject}")
    st.markdown(f"**Topic:** {selected_topic}")

    st.markdown("### Question")
    st.success(row['question'])

    with st.expander("View Explanation"):
        st.write(row['explanation'])

    # Navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("◀ Back"):
            st.session_state.idx = max(0, st.session_state.idx-1)
    with col3:
        if st.button("Next ▶"):
            st.session_state.idx = min(n-1, st.session_state.idx+1)
    with col2:
        if st.button("Reset"):
            st.session_state.idx = 0

    # Build entire page text for TTS
    page_text = "\n".join([
        f"Subject: {selected_subject}",
        f"Topic: {selected_topic}",
        f"Question: {row['question']}",
        f"Explanation: {row['explanation']}"
    ])

    st.markdown("---")
    if st.button("🔊 Read Page Aloud"):
        play_text(page_text)

if __name__ == "__main__":
    main()
