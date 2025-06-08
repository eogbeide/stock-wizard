import streamlit as st
import pandas as pd
import re
from gtts import gTTS
import tempfile
import os

# Helper to play arbitrary text via gTTS
def play_text(text: str):
    # Clean up text
    cleaned = text.strip()
    if not cleaned:
        st.warning("Nothing to read.")
        return
    tts = gTTS(text=cleaned, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
    os.remove(fp.name)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/flashcards.csv"
    try:
        return pd.read_csv(url, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    st.title("Flashcards")

    df = load_data()
    if df.empty:
        st.error("No data available. Please check the CSV file URL.")
        return

    df.columns = df.columns.str.strip()  # Clean column names

    # Subject selector
    subjects = df['Subject'].unique()
    selected_subject = st.sidebar.selectbox("Select Subject:", subjects)
    subject_data = df[df['Subject'] == selected_subject]

    if subject_data.empty:
        st.write("No topics available for the selected subject.")
        return

    # Topic selector
    topics = subject_data['Topic'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", topics)
    topic_data = subject_data[subject_data['Topic'] == selected_topic]

    if topic_data.empty:
        st.write("No flashcards for the selected topic.")
        return

    # Track which row and flashcard pair
    if 'current_row_index' not in st.session_state:
        st.session_state.current_row_index = 0
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    # Ensure row index in bounds
    if st.session_state.current_row_index >= len(topic_data):
        st.write("No more flashcards available for the selected topic.")
        return

    # Get current row
    row = topic_data.iloc[st.session_state.current_row_index]
    text = row['Questions and Answers']

    # Extract QA pairs
    pairs = re.findall(
        r'Flashcard \d+:\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\s*Flashcard \d+:|$)',
        text, re.DOTALL
    )
    if not pairs:
        st.error("No valid question/answer pairs found. Check CSV format.")
        return

    # Clamp question_index
    total = len(pairs)
    st.session_state.question_index = min(st.session_state.question_index, total - 1)
    q, a = pairs[st.session_state.question_index]
    q = q.strip()
    a = a.strip()

    # Display flashcard
    flash_num = st.session_state.question_index + 1
    st.subheader(f"Flashcard {flash_num}: {q}")

    # Read question aloud
    if st.button("🔊 Read Question Aloud", key="read_q"):
        play_text(q)

    # Show answer in expander
    with st.expander("Show Answer"):
        st.info(a)
        if st.button("🔊 Read Answer Aloud", key="read_a"):
            play_text(a)

    # Read full flashcard (Q + A)
    if st.button("🔊 Read Flashcard Aloud", key="read_full"):
        play_text(f"Question: {q}\nAnswer: {a}")

    # Navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("◀ Back", key="nav_back"):
            if st.session_state.question_index > 0:
                st.session_state.question_index -= 1
            else:
                # go to previous row if available
                if st.session_state.current_row_index > 0:
                    st.session_state.current_row_index -= 1
                    st.session_state.question_index = 0
    with col3:
        if st.button("Next ▶", key="nav_next"):
            if st.session_state.question_index < total - 1:
                st.session_state.question_index += 1
            else:
                # advance to next row
                st.session_state.current_row_index += 1
                st.session_state.question_index = 0
    with col2:
        if st.button("Reset", key="nav_reset"):
            st.session_state.current_row_index = 0
            st.session_state.question_index = 0

if __name__ == "__main__":
    main()
