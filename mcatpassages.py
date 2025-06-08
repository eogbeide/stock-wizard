import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os

def play_text(text: str):
    """Convert text to speech, play it, then delete the temp file."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
    os.remove(fp.name)

def main():
    # Comic Sans + small font
    st.markdown(
        """
        <style>
        body { font-family: 'Comic Sans MS'; font-size:10px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("MCAT Labs")

    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labs.csv"
        return pd.read_csv(url, encoding='ISO-8859-1')

    df = load_data()
    df.columns = df.columns.str.strip()

    if 'Subject' not in df.columns:
        st.error("The 'Subject' column is missing from the data.")
        return

    # Subject/topic selectors
    subjects = df['Subject'].unique()
    sel_subj = st.sidebar.selectbox("Select Subject:", subjects)
    sub_df = df[df['Subject'] == sel_subj]

    topics = sub_df['Topic'].unique()
    sel_topic = st.sidebar.selectbox("Select Topic:", topics)
    topic_df = sub_df[sub_df['Topic'] == sel_topic]

    if topic_df.empty:
        st.write("No data available for the selected topic.")
        return

    row = topic_df.iloc[0]

    # --- Passage / Description ---
    st.subheader("Passage")
    desc = str(row.get('Description', '')).strip()
    st.info(desc)
    if st.button("🔊 Read Passage Aloud", key="tts_desc"):
        play_text(desc)

    # --- Questions & Answers ---
    st.subheader("Questions and Answers")
    qa = str(row.get('Questions and Answers', '')).strip()
    with st.expander("View Questions & Answers"):
        st.write(qa)
    if st.button("🔊 Read Q&A Aloud", key="tts_qa"):
        play_text(qa)

    # --- Explanation (if exists) ---
    if 'Explanation' in row.index and pd.notna(row['Explanation']):
        st.subheader("Explanation")
        exp = str(row['Explanation']).strip()
        st.info(exp)
        if st.button("🔊 Read Explanation Aloud", key="tts_exp"):
            play_text(exp)

if __name__ == "__main__":
    main()
