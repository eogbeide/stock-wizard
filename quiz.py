import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os
import re

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

quiz_data = load_data()
if quiz_data.empty:
    st.sidebar.warning("No quiz data available.")
    st.stop()

# Sidebar navigation
st.sidebar.title('Quiz Navigation')
subject = st.sidebar.selectbox('Select Subject', quiz_data['Subject'].unique())
filtered = quiz_data[quiz_data['Subject'] == subject]

topic = st.sidebar.selectbox('Select Topic', filtered['Topic'].unique())
filtered = filtered[filtered['Topic'] == topic].reset_index(drop=True)

# Session index
if 'idx' not in st.session_state:
    st.session_state.idx = 0
max_idx = len(filtered) - 1
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))
i = st.session_state.idx

# Helper to format paragraphs
def format_html_paragraphs(text: str) -> str:
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

# Play TTS
def play_tts(text: str):
    if not text:
        st.warning("No text to read.")
        return
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
    except Exception:
        st.error("🔊 Text-to-speech failed.")
    finally:
        try: os.remove(fp.name)
        except: pass

def show_item(idx: int):
    row = filtered.iloc[idx]
    # Passage
    st.markdown("### 📘 Passage")
    passage_html = format_html_paragraphs(str(row['Passage']))
    st.markdown(passage_html, unsafe_allow_html=True)
    if st.button("🔊 Read Passage Aloud", key=f"play_passage_{idx}"):
        play_tts(row['Passage'])

    # Question & Answers
    st.markdown("### ❓ Question")
    question_html = f"<p><strong>Question {idx+1}:</strong> {row['Question']}</p>"
    options = [opt.strip() for opt in str(row['Answer']).split(';')]
    options_html = "<ul>" + "".join(f"<li>{opt}</li>" for opt in options) + "</ul>"
    st.markdown(question_html + options_html, unsafe_allow_html=True)

    # Explanation
    explanation = str(row.get('Explanation','') or '').strip()
    if explanation:
        if st.checkbox("Show Explanation", key=f"show_exp_{idx}"):
            st.markdown("### 📝 Explanation")
            exp_html = format_html_paragraphs(explanation)
            st.markdown(exp_html, unsafe_allow_html=True)

    # Full TTS
    full_text = f"{row['Passage']}\n\nQuestion: {row['Question']}. Options: {'; '.join(options)}"
    if explanation:
        full_text += f"\n\nExplanation: {explanation}"
    if st.button("🔊 Read Q&A + Explanation", key=f"play_full_{idx}"):
        play_tts(full_text)

# Top controls
st.markdown("---")
if st.button("⬅️ Previous Question"):
    if i > 0:
        st.session_state.idx -= 1
if st.button("Next Question ➡️"):
    if i < max_idx:
        st.session_state.idx += 1
st.markdown("---")

show_item(i)
