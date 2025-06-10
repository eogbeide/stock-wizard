import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import re
import os

# Load data from Excel on GitHub
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

quiz_data = load_data()

st.sidebar.title('Quiz Navigation')
if quiz_data.empty:
    st.warning("No quiz data available.")
    st.stop()

# Subject/topic selection
subjects = quiz_data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subjects)
filtered = quiz_data[quiz_data['Subject'] == selected_subject]

topics = filtered['Topic'].unique()
selected_topic = st.sidebar.selectbox('Select Topic', topics)
filtered = filtered[filtered['Topic'] == selected_topic].reset_index(drop=True)

# Ensure idx is within bounds
if 'idx' not in st.session_state:
    st.session_state.idx = 0
max_idx = len(filtered) - 1
if max_idx < 0:
    st.warning("No questions for this Subject/Topic.")
    st.stop()
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))

def format_html(text: str) -> str:
    """Convert raw text into HTML-formatted paragraphs with line breaks."""
    text = str(text).strip()
    paragraphs = re.split(r'\n\s*\n', text)
    return ''.join(f"<p>{p.strip().replace('\n', '<br>')}</p>" for p in paragraphs)

def play_tts(text: str):
    """Convert text to speech and serve audio file (mobile-safe)."""
    if not text:
        st.warning("No text to read.")
        return

    tts = gTTS(text=text, lang='en')
    # create a temp file that persists until we delete it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()

    tts.save(tmp_name)
    # Safari/iOS requires the file to still exist when st.audio is rendered
    st.audio(tmp_name, format='audio/mp3', start_time=0)

    # Optionally clean up after a short delay (won't affect playback)
    # os.remove(tmp_name)

def show_item(i: int):
    row = filtered.iloc[i]

    passage = str(row['Passage']).strip()
    formatted_passage = format_html(passage)

    answers = [opt.strip() for opt in str(row['Answer']).split(';')]
    question = f"Question {i+1}: {row['Question']}"
    qa_text = f"{question}\nAnswers:\n" + "\n".join(f"- {a}" for a in answers)

    raw_exp = row.get('Explanation', '')
    explanation = str(raw_exp).strip() if pd.notna(raw_exp) else ''
    full_tts_text = qa_text + (f"\nExplanation:\n{explanation}" if explanation else "")

    # Sidebar audio controls
    st.sidebar.markdown("### 🔊 Audio Controls")
    if st.sidebar.button("▶️ Play Passage (Sidebar)", key=f"sb_passage_{i}"):
        play_tts(passage)
    if st.sidebar.button("▶️ Play Full (Sidebar)", key=f"sb_full_{i}"):
        play_tts(full_tts_text)

    # Main UI
    st.markdown("### 📘 Passage")
    st.markdown(formatted_passage, unsafe_allow_html=True)
    if st.button("🔊 Read Passage Aloud", key=f"tts_passage_{i}"):
        play_tts(passage)

    st.markdown(f"```text\n{qa_text}\n```")
    if explanation and st.checkbox("Show Explanation", key=f"show_exp_{i}"):
        st.markdown("### 📝 Explanation")
        st.markdown(format_html(explanation), unsafe_allow_html=True)

    if st.button("🔊 Read Q&A + Explanation Aloud", key=f"tts_full_{i}"):
        play_tts(full_tts_text)

# Navigation Buttons
col1, _, col2 = st.columns([1, 4, 1])
with col1:
    if st.button("◀️ Back") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶️") and st.session_state.idx < max_idx:
        st.session_state.idx += 1

show_item(st.session_state.idx)
