import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import re

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
    formatted = ''.join(f"<p>{p.strip().replace('\n', '<br>')}</p>" for p in paragraphs)
    return formatted

def generate_audio_bytes(text: str):
    """Generate TTS audio as a bytes buffer."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        with open(fp.name, 'rb') as audio_file:
            audio_bytes = audio_file.read()
    return audio_bytes

def show_item(i: int):
    row = filtered.iloc[i]

    # --- Format content ---
    passage = str(row['Passage']).strip()
    formatted_passage = format_html(passage)

    answers = [opt.strip() for opt in str(row['Answer']).split(';')]
    question = f"Question {i+1}: {row['Question']}"
    qa_text = f"{question}\nAnswers:\n" + "\n".join(f"- {a}" for a in answers)

    raw_exp = row.get('Explanation', '')
    explanation = str(raw_exp).strip() if pd.notna(raw_exp) else ''
    full_tts_text = f"{question}\n" + "\n".join(answers)
    if explanation:
        full_tts_text += f"\nExplanation:\n{explanation}"

    # --- Generate audio once ---
    passage_audio = generate_audio_bytes(passage)
    full_audio = generate_audio_bytes(full_tts_text)

    # --- Sidebar audio controls ---
    st.sidebar.markdown("### 🔊 Audio Player (Sidebar)")
    st.sidebar.audio(passage_audio, format='audio/mp3', start_time=0)
    st.sidebar.audio(full_audio, format='audio/mp3', start_time=0)

    # --- Main UI ---
    st.markdown("### 📘 Passage")
    st.markdown(formatted_passage, unsafe_allow_html=True)

    st.markdown("### 🔊 Audio Player (Top)")
    st.audio(passage_audio, format='audio/mp3', start_time=0)
    st.audio(full_audio, format='audio/mp3', start_time=0)

    st.markdown(f"```text\n{qa_text}\n```")

    if explanation and st.checkbox("Show Explanation", key=f"show_exp_{i}"):
        st.markdown("### 📝 Explanation")
        st.markdown(format_html(explanation), unsafe_allow_html=True)

# Navigation Buttons
col1, _, col2 = st.columns([1, 4, 1])
with col1:
    if st.button("◀️ Back") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶️") and st.session_state.idx < max_idx:
        st.session_state.idx += 1

show_item(st.session_state.idx)
