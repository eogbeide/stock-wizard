import streamlit as st
import pandas as pd
import re
import io
from gtts import gTTS
import streamlit.components.v1 as components

# --- Load & cache data ---
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

# --- Sidebar navigation ---
st.sidebar.title("Quiz Navigation")
subjects = quiz_data['Subject'].unique()
subject = st.sidebar.selectbox("Subject", subjects)
filtered = quiz_data[quiz_data['Subject'] == subject]

topics = filtered['Topic'].unique()
topic = st.sidebar.selectbox("Topic", topics)
filtered = filtered[filtered['Topic'] == topic].reset_index(drop=True)

# --- Session state for index ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0
max_idx = len(filtered) - 1
if max_idx < 0:
    st.sidebar.warning("No questions for this selection.")
    st.stop()
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))
i = st.session_state.idx

# --- Style snippet ---
components.html("""
<style>
  .passage { padding: 0.5em; background: #f9f9f9; border-radius: 4px; }
  .question { margin-top: 1em; }
  .question strong { font-size: 1.1em; }
  .question ul { margin-top: 0.2em; padding-left: 1.2em; }
  .explanation { margin-top: 1em; color: #555; font-style: italic; }
  .audio-controls { margin: 0.5em 0; }
</style>
""", height=0)

def format_html(text: str) -> str:
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

def make_audio_buffer(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    tts = gTTS(text=text, lang='en')
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

# --- Display content + audio controls ---
row = filtered.iloc[i]
passage = str(row['Passage']).strip()
answers = [a.strip() for a in str(row['Answer']).split(';')]
question = f"Question {i+1}: {row['Question']}"
qa_text = question + "\n" + "\n".join(f"- {a}" for a in answers)
explanation = str(row.get('Explanation','') or '').strip()

# prepare buffers
passage_buf = make_audio_buffer(passage)
full_buf = make_audio_buffer(qa_text + ("\n\nExplanation:\n"+explanation if explanation else ""))

# --- Top audio players ---
st.markdown("### 🔊 Audio Controls (Top)")
st.audio(passage_buf, format="audio/mp3", start_time=0)
st.audio(full_buf, format="audio/mp3", start_time=0)

# --- Main content ---
st.markdown(f"## {subject} ({i+1} of {max_idx+1})")

st.markdown(f'<div class="passage">{format_html(passage)}</div>', unsafe_allow_html=True)

st.markdown('<div class="question"><strong>' + question + '</strong><ul>' +
            ''.join(f'<li>{a}</li>' for a in answers) +
            '</ul></div>', unsafe_allow_html=True)

if explanation:
    if st.checkbox("Show Explanation"):
        st.markdown('<div class="explanation">' + format_html(explanation) + '</div>', unsafe_allow_html=True)

# --- Sidebar audio players ---
st.sidebar.markdown("### 🔊 Audio Controls (Sidebar)")
st.sidebar.audio(passage_buf, format="audio/mp3", start_time=0)
st.sidebar.audio(full_buf, format="audio/mp3", start_time=0)

# --- Navigation ---
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶") and i < max_idx:
        st.session_state.idx += 1
