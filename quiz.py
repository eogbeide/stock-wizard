import streamlit as st
import pandas as pd
import re
import io
from gtts import gTTS

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

st.sidebar.title('Quiz Navigation')
if quiz_data.empty:
    st.sidebar.warning("No quiz data available.")
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
    st.sidebar.warning("No questions for this Subject/Topic.")
    st.stop()
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))

def format_html(text: str) -> str:
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n', '<br>')}</p>" for p in paras)

def make_audio_buffer(text: str) -> io.BytesIO:
    """Generate TTS audio into a BytesIO buffer."""
    tts = gTTS(text=text, lang='en')
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

def show_item(i: int):
    row = filtered.iloc[i]

    # Passage
    passage = str(row['Passage']).strip()
    st.markdown("### 📘 Passage")
    st.markdown(format_html(passage), unsafe_allow_html=True)

    # Q&A
    answers = [opt.strip() for opt in str(row['Answer']).split(';')]
    question = f"Question {i+1}: {row['Question']}"
    qa_text = f"{question}\nAnswers:\n" + "\n".join(f"- {a}" for a in answers)
    st.markdown(f"```text\n{qa_text}\n```")

    # Explanation (optional)
    explanation = str(row.get('Explanation','')).strip()
    if explanation:
        if st.checkbox("Show Explanation", key=f"exp_{i}"):
            st.markdown("### 📝 Explanation")
            st.markdown(format_html(explanation), unsafe_allow_html=True)

    # Sidebar audio players
    st.sidebar.markdown("### 🔊 Audio Controls")
    passage_buf = make_audio_buffer(passage)
    st.sidebar.audio(passage_buf, format="audio/mp3", start_time=0)

    full_text = qa_text + (f"\nExplanation:\n{explanation}" if explanation else "")
    full_buf = make_audio_buffer(full_text)
    st.sidebar.audio(full_buf, format="audio/mp3", start_time=0)

# Navigation Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("◀️ Back") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶️") and st.session_state.idx < max_idx:
        st.session_state.idx += 1

show_item(st.session_state.idx)
