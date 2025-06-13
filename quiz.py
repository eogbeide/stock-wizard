import streamlit as st
import pandas as pd
import re
import tempfile
import os
from gtts import gTTS
import streamlit.components.v1 as components

# --- Load & cache data ---
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return pd.DataFrame()

quiz_data = load_data()
if quiz_data.empty:
    st.sidebar.warning("No quiz data available.")
    st.stop()

# --- Global CSS for HTML formatting ---
components.html("""
<style>
  .passage { 
    padding: 1em; 
    background: #f5f5f5; 
    border-left: 4px solid #0078d4; 
    margin-bottom: 1em;
  }
  .question { 
    margin-top: 1em; 
    font-weight: bold; 
  }
  .options {
    margin-left: 1em;
    list-style-type: disc;
  }
  .explanation { 
    margin-top: 1em; 
    padding: 0.8em; 
    background: #e8f4fd; 
    border-left: 4px solid #00a3e0;
  }
</style>
""", height=0)

# --- Sidebar navigation & question selector ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", quiz_data["Subject"].unique())
filtered = quiz_data[quiz_data["Subject"] == subject]

topic = st.sidebar.selectbox("Topic", filtered["Topic"].unique())
filtered = filtered[filtered["Topic"] == topic].reset_index(drop=True)

# Compute max index
max_idx = len(filtered) - 1
if max_idx < 0:
    st.sidebar.warning("No questions here.")
    st.stop()

# Question slider
current_q = st.sidebar.slider(
    "Go to question",
    min_value=1,
    max_value=max_idx + 1,
    value=st.session_state.get("idx", 0) + 1,
    format="Question %d"
)
# Update session index
st.session_state.idx = current_q - 1
i = st.session_state.idx

# --- Helper to play TTS ---
def play_tts(text: str):
    if not text:
        st.warning("No text to read.")
        return
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
    except Exception:
        st.error("🔊 Text-to-speech failed.")
    finally:
        try:
            os.remove(fp.name)
        except:
            pass

# --- Render current item ---
row = filtered.iloc[i]
passage = str(row["Passage"]).strip()
answers = [a.strip() for a in str(row["Answer"]).split(";")]
question_html = f"Question {i+1}: {row['Question']}"
explanation = str(row.get("Explanation","") or "").strip()

# Passage
st.markdown('<div class="passage">{}</div>'.format(
    "<br><br>".join(re.split(r"\n\s*\n", passage.replace("\n","<br>")))
), unsafe_allow_html=True)
if st.button("🔊 Read Passage Aloud"):
    play_tts(passage)

# Question & Options
st.markdown(f'<div class="question">{question_html}</div>', unsafe_allow_html=True)
st.markdown(f'<ul class="options">{"".join(f"<li>{opt}</li>" for opt in answers)}</ul>', unsafe_allow_html=True)

# Explanation
if explanation and st.checkbox("Show Explanation"):
    st.markdown('<div class="explanation">{}</div>'.format(
        "<br><br>".join(re.split(r"\n\s*\n", explanation.replace("\n","<br>")))
    ), unsafe_allow_html=True)

# Read full Q&A + Explanation
full_text = passage + "\n\n" + question_html + "\n" + "\n".join(f"- {opt}" for opt in answers)
if explanation:
    full_text += "\n\nExplanation:\n" + explanation
if st.button("🔊 Read Q&A + Explanation Aloud"):
    play_tts(full_text)

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("◀️ Back") and i > 0:
        st.session_state.idx -= 1
        st.experimental_rerun()
with col2:
    if st.button("Next ▶️") and i < max_idx:
        st.session_state.idx += 1
        st.experimental_rerun()
