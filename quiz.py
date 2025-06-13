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

# --- Global CSS ---
components.html("""
<style>
  .passage { 
    padding: 1em; background: #f9f9f9; border-left: 4px solid #0078d4; margin-bottom: 1em;
  }
  .question-block {
    margin: 1em 0; padding: 0.5em; background: #eef; border-left: 4px solid #004080;
  }
  .options {
    margin: 0.5em 0 1em 1.5em;
    list-style-type: disc;
  }
  .explanation-block { 
    margin: 1em 0; padding: 1em; background: #f0f9e8; border-left: 4px solid #40a860;
  }
</style>
""", height=0)

# --- Sidebar Navigation & Question Selector ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", quiz_data["Subject"].unique())
filtered = quiz_data[quiz_data["Subject"] == subject]

topic = st.sidebar.selectbox("Topic", filtered["Topic"].unique())
filtered = filtered[filtered["Topic"] == topic].reset_index(drop=True)

max_idx = len(filtered) - 1
if max_idx < 0:
    st.sidebar.warning("No questions here.")
    st.stop()

question_labels = [f"Question {j+1}" for j in range(max_idx+1)]
default_idx = st.session_state.get("idx", 0)
selected = st.sidebar.selectbox("Go to question", question_labels, index=default_idx)
st.session_state.idx = question_labels.index(selected)
i = st.session_state.idx

# --- TTS Helper ---
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
        try: os.remove(fp.name)
        except: pass

# --- Show Current Item ---
row = filtered.iloc[i]
passage = str(row["Passage"] or "").strip()
question = str(row["Question"] or "").strip()
answers = [a.strip() for a in str(row["Answer"] or "").split(";")]
explanation = str(row.get("Explanation","") or "").strip()

st.title(f"{subject} — {question_labels[i]} ({i+1}/{max_idx+1})")

# Passage
st.markdown(f'<div class="passage">{passage.replace("\\n","<br>")}</div>', unsafe_allow_html=True)
if st.button("🔊 Read Passage Aloud"):
    play_tts(passage)

# Question
st.markdown(f'<div class="question-block"><strong>Question:</strong><br>{question}</div>', unsafe_allow_html=True)

# Options
st.markdown(f'<ul class="options">{"".join(f"<li>{opt}</li>" for opt in answers)}</ul>', unsafe_allow_html=True)

# Explanation
if explanation:
    if st.checkbox("Show Explanation"):
        st.markdown(f'<div class="explanation-block">{explanation.replace("\\n","<br><br>")}</div>', unsafe_allow_html=True)
    if st.button("🔊 Read Explanation"):
        play_tts(explanation)

# Read full Q&A + Explanation
full_text = "\n\n".join([passage, f"Question: {question}", "Answers: " + "; ".join(answers)])
if explanation:
    full_text += "\n\nExplanation:\n" + explanation
if st.button("🔊 Read Q&A + Explanation Aloud"):
    play_tts(full_text)

# Navigation Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("◀️ Back") and i > 0:
        st.session_state.idx -= 1
        st.experimental_rerun()
with col2:
    if st.button("Next ▶️") and i < max_idx:
        st.session_state.idx += 1
        st.experimental_rerun()
