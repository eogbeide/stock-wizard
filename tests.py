import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

# --- Load & cache data ---
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/tests.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.sidebar.warning("No data available.")
    st.stop()

# --- Global CSS for formatting ---
components.html("""
<style>
  .explanation-block { 
    margin: 1em 0; 
    padding: 1em; 
    background: #f5f5f5; 
    border-left: 4px solid #0078d4;
  }
</style>
""", height=0)

# --- Sidebar: Subject selector ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
subset = df[df["Subject"] == subject].reset_index(drop=True)

# --- Sidebar: Question selector (dropdown) ---
max_idx = len(subset) - 1
if max_idx < 0:
    st.sidebar.warning("No explanations for this subject.")
    st.stop()

question_labels = [f"Item {j+1}" for j in range(max_idx + 1)]
default = st.session_state.get("idx", 0)
selected = st.sidebar.selectbox("Go to item", question_labels, index=default)
st.session_state.idx = question_labels.index(selected)
i = st.session_state.idx

# --- Browser TTS controls helper ---
def inject_tts(text: str, key: str, label: str):
    safe = text.replace("\\", "\\\\").replace("`", "'").replace("\n", "\\n")
    components.html(f"""
<div style="margin:8px 0;"><strong>{label}</strong><br>
  <button id="{key}_play">▶️ Play</button>
  <button id="{key}_pause" disabled>⏸️ Pause</button>
  <button id="{key}_resume" disabled>⏯️ Resume</button>
  <button id="{key}_stop" disabled>⏹️ Stop</button>
</div>
<script>
  const paras = `{safe}`.split(/\\n\\s*\\n/);
  const utterances = paras.map(p => {{
    const u = new SpeechSynthesisUtterance(p);
    u.rate = 0.6;
    return u;
  }});
  function pickVoice() {{
    const vs = speechSynthesis.getVoices();
    return vs.find(v => /samantha|victoria|zira|female/i.test(v.name))
        || vs.find(v => v.lang.startsWith("en"));
  }}
  function setupVoices() {{
    const voice = pickVoice();
    if (voice) utterances.forEach(u => u.voice = voice);
  }}
  if (speechSynthesis.getVoices().length) setupVoices();
  else speechSynthesis.onvoiceschanged = setupVoices;

  let idxUt = 0;
  const playBtn = document.getElementById("{key}_play");
  const pauseBtn = document.getElementById("{key}_pause");
  const resumeBtn = document.getElementById("{key}_resume");
  const stopBtn = document.getElementById("{key}_stop");

  function speakNext() {{
    if (idxUt >= utterances.length) return finish();
    const u = utterances[idxUt++];
    u.onend = () => setTimeout(speakNext, 1000);
    speechSynthesis.speak(u);
  }}
  function start() {{
    speechSynthesis.cancel();
    idxUt = 0;
    speakNext();
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    stopBtn.disabled = false;
  }}
  function finish() {{
    playBtn.disabled = false;
    pauseBtn.disabled = true;
    resumeBtn.disabled = true;
    stopBtn.disabled = true;
  }}

  playBtn.onclick = start;
  pauseBtn.onclick = () => {{ speechSynthesis.pause(); pauseBtn.disabled = true; resumeBtn.disabled = false; }};
  resumeBtn.onclick = () => {{ speechSynthesis.resume(); resumeBtn.disabled = true; pauseBtn.disabled = false; }};
  stopBtn.onclick = () => {{ speechSynthesis.cancel(); finish(); }};
  utterances[utterances.length - 1].onend = finish;
</script>
""", height=140)

# --- Render the selected explanation ---
row = subset.iloc[i]
exp_text = str(row["Explanation"]).strip()

st.title(f"{subject} — Item {i+1} of {max_idx+1}")

st.markdown(f'<div class="explanation-block">{exp_text.replace("\\n", "<br><br>")}</div>', unsafe_allow_html=True)

# TTS control
inject_tts(exp_text, f"exp_{i}", "🔊 Read Explanation")

# --- Navigation buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx = i - 1
        st.experimental_rerun()
with col2:
    if st.button("Next ▶") and i < max_idx:
        st.session_state.idx = i + 1
        st.experimental_rerun()
