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

# --- Sidebar: Subject & Topic ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
subset = df[df["Subject"] == subject]

topic = st.sidebar.selectbox("Topic", subset["Topic"].unique())
subset = subset[subset["Topic"] == topic].reset_index(drop=True)

# --- Sidebar: Question dropdown ---
max_idx = len(subset) - 1
if max_idx < 0:
    st.sidebar.warning("No questions for this selection.")
    st.stop()

# Build list of question labels
question_labels = [f"Question {j+1}" for j in range(max_idx + 1)]
# Default index from session or 0
default_q = st.session_state.get("idx", 0)
# Dropdown in sidebar
selected = st.sidebar.selectbox("Go to question", question_labels, index=default_q)
# Sync session_state
st.session_state.idx = question_labels.index(selected)
i = st.session_state.idx

# --- Helper: format HTML paragraphs ---
def format_html(text: str) -> str:
    paras = re.split(r"\n\s*\n", text.strip())
    return "".join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

# --- Helper: inject browser TTS controls ---
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
    u.rate = 0.6;  // slower
    return u;
  }});
  function pickVoice() {{
    const vs = speechSynthesis.getVoices();
    return vs.find(v => /samantha|victoria|zira|female/i.test(v.name))
        || vs.find(v => v.lang.startsWith("en"));
  }}
  function setup() {{
    const v = pickVoice();
    if(v) utterances.forEach(u => u.voice = v);
  }}
  if (speechSynthesis.getVoices().length) setup();
  else speechSynthesis.onvoiceschanged = setup;

  let idx = 0;
  const play = document.getElementById("{key}_play");
  const pause = document.getElementById("{key}_pause");
  const resume = document.getElementById("{key}_resume");
  const stop = document.getElementById("{key}_stop");

  function speakNext() {{
    if(idx>=utterances.length) return finish();
    const u = utterances[idx++];
    u.onend = () => setTimeout(speakNext, 1000);  // 1s pause
    speechSynthesis.speak(u);
  }}
  function start() {{
    speechSynthesis.cancel();
    idx=0;
    speakNext();
    play.disabled=true; pause.disabled=false; stop.disabled=false;
  }}
  function finish() {{
    play.disabled=false; pause.disabled=true; resume.disabled=true; stop.disabled=true;
  }}
  play.onclick=start;
  pause.onclick=()=>{{ speechSynthesis.pause(); pause.disabled=true; resume.disabled=false; }};
  resume.onclick=()=>{{ speechSynthesis.resume(); resume.disabled=true; pause.disabled=false; }};
  stop.onclick=()=>{{ speechSynthesis.cancel(); finish(); }};
  utterances[utterances.length-1].onend = finish;
</script>
""", height=140)

# --- Render selected question ---
row = subset.iloc[i]
passage = str(row["Passage"]).strip()
answers = [a.strip() for a in str(row["Answer"]).split(";")]
question_text = f"Question {i+1}: {row['Question']}"
explanation = str(row.get("Explanation","") or "").strip()

st.title(f"{subject} — {question_labels[i]} ({i+1}/{max_idx+1})")

# Passage display & TTS
st.markdown(f'<div class="passage">{format_html(passage)}</div>', unsafe_allow_html=True)
inject_tts(passage, f"passage_{i}", "🔊 Read Passage")

# Q&A display
st.markdown(f'<div class="question">{question_text}</div>', unsafe_allow_html=True)
st.markdown(f'<ul class="options">{"".join(f"<li>{opt}</li>" for opt in answers)}</ul>', unsafe_allow_html=True)

# Explanation display & TTS
if explanation:
    if st.checkbox("Show Explanation"):
        st.markdown(f'<div class="explanation">{format_html(explanation)}</div>', unsafe_allow_html=True)
    inject_tts(explanation, f"exp_{i}", "🔊 Read Explanation")

# Next/Prev buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i>0:
        st.session_state.idx = i-1
        st.experimental_rerun()
with col2:
    if st.button("Next ▶") and i<max_idx:
        st.session_state.idx = i+1
        st.experimental_rerun()
