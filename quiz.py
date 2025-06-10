import streamlit as sts
import pandas as pd
import re
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
    st.sidebar.warning("No questions here.")
    st.stop()
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))
i = st.session_state.idx

# --- Helper to format HTML paragraphs ---
def format_html(text: str) -> str:
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

# --- Inject JS TTS controls ---
def inject_tts(text: str, key: str, label: str):
    """
    Renders Play/Pause/Resume/Stop controls that speak `text`
    paragraph-by-paragraph at 70% speed with a soft female voice.
    """
    safe = text.replace("\\","\\\\").replace("`","'").replace("\n","\\n")
    components.html(f"""
<div style="margin:8px 0;">
  <strong>{label}</strong><br>
  <button id="{key}_play">▶️ Play</button>
  <button id="{key}_pause" disabled>⏸️ Pause</button>
  <button id="{key}_resume" disabled>⏯️ Resume</button>
  <button id="{key}_stop" disabled>⏹️ Stop</button>
</div>
<script>
  const text = `{safe}`;
  const paras = text.split(/\\n\\s*\\n/);
  const utterances = paras.map(p => {{
    const u = new SpeechSynthesisUtterance(p);
    u.rate = 0.7;  // slower
    return u;
  }});
  function pickVoice() {{
    const voices = speechSynthesis.getVoices();
    return voices.find(v => /zira|samantha|victoria|female/i.test(v.name))
        || voices.find(v => v.lang.startsWith('en'));
  }}
  function setup() {{
    const v = pickVoice();
    if (v) utterances.forEach(u => u.voice = v);
  }}
  if (speechSynthesis.getVoices().length) setup();
  else speechSynthesis.onvoiceschanged = setup;

  let idx = 0;
  const playBtn = document.getElementById("{key}_play");
  const pauseBtn = document.getElementById("{key}_pause");
  const resumeBtn = document.getElementById("{key}_resume");
  const stopBtn = document.getElementById("{key}_stop");

  function speakNext() {{
    if (idx >= utterances.length) return finish();
    const u = utterances[idx++];
    u.onend = () => setTimeout(speakNext, 600);
    speechSynthesis.speak(u);
  }}
  function start() {{
    speechSynthesis.cancel();
    idx = 0;
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
  pauseBtn.onclick = () => {{
    speechSynthesis.pause();
    pauseBtn.disabled = true;
    resumeBtn.disabled = false;
  }};
  resumeBtn.onclick = () => {{
    speechSynthesis.resume();
    resumeBtn.disabled = true;
    pauseBtn.disabled = false;
  }};
  stopBtn.onclick = () => {{
    speechSynthesis.cancel();
    finish();
  }};
  utterances[utterances.length - 1].onend = finish;
</script>
""", height=100)

# --- Display current item ---
row = filtered.iloc[i]
passage = str(row['Passage']).strip()

# Top controls
st.markdown("### 🔊 Audio Controls (Top)")
inject_tts(passage, f"top_{i}", "Passage")

# Passage
st.markdown(f"## {subject} ({i+1} of {max_idx+1})")
st.markdown(format_html(passage), unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("### 🔊 Audio Controls (Sidebar)")
inject_tts(passage, f"side_{i}", "Passage")

# Navigation
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶") and i < max_idx:
        st.session_state.idx += 1
