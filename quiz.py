import streamlit as st
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
        st.sidebar.error(f"Error loading data: {e}")
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

# --- Function to format paragraphs ---
def format_html(text: str) -> str:
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

# --- Inject JS TTS controls ---
def inject_tts(text: str, key: str, label: str):
    """
    Renders Play/Pause/Resume/Stop controls that read `text`
    at 70% speed with a soft female voice.
    """
    safe = text.replace("\\","\\\\").replace("`","'").replace("\n","\\n")
    components.html(f'''
<div style="margin:10px 0;"><strong>{label}</strong><br>
  <button id="{key}_play">▶️ Play</button>
  <button id="{key}_pause" disabled>⏸️ Pause</button>
  <button id="{key}_resume" disabled>⏯️ Resume</button>
  <button id="{key}_stop" disabled>⏹️ Stop</button>
</div>
<script>
  const paras = `{safe}`.split(/\\n\\s*\\n/);
  const utter = paras.map(p => {{
    const u = new SpeechSynthesisUtterance(p);
    u.rate = 0.7;
    return u;
  }});
  function pickVoice() {{
    const vs = speechSynthesis.getVoices();
    return vs.find(v => /female|zira|samantha|victoria/i.test(v.name))
        || vs.find(v => v.lang.startsWith('en'));
  }}
  function setup() {{
    const v = pickVoice();
    if(v) utter.forEach(u=>u.voice=v);
  }}
  if(speechSynthesis.getVoices().length) setup();
  else speechSynthesis.onvoiceschanged = setup;

  let idx=0;
  const play = document.getElementById("{key}_play");
  const pause = document.getElementById("{key}_pause");
  const resume = document.getElementById("{key}_resume");
  const stop = document.getElementById("{key}_stop");

  function speakNext() {{
    if(idx>=utter.length) return finish();
    const u=utter[idx++];
    u.onend = ()=> setTimeout(speakNext,600);
    speechSynthesis.speak(u);
  }}
  function start() {{
    speechSynthesis.cancel();
    idx=0;
    speakNext();
    play.disabled=true;
    pause.disabled=false;
    stop.disabled=false;
  }}
  function finish() {{
    play.disabled=false;
    pause.disabled=true;
    resume.disabled=true;
    stop.disabled=true;
  }}
  play.onclick=start;
  pause.onclick=()=>{{ speechSynthesis.pause(); pause.disabled=true; resume.disabled=false; }};
  resume.onclick=()=>{{ speechSynthesis.resume(); resume.disabled=true; pause.disabled=false; }};
  stop.onclick=()=>{{ speechSynthesis.cancel(); finish(); }};
  utter[utter.length-1].onend=finish;
</script>
''', height=120)

# --- Render Top Controls ---
st.markdown("### 🔊 Audio Controls (Top)")
row = filtered.iloc[i]
passage = str(row['Passage']).strip()
inject_tts(passage, f"top_passage_{i}", "Read Passage")

# --- Render Passage with spaced paragraphs ---
st.markdown(f"<div>{format_html(passage)}</div>", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown("### 🔊 Audio Controls (Sidebar)")
inject_tts(passage, f"side_passage_{i}", "Read Passage")

# --- Navigation ---
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i>0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶") and i<max_idx:
        st.session_state.idx += 1
