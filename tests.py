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

df = load_data()
if df.empty:
    st.sidebar.warning("No quiz data available.")
    st.stop()

# --- Sidebar navigation ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
subset = df[df["Subject"] == subject]

if "Topic" in subset.columns:
    topic = st.sidebar.selectbox("Topic", subset["Topic"].unique())
    subset = subset[subset["Topic"] == topic].reset_index(drop=True)
else:
    subset = subset.reset_index(drop=True)

# --- Session state for index ---
if "idx" not in st.session_state:
    st.session_state.idx = 0
max_idx = len(subset) - 1
if max_idx < 0:
    st.sidebar.warning("No items here.")
    st.stop()
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))
i = st.session_state.idx

# --- HTML paragraph formatting ---
def format_html(text: str) -> str:
    paras = re.split(r"\n\s*\n", text.strip())
    return "".join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

# --- Inject browser TTS controls ---
def inject_tts(text: str, key: str, label: str):
    safe = text.replace("\\","\\\\").replace("`","'").replace("\n","\\n")
    components.html(f"""
<div style="margin:8px 0;"><strong>{label}</strong><br>
  <button id="{key}_play">▶️ Play</button>
  <button id="{key}_pause" disabled>⏸️ Pause</button>
  <button id="{key}_resume" disabled>⏯️ Resume</button>
  <button id="{key}_stop" disabled>⏹️ Stop</button>
</div>
<script>
  const paras = `{safe}`.split(/\\n\\s*\\n/);
  const utterances = paras.map(p=> {{
    let u = new SpeechSynthesisUtterance(p);
    u.rate = 0.8;
    return u;
  }});
  function pickVoice() {{
    let vs = speechSynthesis.getVoices();
    return vs.find(v=>/samantha|victoria|zira|female/i.test(v.name))
      || vs.find(v=>v.lang.startsWith("en"));
  }}
  function setup() {{
    let v = pickVoice();
    if(v) utterances.forEach(u=>u.voice=v);
  }}
  if(speechSynthesis.getVoices().length) setup();
  else speechSynthesis.onvoiceschanged = setup;

  let idx=0;
  const playBtn = document.getElementById("{key}_play");
  const pauseBtn = document.getElementById("{key}_pause");
  const resumeBtn = document.getElementById("{key}_resume");
  const stopBtn = document.getElementById("{key}_stop");

  function speakNext() {{
    if(idx>=utterances.length) return finish();
    let u=utterances[idx++];
    u.onend = ()=>setTimeout(speakNext,600);
    speechSynthesis.speak(u);
  }}
  function start(){{
    speechSynthesis.cancel();
    idx=0;
    speakNext();
    playBtn.disabled=true; pauseBtn.disabled=false; stopBtn.disabled=false;
  }}
  function finish(){{
    playBtn.disabled=false; pauseBtn.disabled=true; resumeBtn.disabled=true; stopBtn.disabled=true;
  }}
  playBtn.onclick=start;
  pauseBtn.onclick=()=>{{ speechSynthesis.pause(); pauseBtn.disabled=true; resumeBtn.disabled=false; }};
  resumeBtn.onclick=()=>{{ speechSynthesis.resume(); resumeBtn.disabled=true; pauseBtn.disabled=false; }};
  stopBtn.onclick=()=>{{ speechSynthesis.cancel(); finish(); }};
  utterances[utterances.length-1].onend=finish;
</script>
""", height=120)

# --- Render page ---
row = subset.iloc[i]
text = str(row.get("Passage", "")).strip()

st.title(f"{subject} ({i+1}/{max_idx+1})")

# Top controls
inject_tts(text, f"top_{i}", "Read Passage")

# Passage content
st.markdown(format_html(text), unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("### 🔊 Audio Controls")
inject_tts(text, f"side_{i}", "Read Passage")

# Navigation
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶") and i < max_idx:
        st.session_state.idx += 1
