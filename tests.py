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

# --- Sidebar navigation ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
sub = df[df["Subject"] == subject]

if "Topic" in sub.columns:
    topic = st.sidebar.selectbox("Topic", sub["Topic"].unique())
    sub = sub[sub["Topic"] == topic].reset_index(drop=True)
else:
    sub = sub.reset_index(drop=True)

# --- Session index ---
if "idx" not in st.session_state:
    st.session_state.idx = 0
max_idx = len(sub) - 1
if max_idx < 0:
    st.sidebar.warning("No items here.")
    st.stop()
st.session_state.idx = min(max(st.session_state.idx, 0), max_idx)
i = st.session_state.idx

# --- Helper: HTML paragraph formatting ---
def format_html(text: str) -> str:
    paras = re.split(r"\n\s*\n", text.strip())
    return "".join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

# --- Helper: Inject browser TTS controls with explicit female voice ---
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
  // Split into paragraphs
  const paras = `{safe}`.split(/\\n\\s*\\n/);

  // Build utterances with 80% speed
  const utterances = paras.map(p => {{
    const u = new SpeechSynthesisUtterance(p);
    u.rate = 0.8;
    return u;
  }});

  // Explicitly pick the first female English voice available
  function pickFemaleVoice() {{
    const vs = window.speechSynthesis.getVoices();
    // common female voice names
    const female = vs.find(v =>
      v.lang.startsWith('en') &&
      /samantha|victoria|zira|female/i.test(v.name)
    );
    return female || vs.find(v => v.lang.startsWith('en'));
  }}

  function setupVoices() {{
    const voice = pickFemaleVoice();
    if (voice) utterances.forEach(u => u.voice = voice);
  }}

  if (speechSynthesis.getVoices().length) {{
    setupVoices();
  }} else {{
    speechSynthesis.onvoiceschanged = setupVoices;
  }}

  let idx = 0;
  const playBtn = document.getElementById("{key}_play");
  const pauseBtn = document.getElementById("{key}_pause");
  const resumeBtn = document.getElementById("{key}_resume");
  const stopBtn = document.getElementById("{key}_stop");

  function speakNext() {{
    if (idx >= utterances.length) return finish();
    const u = utterances[idx++];
    u.onend = () => setTimeout(speakNext, 600);
    window.speechSynthesis.speak(u);
  }}

  function start() {{
    window.speechSynthesis.cancel();
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
    window.speechSynthesis.pause();
    pauseBtn.disabled = true;
    resumeBtn.disabled = false;
  }};
  resumeBtn.onclick = () => {{
    window.speechSynthesis.resume();
    resumeBtn.disabled = true;
    pauseBtn.disabled = false;
  }};
  stopBtn.onclick = () => {{
    window.speechSynthesis.cancel();
    finish();
  }};

  // When done naturally
  utterances[utterances.length - 1].onend = finish;
</script>
""", height=140)

# --- Render page ---
row = sub.iloc[i]
text = str(row.get("Explanation", "") or "").strip()

st.title(f"{subject} ({i+1} of {max_idx+1})")

# Top TTS controls
inject_tts(text, f"top_{i}", "Read Explanation")

# Explanation display
st.markdown(format_html(text), unsafe_allow_html=True)

# Sidebar TTS controls
st.sidebar.markdown("### 🔊 Audio Controls")
inject_tts(text, f"side_{i}", "Read Explanation")

# Navigation
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶") and i < max_idx:
        st.session_state.idx += 1
