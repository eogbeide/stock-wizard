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
  .explanation-block p {
    margin-bottom: 1em;
  }
</style>
""", height=0)

# --- Sidebar: Subject selector ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
subset = df[df["Subject"] == subject].reset_index(drop=True)

# --- Sidebar: Item selector ---
count = len(subset)
if count == 0:
    st.sidebar.warning("No explanations for this subject.")
    st.stop()

labels = [f"Item {i+1}" for i in range(count)]
default = st.session_state.get("idx", 0)
sel = st.sidebar.selectbox("Go to item", labels, index=default)
st.session_state.idx = labels.index(sel)
i = st.session_state.idx

# --- TTS controls injection ---
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
    let u = new SpeechSynthesisUtterance(p);
    u.rate = 0.6;
    return u;
  }});
  function pickVoice() {{
    const vs = speechSynthesis.getVoices();
    return vs.find(v => /samantha|victoria|zira|female/i.test(v.name))
        || vs.find(v => v.lang.startsWith("en"));
  }}
  function setup() {{
    const v = pickVoice();
    if (v) utterances.forEach(u => u.voice = v);
  }}
  if (speechSynthesis.getVoices().length) setup();
  else speechSynthesis.onvoiceschanged = setup;

  let idxUt=0;
  const playBtn = document.getElementById("{key}_play");
  const pauseBtn = document.getElementById("{key}_pause");
  const resumeBtn = document.getElementById("{key}_resume");
  const stopBtn = document.getElementById("{key}_stop");

  function speakNext() {{
    if (idxUt >= utterances.length) return finish();
    let u = utterances[idxUt++];
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

# --- Render selected item ---
row = subset.iloc[i]
exp = str(row["Explanation"]).strip()

st.title(f"{subject} — Item {i+1} of {count}")

# Explanation block
html_exp = "".join(f"<p>{p}</p>" for p in exp.split("\n\n"))
st.markdown(f'<div class="explanation-block">{html_exp}</div>', unsafe_allow_html=True)

# Sidebar TTS
st.sidebar.markdown("### 🔊 Audio Controls")
inject_tts(exp, f"exp{i}", "Read Explanation")

# --- Navigation buttons ---
c1, c2 = st.columns(2)
with c1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx = i - 1
        st.experimental_rerun()
with c2:
    if st.button("Next ▶") and i < count-1:
        st.session_state.idx = i + 1
        st.experimental_rerun()
