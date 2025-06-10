import streamlit as stb
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
    st.sidebar.warning("No questions for this selection.")
    st.stop()
st.session_state.idx = max(0, min(st.session_state.idx, max_idx))
i = st.session_state.idx

# --- Style snippet ---
components.html("""
<style>
  .passage { padding: 0.5em; background: #f9f9f9; border-radius: 4px; }
  .question { margin-top: 1em; }
  .question strong { font-size: 1.1em; }
  .question ul { margin-top: 0.2em; padding-left: 1.2em; }
  .explanation { margin-top: 1em; color: #555; font-style: italic; }
</style>
""", height=0)

def format_html(text: str) -> str:
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n','<br>')}</p>" for p in paras)

def inject_tts_controls(text: str, key: str, label: str):
    """
    Inject Play/Pause/Resume/Stop controls using the browser SpeechSynthesis API.
    Soft female voice at 80% speed, paragraph pauses.
    """
    safe = text.replace("\\", "\\\\").replace("`", "'").replace("\n", "\\n")
    html = f'''
    <div class="audio-controls">
      <strong>{label}</strong><br>
      <button id="{key}_play">▶️ Play</button>
      <button id="{key}_pause" disabled>⏸️ Pause</button>
      <button id="{key}_resume" disabled>⏯️ Resume</button>
      <button id="{key}_stop" disabled>⏹️ Stop</button>
    </div>
    <script>
      const paras = `{safe}`.split(/\\n\\s*\\n/);
      const utter = paras.map(p => {{
        const u = new SpeechSynthesisUtterance(p);
        u.rate = 0.8;
        return u;
      }});
      function pickVoice() {{
        const vs = speechSynthesis.getVoices();
        return vs.find(v => /zira|samantha|victoria|female/i.test(v.name))
               || vs.find(v => v.lang.startsWith('en'));
      }}
      function setup() {{
        const v = pickVoice();
        if(v) utter.forEach(u => u.voice = v);
      }}
      if (speechSynthesis.getVoices().length) setup();
      else speechSynthesis.onvoiceschanged = setup;

      let idx = 0;
      const play = document.getElementById("{key}_play");
      const pause = document.getElementById("{key}_pause");
      const resume = document.getElementById("{key}_resume");
      const stop = document.getElementById("{key}_stop");

      function speakNext() {{
        if (idx >= utter.length) return finish();
        const u = utter[idx++];
        u.onend = () => setTimeout(speakNext, 600);
        speechSynthesis.speak(u);
      }}
      function start() {{
        speechSynthesis.cancel();
        idx = 0;
        speakNext();
        play.disabled = true;
        pause.disabled = false;
        stop.disabled = false;
      }}
      function finish() {{
        play.disabled = false;
        pause.disabled = true;
        resume.disabled = true;
        stop.disabled = true;
      }}

      play.onclick = start;
      pause.onclick = () => {{
        speechSynthesis.pause();
        pause.disabled = true;
        resume.disabled = false;
      }};
      resume.onclick = () => {{
        speechSynthesis.resume();
        resume.disabled = true;
        pause.disabled = false;
      }};
      stop.onclick = () => {{
        speechSynthesis.cancel();
        finish();
      }};
      utter[utter.length - 1].onend = finish;
    </script>
    '''
    components.html(html, height=100)

# --- Prepare texts ---
row = filtered.iloc[i]
passage = str(row['Passage']).strip()
answers = [a.strip() for a in str(row['Answer']).split(';')]
question = f"Question {i+1}: {row['Question']}"
qa = question + "\n" + "\n".join(f"- {a}" for a in answers)
explanation = str(row.get('Explanation','') or '').strip()
full = passage + "\\n\\n" + qa + (("\\n\\nExplanation:\\n" + explanation) if explanation else "")

# --- Top controls ---
st.markdown("### 🔊 Audio Controls (Top)")
inject_tts_controls(passage, f"top_passage_{i}", "Passage")
inject_tts_controls(full, f"top_full_{i}", "Q&A + Explanation")

# --- Content ---
st.markdown(f"## {subject} ({i+1} of {max_idx+1})")
st.markdown(f'<div class="passage">{format_html(passage)}</div>', unsafe_allow_html=True)
st.markdown('<div class="question"><strong>' + question + '</strong><ul>' +
            ''.join(f'<li>{a}</li>' for a in answers) +
            '</ul></div>', unsafe_allow_html=True)
if explanation:
    if st.checkbox("Show Explanation"):
        st.markdown('<div class="explanation">' + format_html(explanation) + '</div>', unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.markdown("### 🔊 Audio Controls (Sidebar)")
inject_tts_controls(passage, f"sb_passage_{i}", "Passage")
inject_tts_controls(full, f"sb_full_{i}", "Q&A + Explanation")

# --- Navigation ---
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶") and i < max_idx:
        st.session_state.idx += 1
