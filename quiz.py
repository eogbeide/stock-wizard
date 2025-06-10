import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_html(text: str) -> str:
    # Wrap paragraphs and line breaks in HTML
    paras = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n', '<br>')}</p>" for p in paras)

def inject_tts_controls(text: str, key: str):
    """Inject JS TTS controls using the browser SpeechSynthesis API."""
    safe = text.replace("\\", "\\\\").replace("`", "'").replace("\n", "\\n")
    html = f'''
    <div style="margin: 0.5em 0;">
      <button id="{key}_play">▶️ Play</button>
      <button id="{key}_pause" disabled>⏸️ Pause</button>
      <button id="{key}_resume" disabled>⏯️ Resume</button>
      <button id="{key}_stop" disabled>⏹️ Stop</button>
    </div>
    <script>
      // Split into paragraphs
      const paras = `{safe}`.split(/\\n\\s*\\n/);
      const utter = paras.map(p => {{
        const u = new SpeechSynthesisUtterance(p);
        u.rate = 0.8;
        return u;
      }});
      // pick soft female voice
      function pickVoice() {{
        const vs = speechSynthesis.getVoices();
        return vs.find(v => /female|zira|samantha|victoria/i.test(v.name)) 
               || vs.find(v => v.lang.startsWith('en'));
      }}
      function setup() {{
        const v = pickVoice();
        if(v) utter.forEach(u => u.voice = v);
      }}
      if(speechSynthesis.getVoices().length) setup();
      else speechSynthesis.onvoiceschanged = setup;

      let idx = 0;
      const play = document.getElementById("{key}_play");
      const pause = document.getElementById("{key}_pause");
      const resume = document.getElementById("{key}_resume");
      const stop = document.getElementById("{key}_stop");

      function speakNext() {{
        if(idx >= utter.length) return endAll();
        const u = utter[idx++];
        u.onend = () => setTimeout(speakNext, 600);
        speechSynthesis.speak(u);
      }}
      function startAll() {{
        speechSynthesis.cancel();
        idx = 0;
        speakNext();
        play.disabled = true;
        pause.disabled = false;
        stop.disabled = false;
      }}
      function endAll() {{
        play.disabled = false;
        pause.disabled = true;
        resume.disabled = true;
        stop.disabled = true;
      }}

      play.onclick = startAll;
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
        endAll();
      }};
      utter[utter.length-1].onend = endAll;
    </script>
    '''
    components.html(html, height=80)

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Mobile-Safe Story-Style TTS")

    df = load_data()
    if df.empty:
        st.stop()

    st.sidebar.title("Quiz Navigation")
    subjects = df['Subject'].unique()
    subj = st.sidebar.selectbox("Subject", subjects)
    data = df[df['Subject']==subj]

    if 'Topic' in data.columns:
        topics = data['Topic'].unique()
        top = st.sidebar.selectbox("Topic", topics)
        data = data[data['Topic']==top]

    data = data.reset_index(drop=True)
    if data.empty:
        st.sidebar.warning("No items here.")
        st.stop()

    key_state = (subj, top) if 'Topic' in df.columns else (subj, None)
    if 'idx' not in st.session_state or st.session_state.last!=key_state:
        st.session_state.idx = 0
        st.session_state.last = key_state

    i = st.session_state.idx
    end = len(data)-1
    row = data.iloc[i]

    # Prepare texts
    passage = str(row['Passage'])
    qa = f"Question {i+1}: {row['Question']}. " + "; ".join(str(a).strip() for a in row['Answer'].split(';'))
    exp = row.get('Explanation','') or ''
    full = passage + "\\n\\n" + qa + ("\\n\\nExplanation: "+exp if exp else "")

    # Controls above
    inject_tts_controls(full, f"above_{i}")

    st.subheader(f"{subj} ({i+1} of {end+1})")
    st.markdown(format_html(passage), unsafe_allow_html=True)
    st.markdown(f"**Q&A:** {qa}")
    if exp:
        if st.checkbox("Show Explanation"):
            st.markdown(format_html(exp), unsafe_allow_html=True)

    # Controls below
    inject_tts_controls(full, f"below_{i}")

    # Sidebar controls
    st.sidebar.markdown("### 🔊 Audio")
    inject_tts_controls(full, f"side_{i}")

    # Navigation
    c1,c2 = st.columns(2)
    with c1:
        if st.button("◀ Back") and i>0:
            st.session_state.idx -=1
    with c2:
        if st.button("Next ▶") and i<end:
            st.session_state.idx +=1

if __name__=="__main__":
    main()
