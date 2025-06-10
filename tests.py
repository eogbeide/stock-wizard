import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

# URL of the Excel file
URL = "https://github.com/eogbeide/stock-wizard/raw/main/tests.xlsx"

@st.cache_data
def load_data():
    try:
        return pd.read_excel(URL)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_html(text: str) -> str:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n', '<br>')}</p>" for p in paragraphs)

def browser_tts_controls(text: str, key_prefix: str):
    """
    Renders Play/Pause/Resume/Stop buttons that speak `text`
    paragraph-by-paragraph at 80% speed with a soft female voice.
    """
    # Escape backticks & linebreaks for JS
    safe = text.replace("\\", "\\\\").replace("`", "'").replace("\n", "\\n")
    html = f"""
    <div id="{key_prefix}_controls">
      <button id="{key_prefix}_play">▶️ Play</button>
      <button id="{key_prefix}_pause" disabled>⏸️ Pause</button>
      <button id="{key_prefix}_resume" disabled>⏯️ Resume</button>
      <button id="{key_prefix}_stop" disabled>⏹️ Stop</button>
    </div>
    <script>
      // Split into paragraphs on blank lines
      const paras = `{safe}`.split(/\\n\\s*\\n/);
      // Build utterances array
      const utterances = paras.map(p => {{
        const u = new SpeechSynthesisUtterance(p);
        u.rate = 0.8;
        return u;
      }});
      // Select a soft female English voice
      function selectVoice() {{
        const voices = window.speechSynthesis.getVoices();
        return voices.find(v =>
          v.lang.startsWith('en') &&
          /female|zira|samantha|victoria/i.test(v.name)
        ) || voices.find(v => v.lang.startsWith('en'));
      }}
      function setupVoices() {{
        const chosen = selectVoice();
        if(chosen) utterances.forEach(u => u.voice = chosen);
      }}
      if(window.speechSynthesis.getVoices().length) {{
        setupVoices();
      }} else {{
        window.speechSynthesis.onvoiceschanged = setupVoices;
      }}

      let current = 0;
      const playBtn = document.getElementById("{key_prefix}_play");
      const pauseBtn = document.getElementById("{key_prefix}_pause");
      const resumeBtn = document.getElementById("{key_prefix}_resume");
      const stopBtn = document.getElementById("{key_prefix}_stop");

      function speakNext() {{
        if(current >= utterances.length) return finishPlayback();
        const u = utterances[current++];
        u.onend = () => setTimeout(speakNext, 600);  // 600 ms pause
        window.speechSynthesis.speak(u);
      }}

      function startPlayback() {{
        window.speechSynthesis.cancel();
        current = 0;
        speakNext();
        playBtn.disabled = true;
        pauseBtn.disabled = false;
        stopBtn.disabled = false;
      }}
      function finishPlayback() {{
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
        stopBtn.disabled = true;
      }}

      playBtn.onclick = startPlayback;
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
        finishPlayback();
      }};

      // When all utterances finish naturally
      utterances[utterances.length - 1].onend = finishPlayback;
    </script>
    """
    components.html(html, height=80)

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Story-Style TTS Playback")

    df = load_data()
    if df.empty:
        return

    st.sidebar.title("Quiz Navigation")
    subjects = df['Subject'].unique()
    subject = st.sidebar.selectbox("Subject", subjects)
    filtered = df[df['Subject'] == subject]

    if 'Topic' in filtered.columns:
        topics = filtered['Topic'].unique()
        topic = st.sidebar.selectbox("Topic", topics)
        filtered = filtered[filtered['Topic'] == topic]

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        st.sidebar.warning("No items here.")
        return

    key = (subject, topic) if 'Topic' in df.columns else (subject, None)
    if 'idx' not in st.session_state or st.session_state.get('last') != key:
        st.session_state.idx = 0
        st.session_state.last = key

    idx = st.session_state.idx
    max_idx = len(filtered) - 1

    row = filtered.iloc[idx]
    explanation = str(row['Explanation'])
    st.subheader(f"{subject} ({idx+1} of {max_idx+1})")
    st.markdown(format_html(explanation), unsafe_allow_html=True)

    # Inline controls
    browser_tts_controls(explanation, f"main_{idx}")

    # Sidebar controls
    st.sidebar.markdown("### 🔊 Audio Controls")
    browser_tts_controls(explanation, f"side_{idx}")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back") and idx > 0:
            st.session_state.idx -= 1
    with col2:
        if st.button("Next ▶") and idx < max_idx:
            st.session_state.idx += 1

if __name__ == "__main__":
    main()
