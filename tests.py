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
    """Wrap paragraphs and line breaks in HTML."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return ''.join(f"<p>{p.replace('\n', '<br>')}</p>" for p in paragraphs)

def browser_tts_controls(text: str, key_prefix: str):
    """
    Inject HTML+JS to create Play/Pause/Resume/Stop buttons that speak `text`
    via the Web Speech API with full control.
    """
    safe = text.replace("`", "'").replace("\\", "\\\\")
    html = f"""
    <div id="{key_prefix}_controls">
      <button id="{key_prefix}_play">▶️ Play</button>
      <button id="{key_prefix}_pause" disabled>⏸️ Pause</button>
      <button id="{key_prefix}_resume" disabled>⏯️ Resume</button>
      <button id="{key_prefix}_stop" disabled>⏹️ Stop</button>
    </div>
    <script>
      const utter_{key_prefix} = new SpeechSynthesisUtterance(`{safe}`);
      utter_{key_prefix}.rate = 1; // normal speed
      // pick the first default english voice
      utter_{key_prefix}.voice = window.speechSynthesis.getVoices()
        .find(v=>v.lang.startsWith('en')) || null;

      const playBtn = document.getElementById("{key_prefix}_play");
      const pauseBtn = document.getElementById("{key_prefix}_pause");
      const resumeBtn = document.getElementById("{key_prefix}_resume");
      const stopBtn = document.getElementById("{key_prefix}_stop");

      playBtn.onclick = () => {{
        window.speechSynthesis.cancel(); // reset any ongoing
        window.speechSynthesis.speak(utter_{key_prefix});
        playBtn.disabled = true;
        pauseBtn.disabled = false;
        stopBtn.disabled = false;
      }};
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
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
        stopBtn.disabled = true;
      }};
      // Re-enable Play when speech ends naturally
      utter_{key_prefix}.onend = () => {{
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        resumeBtn.disabled = true;
        stopBtn.disabled = true;
      }};
      // Load voices asynchronously
      window.speechSynthesis.onvoiceschanged = () => {{
        const v = window.speechSynthesis.getVoices()
                  .find(v=>v.lang.startsWith('en'));
        if(v) utter_{key_prefix}.voice = v;
      }};
    </script>
    """
    components.html(html, height=50)

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Test Explanations with Full Audio Controls")

    df = load_data()
    if df.empty:
        return

    st.sidebar.title("Quiz Navigation")
    subjects = df['Subject'].unique()
    subject = st.sidebar.selectbox("Subject", subjects)
    filtered = df[df['Subject'] == subject]

    # Handle optional Topic column
    if 'Topic' in filtered.columns:
        topics = filtered['Topic'].unique()
        topic = st.sidebar.selectbox("Topic", topics)
        filtered = filtered[filtered['Topic'] == topic]

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        st.sidebar.warning("No items here.")
        return

    # Index management
    key = (subject, topic) if 'Topic' in df.columns else (subject, None)
    if 'idx' not in st.session_state or st.session_state.get('last') != key:
        st.session_state.idx = 0
        st.session_state.last = key

    idx = st.session_state.idx
    max_idx = len(filtered) - 1

    row = filtered.iloc[idx]
    explanation = str(row['Explanation'])
    formatted = format_html(explanation)

    # Main content
    st.subheader(f"{subject} ({idx+1} of {max_idx+1})")
    st.markdown(formatted, unsafe_allow_html=True)
    browser_tts_controls(explanation, f"main_{idx}")

    # Sidebar controls
    st.sidebar.markdown("### 🔊 Audio Controls")
    browser_tts_controls(explanation, f"side_{idx}")

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back") and idx > 0:
            st.session_state.idx -= 1
    with col2:
        if st.button("Next ▶") and idx < max_idx:
            st.session_state.idx += 1

if __name__ == "__main__":
    main()
