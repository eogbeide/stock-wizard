import streamlit as stsg
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

def browser_tts(text: str, button_label: str, key: str):
    """Render a JS button that speaks `text` via the browser SpeechSynthesis API."""
    # Escape backticks
    safe = text.replace("`", "'").replace("\\", "\\\\")
    html = f"""
    <button id="{key}">{button_label}</button>
    <script>
      const btn = document.getElementById("{key}");
      btn.addEventListener("click", () => {{
        const msg = new SpeechSynthesisUtterance(`{safe}`);
        window.speechSynthesis.speak(msg);
      }});
    </script>
    """
    # Height just enough for a button
    components.html(html, height=60)

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Test Explanations with Browser TTS")

    df = load_data()
    if df.empty:
        return

    st.sidebar.title("Quiz Navigation")
    subjects = df['Subject'].unique()
    subject = st.sidebar.selectbox("Subject", subjects)
    filtered = df[df['Subject'] == subject]

    topics = filtered['Topic'].unique()
    topic = st.sidebar.selectbox("Topic", topics)
    filtered = filtered[filtered['Topic'] == topic].reset_index(drop=True)
    if filtered.empty:
        st.sidebar.warning("No questions here.")
        return

    # Manage index in session state
    if 'idx' not in st.session_state or st.session_state.get('last') != (subject, topic):
        st.session_state.idx = 0
        st.session_state.last = (subject, topic)

    idx = st.session_state.idx
    max_idx = len(filtered) - 1

    row = filtered.iloc[idx]
    explanation = str(row['Explanation'])
    formatted = format_html(explanation)

    # Top-level TTS button
    st.markdown(f"### {subject} — Explanation {idx+1} of {max_idx+1}")
    st.markdown(formatted, unsafe_allow_html=True)
    browser_tts(explanation, "🔊 Play Explanation", f"tts_top_{idx}")

    # Sidebar TTS button
    st.sidebar.markdown("### 🔊 Play Explanation")
    browser_tts(explanation, "▶ Play (Sidebar)", f"tts_sb_{idx}")

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
