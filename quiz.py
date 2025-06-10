import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

# URL of the Excel file
URL = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"

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

def browser_tts(text: str, label: str, key: str):
    safe = text.replace("`", "'").replace("\\", "\\\\")
    html = f"""
    <button id="{key}">{label}</button>
    <script>
      document.getElementById("{key}").onclick = () => {{
        const msg = new SpeechSynthesisUtterance(`{safe}`);
        window.speechSynthesis.speak(msg);
      }};
    </script>
    """
    components.html(html, height=60)

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Test Explanations with Browser TTS")

    df = load_data()
    if df.empty:
        return

    st.sidebar.title("Quiz Navigation")
    # 1) Subject picker
    subjects = df['Subject'].unique()
    subject = st.sidebar.selectbox("Subject", subjects)
    filtered = df[df['Subject'] == subject]

    # 2) Optional Topic picker
    if 'Topic' in filtered.columns:
        topics = filtered['Topic'].unique()
        topic = st.sidebar.selectbox("Topic", topics)
        filtered = filtered[filtered['Topic'] == topic]
    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        st.sidebar.warning("No items for this selection.")
        return

    # Manage index
    key = (subject, filtered.index.name if 'Topic' not in df.columns else (subject, topic))
    if 'idx' not in st.session_state or st.session_state.get('last') != key:
        st.session_state.idx = 0
        st.session_state.last = key

    idx = st.session_state.idx
    max_idx = len(filtered) - 1

    row = filtered.iloc[idx]
    explanation = str(row['Explanation'])
    formatted = format_html(explanation)

    # Main display
    st.subheader(f"{subject} (Item {idx+1} of {max_idx+1})")
    st.markdown(formatted, unsafe_allow_html=True)
    browser_tts(explanation, "🔊 Play Explanation", f"tts_main_{idx}")

    # Sidebar TTS
    st.sidebar.markdown("### 🔊 Audio")
    browser_tts(explanation, "▶ Play (Sidebar)", f"tts_side_{idx}")

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
