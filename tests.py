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
    """Renders a button that speaks text in a male slower voice via the Web Speech API."""
    safe = text.replace("`", "'").replace("\\", "\\\\")
    html = f"""
    <button id="{key}">{label}</button>
    <script>
      const btn = document.getElementById("{key}");
      btn.addEventListener("click", () => {{
        // Wait for voices to be loaded
        let attempt = 0;
        function speak() {{
          const msg = new SpeechSynthesisUtterance(`{safe}`);
          msg.rate = 0.8;  // slower
          const voices = window.speechSynthesis.getVoices();
          // pick a male English voice if available
          const male = voices.find(v => 
            v.lang.startsWith('en') 
            && /male/i.test(v.name)
          ) || voices.find(v => v.lang.startsWith('en'));
          if (male) msg.voice = male;
          window.speechSynthesis.speak(msg);
        }}
        if (window.speechSynthesis.getVoices().length > 0) {{
          speak();
        }} else {{
          window.speechSynthesis.onvoiceschanged = () => {{
            if (attempt++ < 5) speak();
          }};
        }}
      }});
    </script>
    """
    components.html(html, height=60)

def main():
    st.set_page_config(layout="centered")
    st.title("📘 Test Explanations with Male Slower Voice")

    df = load_data()
    if df.empty:
        return

    st.sidebar.title("Quiz Navigation")
    subjects = df['Subject'].unique()
    subject = st.sidebar.selectbox("Subject", subjects)
    filtered = df[df['Subject'] == subject].copy()

    if 'Topic' in filtered.columns:
        topics = filtered['Topic'].unique()
        topic = st.sidebar.selectbox("Topic", topics)
        filtered = filtered[filtered['Topic'] == topic]

    filtered = filtered.reset_index(drop=True)
    if filtered.empty:
        st.sidebar.warning("No items here.")
        return

    if 'idx' not in st.session_state or st.session_state.get('last') != (subject, filtered.index.name if 'Topic' not in df.columns else (subject, topic)):
        st.session_state.idx = 0
        st.session_state.last = (subject, filtered.index.name if 'Topic' not in df.columns else (subject, topic))

    idx = st.session_state.idx
    max_idx = len(filtered) - 1

    row = filtered.iloc[idx]
    explanation = str(row['Explanation'])
    formatted = format_html(explanation)

    st.subheader(f"{subject} ({idx+1} of {max_idx+1})")
    st.markdown(formatted, unsafe_allow_html=True)
    browser_tts(explanation, "🔊 Play (Top)", f"tts_main_{idx}")

    st.sidebar.markdown("### 🔊 Audio")
    browser_tts(explanation, "▶ Play (Sidebar)", f"tts_side_{idx}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back") and idx > 0:
            st.session_state.idx -= 1
    with col2:
        if st.button("Next ▶") and idx < max_idx:
            st.session_state.idx += 1

if __name__ == "__main__":
    main()
