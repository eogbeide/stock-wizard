import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile

# Load data from Excel on GitHub
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

quiz_data = load_data()

st.sidebar.title('Quiz Navigation')
if quiz_data.empty:
    st.warning("No quiz data available.")
    st.stop()

# Subject/topic selection
subjects = quiz_data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subjects)
filtered = quiz_data[quiz_data['Subject'] == selected_subject]

topics = filtered['Topic'].unique()
selected_topic = st.sidebar.selectbox('Select Topic', topics)
filtered = filtered[filtered['Topic'] == selected_topic].reset_index(drop=True)

# Ensure idx is within bounds
if 'idx' not in st.session_state:
    st.session_state.idx = 0
# Clamp idx to [0, len(filtered)-1]
max_idx = len(filtered) - 1
if max_idx < 0:
    st.warning("No questions for this Subject/Topic.")
    st.stop()
if st.session_state.idx > max_idx or st.session_state.idx < 0:
    st.session_state.idx = 0

def play_tts(text: str):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3')

def show_item(i: int):
    row = filtered.iloc[i]

    # Passage
    st.markdown("### Passage")
    st.markdown(row['Passage'].replace('\n', '<br><br>'), unsafe_allow_html=True)
    if st.button("🔊 Read Passage Aloud", key=f"tts_passage_{i}"):
        play_tts(row['Passage'])

    # Build Q&A text
    qa_text = (
        f"Question {i+1}: {row['Question']}\n"
        "Answers:\n" + "\n".join(f"- {opt.strip()}" for opt in row['Answer'].split(';'))
    )
    st.markdown(f"```text\n{qa_text}\n```")

    # Explanation (hidden until toggled)
    explanation = row.get('Explanation', '').strip()
    if explanation and st.checkbox("Show Explanation", key=f"show_exp_{i}"):
        st.info(explanation)

    # Combined Q&A + Explanation TTS
    full_tts_text = qa_text
    if explanation:
        full_tts_text += f"\nExplanation:\n{explanation}"
    if st.button("🔊 Read Q&A + Explanation Aloud", key=f"tts_full_{i}"):
        play_tts(full_tts_text)

# Navigation
col1, _, col2 = st.columns([1, 4, 1])
with col1:
    if st.button("◀️ Back") and st.session_state.idx > 0:
        st.session_state.idx -= 1
with col2:
    if st.button("Next ▶️") and st.session_state.idx < max_idx:
        st.session_state.idx += 1

# Finally show current item
show_item(st.session_state.idx)
