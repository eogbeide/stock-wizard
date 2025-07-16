import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from gtts import gTTS
import tempfile

# Cache the data loading
@st.cache_data
def load_data(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content))

DATA_URL = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Sentences.xls'

# Load and handle errors
try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Prepare list of Old Words
old_words = df["Old Word"].dropna().unique().tolist()
n = len(old_words)

# Initialize index in session state
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Sidebar: Old Word selector
choice = st.sidebar.selectbox(
    "Old Word",
    old_words,
    index=st.session_state.idx,
    key="choice_selectbox"
)

# Sync session idx when user picks from sidebar
st.session_state.idx = old_words.index(choice)

# Top-of-page navigation buttons
col1, col2 = st.columns([1,1])
with col1:
    if st.button("⮜ Back"):
        st.session_state.idx = (st.session_state.idx - 1) % n
        st.session_state.choice_selectbox = old_words[st.session_state.idx]
with col2:
    if st.button("Next ⮞"):
        st.session_state.idx = (st.session_state.idx + 1) % n
        st.session_state.choice_selectbox = old_words[st.session_state.idx]

# Current choice after navigation
choice = old_words[st.session_state.idx]

# Look up the selected row
row = df[df["Old Word"] == choice].iloc[0]
new_word = row["New Word"]
sentence = row["Sentence"]

# Display on main page
st.markdown(f"## Old Word\n**{choice}**")
st.markdown(f"### New Word\n**{new_word}**")
st.markdown(f"### Sentence\n{sentence}")

# Text-to-speech
tts = gTTS(text=f"{new_word}. {sentence}", lang="en")
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
