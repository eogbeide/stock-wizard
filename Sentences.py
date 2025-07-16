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

# Initialize navigation index
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# Sidebar navigation buttons
st.sidebar.title("Navigate Words")
if st.sidebar.button("⮜ Back"):
    st.session_state.idx = (st.session_state.idx - 1) % n
if st.sidebar.button("Next ⮞"):
    st.session_state.idx = (st.session_state.idx + 1) % n

# Current choice
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
