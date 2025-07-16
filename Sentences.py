import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from gtts import gTTS
import tempfile

# Cache the data loading, fetching via requests to avoid urllib HTTP errors
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

# Sidebar selector for Old Word
st.sidebar.title("Choose Old Word")
old_words = df["Old Word"].dropna().unique()
choice = st.sidebar.selectbox("Old Word", old_words)

# Look up the selected row
row = df[df["Old Word"] == choice].iloc[0]
new_word = row["New Word"]
sentence = row["Sentence"]

# Display on main page
st.markdown(f"### New Word\n**{new_word}**")
st.markdown(f"### Sentence\n{sentence}")

# Text-to-speech
tts = gTTS(text=f"{new_word}. {sentence}", lang="en")
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
