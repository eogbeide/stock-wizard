import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile

# Cache the data loading
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

DATA_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Sentences.xlsx"
df = load_data(DATA_URL)

# Sidebar selector for Old Word
st.sidebar.title("Choose Old Word")
old_words = df["Old Word"].unique()
choice = st.sidebar.selectbox("Old Word", old_words)

# Look up the selected row
row = df[df["Old Word"] == choice].iloc[0]
new_word = row["New Word"]
sentence = row["Sentence"]

# Display on main page
st.markdown(f"### New Word\n**{new_word}**")
st.markdown(f"### Sentence\n{sentence}")

# Text-to-speech
tts_text = f"{new_word}. {sentence}"
tts = gTTS(text=tts_text, lang="en")

# Save to a temp file and play
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
