import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from gtts import gTTS
import tempfile

# Page config & CSS for a cleaner look
st.set_page_config(page_title="📝 Word Transformer", page_icon="🔄", layout="wide")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius:8px; padding:8px 16px;}
    .stSelectbox>div>div>div>div {border-radius:8px; border:1px solid #ddd; padding:4px;}
    .stMarkdown h2 {color: #2E86C1;}
    .stMarkdown h3 {color: #117A65;}
    </style>
""", unsafe_allow_html=True)

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

# Initialize sidebar_choice in session state
if "sidebar_choice" not in st.session_state:
    st.session_state.sidebar_choice = old_words[0]

# Sidebar: Old Word selector
st.sidebar.title("🔍 Select Old Word")
choice = st.sidebar.selectbox(
    "",
    options=old_words,
    index=old_words.index(st.session_state.sidebar_choice),
    key="sidebar_choice"
)
st.sidebar.markdown(f"**{old_words.index(choice)+1} of {n}**")

# Top navigation buttons
st.title("📝 Word Transformer")
col1, col2, _ = st.columns([1,1,8])
with col1:
    if st.button("⮜ Back"):
        idx = old_words.index(choice)
        st.session_state.sidebar_choice = old_words[(idx - 1) % n]
        st.experimental_rerun()
with col2:
    if st.button("Next ⮞"):
        idx = old_words.index(choice)
        st.session_state.sidebar_choice = old_words[(idx + 1) % n]
        st.experimental_rerun()

# Fetch the row for current choice
row = df[df["Old Word"] == st.session_state.sidebar_choice].iloc[0]
new_word = row["New Word"]
sentence = row["Sentence"]

# Display content in two columns
left, right = st.columns(2)
with left:
    st.markdown("## 🅾️ Old Word")
    st.markdown(f"### **{st.session_state.sidebar_choice}**")
with right:
    st.markdown("## 🆕 New Word")
    st.markdown(f"### **{new_word}**")
    st.markdown("## 📖 Sentence")
    st.markdown(f"> {sentence}")

# Text-to-speech
tts = gTTS(text=f"{new_word}. {sentence}", lang="en")
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
