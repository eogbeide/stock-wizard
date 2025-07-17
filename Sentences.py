import streamlit as std
import pandas as pd
import requests
from io import BytesIO
from gtts import gTTS
import tempfile

# Page config & custom CSS
st.set_page_config(page_title="📝 Word Transformer", page_icon="🔄", layout="wide")
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius:8px; padding:8px 16px;}
    .stSelectbox>div>div>div>div {border-radius:8px; border:1px solid #ddd; padding:4px;}
    .stMarkdown h2 {color: #2E86C1;}
    .stMarkdown h3 {color: #117A65;}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return pd.read_excel(BytesIO(resp.content))

DATA_URL = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Sentences.xls'
try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Prepare list of Old Words
old_words = df["Old Word"].dropna().unique().tolist()
n = len(old_words)

# Initialize index
if "idx" not in st.session_state:
    st.session_state.idx = 0

# Sidebar: Old Word selector
st.sidebar.title("🔍 Select Old Word")
choice = st.sidebar.selectbox(
    "",
    options=old_words,
    index=st.session_state.idx
)
st.sidebar.markdown(f"**{old_words.index(choice)+1} of {n}**")

# Sync idx to sidebar selection
st.session_state.idx = old_words.index(choice)

# Top navigation buttons
st.title("📝 Word Transformer")
col1, col2, _ = st.columns([1,1,8])
with col1:
    if st.button("⮜ Back"):
        st.session_state.idx = (st.session_state.idx - 1) % n
with col2:
    if st.button("Next ⮞"):
        st.session_state.idx = (st.session_state.idx + 1) % n

# Current Old Word
current = old_words[st.session_state.idx]

# Filter all matching rows
matches = df[df["Old Word"] == current]

# Display Old Word
st.markdown("## 🅾️ Old Word")
st.markdown(f"### **{current}**")

# Display all New Words & Sentences
st.markdown("## 🆕 New Words & Sentences")
for i, (_, r) in enumerate(matches.iterrows(), start=1):
    st.markdown(f"**{i}. {r['New Word']}**")
    st.markdown(f"> {r['Sentence']}")

# Text-to-speech: read all
text_to_speak = " ".join(
    f"{r['New Word']}. {r['Sentence']}" for _, r in matches.iterrows()
)
tts = gTTS(text=text_to_speak, lang="en")
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    tts.save(fp.name)
    st.audio(fp.name, format="audio/mp3")
