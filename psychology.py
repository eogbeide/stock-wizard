# streamlit_app.py
import streamlit as st
import requests
import re
from io import BytesIO
from gtts import gTTS

# ------- Config -------
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.docx"
TARGET_PAGE_CHARS = 1400  # ~1–2 minutes of narration per page (tweak to taste)

st.set_page_config(page_title="📖 GitHub Doc Reader with TTS", page_icon="🎧", layout="wide")

# ------- Helpers -------
@st.cache_data(show_spinner=False)
def fetch_text(url: str) -> str:
    """Fetch raw file content and coerce to text best-effort."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.content

    # Try UTF-8, fall back to latin-1
    for enc in ("utf-8", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if not text:
        # Last-resort: strip non-printables
        text = ''.join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else ' ' for b in data)

    # Normalize line endings & compress excessive blank lines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def split_by_formfeed(text: str):
    """Split by explicit page breaks if present."""
    parts = [p.strip() for p in text.split('\f') if p.strip()]
    return parts if len(parts) > 1 else None


def smart_paginate(text: str, max_chars: int):
    """
    Group sentences into pages that stay under max_chars (trying to respect paragraph/sentence boundaries).
    """
    # Prefer paragraph-aware splitting, then sentence-aware packing
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Split paragraphs into sentences
    sentence_pattern = re.compile(r'(?<=\S[.?!])\s+(?=[A-Z0-9"“(])')  # naive sentence boundary
    sentences = []
    for p in paragraphs:
        sentences.extend([s.strip() for s in sentence_pattern.split(p) if s.strip()])

    pages = []
    buf = ""
    for s in sentences:
        # +1 for space/newline when joining
        tentative = (buf + " " + s).strip() if buf else s
        if len(tentative) <= max_chars:
            buf = tentative
        else:
            if buf:
                pages.append(buf.strip())
                buf = s
            else:
                # Single very long sentence: hard wrap
                for i in range(0, len(s), max_chars):
                    pages.append(s[i:i+max_chars].strip())
                buf = ""
    if buf:
        pages.append(buf.strip())

    return pages if pages else [text]


@st.cache_data(show_spinner=False)
def paginate_text(text: str, max_chars: int):
    """Return list of pages from text using form-feed or smart pagination."""
    ff = split_by_formfeed(text)
    if ff:
        return ff
    return smart_paginate(text, max_chars)


def speak(text: str) -> BytesIO:
    """Generate TTS mp3 for the given text using gTTS and return as in-memory bytes."""
    # gTTS has ~5k char practical chunk limit; chunk if needed
    chunks = []
    step = 4500
    for i in range(0, len(text), step):
        chunk = text[i:i+step]
        mp3_bytes = BytesIO()
        tts = gTTS(chunk, lang="en")
        tts.write_to_fp(mp3_bytes)
        mp3_bytes.seek(0)
        chunks.append(mp3_bytes.read())

    # If multiple chunks, concatenate simple byte-wise (most players handle sequential MP3 frames)
    combined = BytesIO()
    for c in chunks:
        combined.write(c)
    combined.seek(0)
    return combined

# ------- UI -------
st.title("📖 GitHub Doc Reader with Text-to-Speech")
st.caption("Enter a raw GitHub file URL (text/Doc content). Navigate pages and listen with TTS.")

with st.sidebar:
    url = st.text_input("GitHub Raw File URL", value=DEFAULT_URL)
    st.markdown(
        "Use a **raw** URL (the one that begins with `https://raw.githubusercontent.com/...`).\n\n"
        "• If your file is a classic `.doc` and doesn't decode cleanly, consider converting to plain text or `.docx`.\n"
        "• You can tweak page size below."
    )
    TARGET = st.slider("Target characters per page", 600, 3000, TARGET_PAGE_CHARS, 50)

# Initialize session state
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""

# Load / reload document
reload_needed = (url != st.session_state.loaded_url)
if reload_needed:
    try:
        with st.spinner("Fetching and preparing document..."):
            text = fetch_text(url)
            pages = paginate_text(text, TARGET)
            st.session_state.pages = pages
            st.session_state.page_idx = 0
            st.session_state.loaded_url = url
    except Exception as e:
        st.error(f"Couldn't load the document. Details: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# Navigation controls
col_prev, col_page, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.page_idx == 0):
        st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
with col_page:
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state.page_idx + 1} / {len(st.session_state.pages)}</div>",
        unsafe_allow_html=True
    )
with col_next:
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# Current page content
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("Page Content", page_text, height=420)

# TTS
play_col, dl_col = st.columns([2,1])
with play_col:
    if st.button("🔊 Read this page aloud"):
        try:
            with st.spinner("Generating audio..."):
                audio_bytes = speak(page_text)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"Audio generation failed: {e}")

with dl_col:
    # Offer a simple download of current page text
    st.download_button(
        "⬇️ Download this page (txt)",
        data=page_text.encode("utf-8"),
        file_name=f"page_{st.session_state.page_idx+1}.txt",
        mime="text/plain"
    )

# Optional: quick jump
with st.expander("Jump to page"):
    idx = st.number_input("Go to page #", min_value=1, max_value=len(st.session_state.pages),
                          value=st.session_state.page_idx + 1, step=1)
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
