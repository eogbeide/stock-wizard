# psychology.py
import re
from io import BytesIO

import requests
import streamlit as st
from gtts import gTTS

# .docx extraction
try:
    from docx import Document
    from docx.enum.text import WD_BREAK
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# ---------- App config ----------
st.set_page_config(page_title="📖 GitHub Doc → Pages + TTS", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.doc"
DEFAULT_PAGE_SIZE = 1400  # characters per page

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    # last resort: strip non-printables
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text(data: bytes) -> str:
    if not DOCX_OK:
        raise RuntimeError("python-docx not available. Add 'python-docx' to requirements.txt.")
    doc = Document(BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
        # explicit page breaks
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                parts.append("\f")
    text = "\n".join(parts)
    text = text.replace("\f\n", "\f")
    return normalize_text(text)

def split_by_formfeed(text: str):
    parts = [p.strip() for p in text.split("\f") if p.strip()]
    return parts if len(parts) > 1 else None

def smart_paginate(text: str, max_chars: int):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]
    sentence_pattern = re.compile(r'(?<=\S[.?!])\s+(?=[A-Z0-9"“(])')
    sentences = []
    for p in paragraphs:
        sentences.extend([s.strip() for s in sentence_pattern.split(p) if s.strip()])

    pages, buf = [], ""
    for s in sentences:
        tentative = (buf + " " + s).strip() if buf else s
        if len(tentative) <= max_chars:
            buf = tentative
        else:
            if buf:
                pages.append(buf)
                buf = s
            else:
                # very long sentence
                for i in range(0, len(s), max_chars):
                    pages.append(s[i : i + max_chars].strip())
                buf = ""
    if buf:
        pages.append(buf)
    return pages or [text]

@st.cache_data(show_spinner=False)
def paginate_text(text: str, max_chars: int):
    ff = split_by_formfeed(text)
    return ff if ff else smart_paginate(text, max_chars)

def tts_mp3(text: str) -> BytesIO:
    # gTTS performs best under ~4500 chars per chunk
    step = 4500
    combined = BytesIO()
    for i in range(0, len(text), step):
        chunk = text[i : i + step]
        buf = BytesIO()
        gTTS(chunk, lang="en").write_to_fp(buf)
        buf.seek(0)
        combined.write(buf.read())
    combined.seek(0)
    return combined

# ---------- Sidebar ----------
st.title("📖 GitHub Doc → Pages + Text-to-Speech")
st.caption("Best with **.docx** or **.txt**. Legacy **.doc** is binary—convert to .docx/.txt for clean text.")
with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    page_size = st.slider("Target characters per page", 600, 3000, DEFAULT_PAGE_SIZE, 50)
    st.markdown(
        "- Use a **raw** URL (starts with `https://raw.githubusercontent.com/...`).\n"
        "- If your file ends with **.doc**, please convert it to **.docx** or **.txt** in the repo."
    )

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# ---------- Load & decode ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching file..."):
            data = fetch_bytes(url)
        lower = url.lower()
        text = None

        if lower.endswith(".docx"):
            text = extract_docx_text(data)
        elif lower.endswith(".txt") or lower.endswith(".md"):
            text = normalize_text(best_effort_bytes_to_text(data))
        elif lower.endswith(".doc"):
            st.warning(
                "This looks like a legacy **.doc** (binary) file. Convert it to **.docx**/**.txt** "
                "in your repo for clean text. You may try a rough fallback decode below."
            )
            use_fallback = st.toggle("Try fallback decode (may be messy)", value=False)
            if not use_fallback:
                st.stop()
            text = normalize_text(best_effort_bytes_to_text(data))
        else:
            # Try best-effort as plain text
            text = normalize_text(best_effort_bytes_to_text(data))

        if not text.strip():
            st.error("No readable text extracted. If this is a `.doc`, convert to `.docx` or `.txt`.")
            st.stop()

        st.session_state.pages = paginate_text(text, page_size)
        st.session_state.page_idx = 0
        st.session_state.loaded_url = url
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

# ---------- Navigation ----------
left, mid, right = st.columns([1, 3, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.page_idx == 0):
        st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
with mid:
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state.page_idx + 1} / {len(st.session_state.pages)}</div>",
        unsafe_allow_html=True,
    )
with right:
    if st.button(
        "Next ➡️",
        use_container_width=True,
        disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1,
    ):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("Page Content", page_text, height=420)

col_play, col_dl = st.columns([2, 1])
with col_play:
    if st.button("🔊 Read this page aloud"):
        try:
            with st.spinner("Generating audio..."):
                audio = tts_mp3(page_text)
            st.audio(audio, format="audio/mp3")
        except Exception as e:
            st.error(f"TTS failed: {e}")

with col_dl:
    st.download_button(
        "⬇️ Download this page (txt)",
        data=page_text.encode("utf-8"),
        file_name=f"page_{st.session_state.page_idx+1}.txt",
        mime="text/plain",
    )

# ---------- Jump ----------
with st.expander("Jump to page"):
    idx = st.number_input(
        "Go to page #",
        min_value=1,
        max_value=len(st.session_state.pages),
        value=st.session_state.page_idx + 1,
        step=1,
    )
    go = st.button("Go")
    if go:
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
