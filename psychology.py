# streamlit_app.py
import streamlit as st
import requests
import re
from io import BytesIO
from gtts import gTTS

# For .docx support
from docx import Document
from docx.enum.text import WD_BREAK

st.set_page_config(page_title="📖 GitHub Doc Reader with TTS", page_icon="🎧", layout="wide")

DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.doc"  # current repo file (legacy .doc)
TARGET_PAGE_CHARS = 1400

# ---------- Helpers ----------
def _best_effort_bytes_to_text(data: bytes) -> str:
    """Last-resort decode of bytes to visible text (for binary/unknown encodings)."""
    # Prefer utf-8, then latin-1
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    # Strip non-printables as a final fallback
    return ''.join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else ' ' for b in data)

@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

def extract_docx_text(data: bytes) -> str:
    """Extract text from a .docx file; preserve explicit page breaks as form-feeds \f."""
    doc = Document(BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        # Capture text
        parts.append(para.text)
        # Detect explicit page breaks in runs and insert form feed
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                parts.append("\f")
    # Also check for section breaks by scanning runs for hard page breaks not caught above
    text = "\n".join(parts)
    text = text.replace("\f\n", "\f")  # clean extra newline after FF
    return normalize_text(text)

def split_by_formfeed(text: str):
    parts = [p.strip() for p in text.split('\f') if p.strip()]
    return parts if len(parts) > 1 else None

def smart_paginate(text: str, max_chars: int):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    sentence_pattern = re.compile(r'(?<=\S[.?!])\s+(?=[A-Z0-9"“(])')
    sentences = []
    for p in paragraphs:
        sentences.extend([s.strip() for s in sentence_pattern.split(p) if s.strip()])
    pages, buf = [], ""
    for s in sentences:
        tentative = (buf + " " + s).strip() if buf else s
        if
