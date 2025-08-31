# lab.py — Read ALL pages & passages (no exclusions) + TTS
import re
import base64
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
st.set_page_config(page_title="📖 Lab Reader (All Pages + Passages → TTS)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labbook.docx"
DEFAULT_PAGE_CHARS = 1600  # used only if no explicit page breaks are present

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    # keep ALL content; just standardize newlines and compress super-long blank gaps a bit
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # cap at max 3 consecutive blanks
    return text.strip()

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    # last resort: retain visible bytes
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text_with_breaks(data: bytes) -> str:
    """
    Extract ALL text from .docx, inserting \f for explicit page breaks.
    Nothing is filtered out.
    """
    if not DOCX_OK:
        raise RuntimeError("python-docx not available. Add 'python-docx' to requirements.txt.")
    doc = Document(BytesIO(data))
    out = []
    for para in doc.paragraphs:
        out.append(para.text)
        # insert form-feed when Word page break is present
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                out.append("\f")
    text = "\n".join(out)
    # clean duplicated newline around form feeds
    text = text.replace("\f\n", "\f").replace("\n\f", "\f")
    return normalize_text(text)

def split_by_formfeed(text: str):
    # Prefer explicit page breaks if present
    parts = [p for p in text.split("\f")]
    # Keep empty pages if any (rare), then trim edges lightly
    parts = [p.strip("\n") for p in parts]
    return parts if len(parts) > 1 else None

def smart_paginate(text: str, max_chars: int):
    """
    If the doc has no explicit page breaks, paginate by sentences/paragraphs
    without dropping anything.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", text)]
    # sentence boundaries (naive but safe)
    sent_pat = re.compile(r'(?<=\S[.?!])\s+(?=[A-Z0-9"“(])')

    sentences = []
    for p in paragraphs:
        if not p.strip():
            sentences.append("")  # preserve paragraph break
            continue
        sentences.extend(sent_pat.split(p))

    pages, buf = [], ""
    for s in sentences:
        # keep paragraph breaks (blank lines) as newlines
        s_clean = s if s == "" else s.strip()
        candidate = (buf + ("\n\n" if s_clean == "" else (" " if buf and s_clean else "")) + s_clean).rstrip()
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                pages.append(buf.strip())
                buf = s_clean
            else:
                # extremely long single chunk: hard wrap to ensure nothing is lost
                for i in range(0, len(s_clean), max_chars):
                    pages.append(s_clean[i : i + max_chars].strip())
                buf = ""
    if buf:
        pages.append(buf.strip())

    return pages if pages else [text]

@st.cache_data(show_spinner=False)
def paginate_full_text(text: str, max_chars: int):
    # First try real page breaks; otherwise paginate smartly
    ff = split_by_formfeed(text)
    return ff if ff else smart_paginate(text, max_chars)

def tts_mp3(text: str) -> BytesIO:
    # gTTS works best in <~4500 char chunks
    step = 4500
    combined = BytesIO()
    for i in range(0, len(text), step):
        chunk = text[i:i+step]
        buf = BytesIO()
        gTTS(chunk, lang="en").write_to_fp(buf)
        buf.seek(0)
        combined.write(buf.read())
    combined.seek(0)
    return combined

def render_speedy_audio(audio_bytes: BytesIO, rate: float = 1.5, autoplay: bool = False):
    """Render a custom HTML5 audio player with adjustable playbackRate (default 2.5x)."""
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto = "autoplay" if autoplay else ""
    elem_id = "tts_player"
    st.components.v1.html(
        f"""
        <div>
          <audio id="{elem_id}" controls {auto} style="width:100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          </audio>
          <script>
            const p = document.getElementById("{elem_id}");
            if (p) {{
              p.playbackRate = {rate};
            }}
          </script>
        </div>
        """,
        height=84,
    )

# ---------- UI ----------
st.title("📖 Lab Reader (All Content) → Text-to-Speech")
st.caption("Reads **everything** from the document, including all pages and passages. Nothing is excluded.")

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 2.5  # default speed

# ---------- Sidebar (inputs + page selector) ----------
with st.sidebar:
    url = st.text_input("GitHub RAW .docx URL", value=DEFAULT_URL)
    target_chars = st.slider(
        "Target characters per page (fallback when no page breaks)",
        800, 3200, DEFAULT_PAGE_CHARS, 100
    )
    st.markdown(
        "- Use a **raw** URL (`https://raw.githubusercontent.com/...`).\n"
        "- If your file isn’t `.docx`, convert it to `.docx` for best results."
    )

# ---------- Load & paginate ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching and preparing document..."):
            data = fetch_bytes(url)
            if url.lower().endswith(".docx"):
                full_text = extract_docx_text_with_breaks(data)
            else:
                # best-effort for .txt/.md
                full_text = normalize_text(best_effort_bytes_to_text(data))

            pages = paginate_full_text(full_text, target_chars)
            st.session_state.pages = pages
            st.session_state.page_idx = 0
            st.session_state.loaded_url = url
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# ---------- Sidebar page selector (dropdown) ----------
with st.sidebar:
    total_pages = len(st.session_state.pages)
    labels = [f"Page {i}" for i in range(1, total_pages + 1)]
    current_idx = min(st.session_state.page_idx, total_pages - 1)
    selected_label = st.selectbox("Go to page", options=labels, index=current_idx)
    st.session_state.page_idx = labels.index(selected_label)

# ---------- TOP: Speed & Read Aloud ----------
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("0.75×"): st.session_state.playback_rate = 0.75
if c2.button("1.0×"):  st.session_state.playback_rate = 1.0
if c3.button("1.5×"):  st.session_state.playback_rate = 1.5
if c4.button("2.0×"):  st.session_state.playback_rate = 2.0
if c5.button("2.5× (default)"): st.session_state.playback_rate = 2.5
if c6.button("3.0×"):  st.session_state.playback_rate = 3.0
st.caption(f"Current speed: **{st.session_state.playback_rate}×**")

# Read aloud button at the top
if st.button("🔊 Generate & Play at selected speed", use_container_width=True):
    try:
        curr_text = st.session_state.pages[st.session_state.page_idx] if st.session_state.pages else ""
        if not curr_text:
            st.warning("Nothing to read on this page.")
        else:
            with st.spinner("Generating audio..."):
                audio_buf = tts_mp3(curr_text)
            render_speedy_audio(audio_buf, rate=st.session_state.playback_rate, autoplay=True)
    except Exception as e:
        st.error(f"TTS failed: {e}")

st.markdown("---")

# ---------- Navigation (buttons) ----------
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
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("Page Content (Full, unfiltered)", page_text, height=520)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=page_text.encode("utf-8"),
    file_name=f"page_{st.session_state.page_idx+1}.txt",
    mime="text/plain",
)
