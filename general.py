# lab.py — All pages/passages + Auto-play + Auto-advance (pause supported)
import re
import base64
from io import BytesIO

import requests
import streamlit as st
from gtts import gTTS
from streamlit.components.v1 import html

# .docx extraction
try:
    from docx import Document
    from docx.enum.text import WD_BREAK
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# ---------- App config ----------
st.set_page_config(page_title="📖 Lab Reader (Auto TTS → Auto Next)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/General.docx"
DEFAULT_PAGE_CHARS = 1600  # fallback when no explicit page breaks exist

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # cap huge blank gaps
    return text.strip()

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text_with_breaks(data: bytes) -> str:
    """Extract ALL text from .docx, inserting \f on explicit page breaks."""
    if not DOCX_OK:
        raise RuntimeError("python-docx not available. Add 'python-docx' to requirements.txt.")
    doc = Document(BytesIO(data))
    out = []
    for para in doc.paragraphs:
        out.append(para.text)
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                out.append("\f")
    text = "\n".join(out).replace("\f\n", "\f").replace("\n\f", "\f")
    return normalize_text(text)

def split_by_formfeed(text: str):
    parts = [p.strip("\n") for p in text.split("\f")]
    return parts if len(parts) > 1 else None

def smart_paginate(text: str, max_chars: int):
    paragraphs = [p for p in re.split(r"\n\s*\n", text)]
    sent_pat = re.compile(r'(?<=\S[.?!])\s+(?=[A-Z0-9"“(])')

    sentences = []
    for p in paragraphs:
        if not p.strip():
            sentences.append("")  # paragraph break
            continue
        sentences.extend(sent_pat.split(p))

    pages, buf = [], ""
    for s in sentences:
        s_clean = s if s == "" else s.strip()
        candidate = (buf + ("\n\n" if s_clean == "" else (" " if buf and s_clean else "")) + s_clean).rstrip()
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                pages.append(buf.strip())
                buf = s_clean
            else:
                for i in range(0, len(s_clean), max_chars):
                    pages.append(s_clean[i:i+max_chars].strip())
                buf = ""
    if buf:
        pages.append(buf.strip())
    return pages if pages else [text]

@st.cache_data(show_spinner=False)
def paginate_full_text(text: str, max_chars: int):
    ff = split_by_formfeed(text)
    return ff if ff else smart_paginate(text, max_chars)

def tts_mp3(text: str) -> BytesIO:
    """Generate MP3 (gTTS) for the text. Chunk to keep requests small."""
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

def audio_player_with_autonext(audio_bytes: BytesIO, autoplay: bool, next_idx: int | None):
    """
    Renders an <audio> element that (optionally) auto-plays.
    If autoplay is True and next_idx is not None, when playback ends it navigates
    to the same app URL with ?p=<next_idx>, which advances the page.
    """
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto_attr = "autoplay" if autoplay else ""
    # Small delay before jumping helps on some browsers after 'ended'
    js = f"""
      <script>
        (function() {{
          const AUTO = {str(autoplay).lower()};
          const NEXT = {('' if next_idx is None else int(next_idx))};
          const audio = document.getElementById('tts-audio');
          if (!audio) return;
          if (AUTO) {{
            audio.addEventListener('ended', function() {{
              {"const u=new URL(window.location); u.searchParams.set('p', NEXT); setTimeout(()=>{ window.location.href = u.toString(); }, 400);" if next_idx is not None else ""}
            }});
          }}
        }})();
      </script>
    """
    html(
        f"""
        <audio id="tts-audio" controls {auto_attr} style="width:100%;">
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          Your browser does not support the audio element.
        </audio>
        {js}
        """,
        height=90,
    )

# ---------- UI & controls ----------
st.title("📖 Lab Reader — Auto-play each page, auto-advance unless paused")
st.caption("Hit **Start** once (browser gesture), then it will read and jump to the next page automatically.")

with st.sidebar:
    url = st.text_input("GitHub RAW .docx URL", value=DEFAULT_URL)
    target_chars = st.slider("Target characters per page (fallback when no page breaks)", 800, 3200, DEFAULT_PAGE_CHARS, 100)
    loop_end = st.toggle("Loop to first page at the end", value=False)

    # Playback controls
    if "auto" not in st.session_state:
        st.session_state.auto = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start / Resume"):
            st.session_state.auto = True
    with c2:
        if st.button("⏸️ Pause"):
            st.session_state.auto = False

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# Sync page index from URL query (?p=)
try:
    q = st.experimental_get_query_params()
    if "p" in q:
        p = int(q["p"][0])
        if 0 <= p:
            st.session_state.page_idx = p
except Exception:
    pass  # ignore malformed values

# ---------- Load & paginate ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching and preparing document..."):
            data = fetch_bytes(url)
            if url.lower().endswith(".docx"):
                full_text = extract_docx_text_with_breaks(data)
            else:
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

# Clamp index in range
st.session_state.page_idx = max(0, min(st.session_state.page_idx, len(st.session_state.pages) - 1))

# ---------- Navigation (manual) ----------
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

# Keep URL query param in sync with current page
st.experimental_set_query_params(p=st.session_state.page_idx)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("Page Content (Full, unfiltered)", page_text, height=520)

# ---------- Auto TTS + Auto-advance tied to audio 'ended' ----------
# Decide what the *next* page index should be (or None if we should stop at the end).
last_idx = len(st.session_state.pages) - 1
if st.session_state.page_idx < last_idx:
    next_idx = st.session_state.page_idx + 1
else:
    next_idx = 0 if loop_end else None

# Generate & play audio (autoplay when auto==True). On 'ended', JS hops to ?p=next_idx.
audio_bytes = tts_mp3(page_text)
audio_player_with_autonext(audio_bytes, autoplay=st.session_state.auto, next_idx=next_idx)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=page_text.encode("utf-8"),
    file_name=f"page_{st.session_state.page_idx+1}.txt",
    mime="text/plain",
)

# ---------- Jump ----------
with st.expander("Jump to page"):
    total = max(1, len(st.session_state.pages))
    idx = st.number_input(
        "Go to page #",
        min_value=1,
        max_value=total,
        value=min(st.session_state.page_idx + 1, total),
        step=1,
    )
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_set_query_params(p=st.session_state.page_idx)
        st.experimental_rerun()
