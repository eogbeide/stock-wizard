# lab.py — All pages/passages + Top Play + 1.5x default + Auto-advance (pause supported) — Options removed
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

# ---------- Option-stripping ----------
# Detect typical MCQ options like:
#  A) text   B. text   (c) text   [D] text   a) text
OPTION_PAT = re.compile(r"""^\s*
    (?:[\(\[]?\s*[A-Ha-h]\s*[\)\]\.:,-])   # A-H or a-h with ), ], ., :, -, etc.
    \s+
""", re.VERBOSE)

def remove_mcq_options_from_text(text: str) -> str:
    """
    Remove lines that look like MCQ options (A) ... B. ... etc).
    Keep all other lines (including 'Answer:' and 'Explanation:').
    """
    out = []
    for ln in text.splitlines():
        if OPTION_PAT.match(ln):
            continue  # skip option line entirely
        out.append(ln)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return cleaned

def audio_player_with_autonext(audio_bytes: BytesIO, autoplay: bool, next_idx: int | None, rate: float):
    """
    Renders an <audio> element with given playback rate that (optionally) auto-plays.
    If autoplay is True and next_idx is not None, on 'ended' it navigates to ?p=<next_idx>.
    """
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto_attr = "autoplay" if autoplay else ""

    js_jump = (
        "if (NEXT!==null) { "
        "const u=new URL(window.location); "
        "u.searchParams.set('p', NEXT); "
        "setTimeout(function(){ window.location.href = u.toString(); }, 400); "
        "}"
    )

    js = """
<script>
(function() {
  const AUTO = %s;
  const NEXT = %s;
  const RATE = %s;
  const audio = document.getElementById('tts-audio');
  if (!audio) return;
  function setRate(){ try { audio.playbackRate = RATE; } catch(e){} }
  audio.addEventListener('loadedmetadata', setRate);
  setRate();
  if (AUTO) {
    audio.addEventListener('ended', function() {
      %s
    });
  }
})();
</script>
""" % ("true" if autoplay else "false",
       "null" if next_idx is None else str(int(next_idx)),
       str(float(rate)),
       js_jump if next_idx is not None else "")

    html(
        """
        <audio id="tts-audio" controls %s style="width:100%%;">
          <source src="data:audio/mp3;base64,%s" type="audio/mp3">
          Your browser does not support the audio element.
        </audio>
        %s
        """ % (auto_attr, b64, js),
        height=90,
    )

# ---------- UI & controls ----------
st.title("📖 Lab Reader — Top Play, 1.5× default, auto-advance on finish")
st.caption("Press **Read this page aloud** once; it honors the chosen speed. On finish, it jumps to the next page (loop optional). Options (A/B/C/...) are removed from text.")

with st.sidebar:
    url = st.text_input("GitHub RAW .docx URL", value=DEFAULT_URL)
    target_chars = st.slider("Target characters per page (fallback when no page breaks)", 800, 3200, DEFAULT_PAGE_CHARS, 100)
    loop_end = st.toggle("Loop to first page at the end", value=False)

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 1.5  # default 1.5×
if "want_audio" not in st.session_state:
    st.session_state.want_audio = False   # render player only after pressing play

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
            st.session_state.want_audio = False
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# Clamp index in range
st.session_state.page_idx = max(0, min(st.session_state.page_idx, len(st.session_state.pages) - 1))

# ---------- TOP: Speed & Play ----------
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("0.75×"): st.session_state.playback_rate = 0.75
if c2.button("1.0×"):  st.session_state.playback_rate = 1.0
if c3.button("1.5× (default)"):  st.session_state.playback_rate = 1.5
if c4.button("2.0×"):  st.session_state.playback_rate = 2.0
if c5.button("2.5×"):  st.session_state.playback_rate = 2.5
if c6.button("3.0×"):  st.session_state.playback_rate = 3.0
st.caption(f"Current speed: **{st.session_state.playback_rate}×**")

# Play button at the TOP (generates the player for this page and starts playing)
if st.button("🔊 Read this page aloud", use_container_width=True):
    st.session_state.want_audio = True

st.markdown("---")

# ---------- Navigation (manual) ----------
left, mid, right = st.columns([1, 3, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.page_idx == 0):
        st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
        st.experimental_set_query_params(p=st.session_state.page_idx)
        st.session_state.want_audio = False
with mid:
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state.page_idx + 1} / {len(st.session_state.pages)}</div>",
        unsafe_allow_html=True,
    )
with right:
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)
        st.experimental_set_query_params(p=st.session_state.page_idx)
        st.session_state.want_audio = False

# Keep URL query param in sync with current page (also helps auto-advance)
st.experimental_set_query_params(p=st.session_state.page_idx)

# ---------- Current page (options removed) ----------
raw_page_text = st.session_state.pages[st.session_state.page_idx]
page_text = remove_mcq_options_from_text(raw_page_text)
st.text_area("Page Content (Options removed)", page_text, height=520)

# ---------- Audio player (only after pressing play) + Auto-advance ----------
if st.session_state.want_audio:
    last_idx = len(st.session_state.pages) - 1
    if st.session_state.page_idx < last_idx:
        next_idx = st.session_state.page_idx + 1
    else:
        next_idx = 0 if loop_end else None

    audio_bytes = tts_mp3(page_text)
    # autoplay=True (because user pressed play), honor selected rate, auto-advance on 'ended'
    audio_player_with_autonext(
        audio_bytes,
        autoplay=True,
        next_idx=next_idx,
        rate=st.session_state.playback_rate,
    )

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=page_text.encode("utf-8"),
    file_name=f"page_{st.session_state.page_idx+1}.txt",
    mime="text/plain",
)
