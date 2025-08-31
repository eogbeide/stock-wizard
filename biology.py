# lab.py — Read ALL pages & passages (no exclusions) + TTS with readable formatting
import re
import html
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
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/BIO.docx"
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

def render_speedy_audio(audio_bytes: BytesIO, rate: float = 1.5, autoplay: bool = True):
    """
    Custom HTML5 audio with adjustable playbackRate. Uses base64 data URL and
    escapes curly braces for f-strings to avoid SyntaxError.
    """
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto = "autoplay" if autoplay else ""
    elem_id = "tts_player"
    st.components.v1.html(
        f"""
        <div class="audio-wrap" style="margin-bottom:0.5rem;">
          <audio id="{elem_id}" controls {auto} style="width:100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          </audio>
          <script>
            const p = document.getElementById("{elem_id}");
            if (p) {{ p.playbackRate = {rate}; }}
          </script>
        </div>
        """,
        height=80,
    )

# --------- Pretty HTML rendering ----------
READABLE_CSS = """
<style>
.readable {
  max-width: 980px;
  margin: 0 auto;
  background: #ffffff;
  color: #0f172a;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  font-size: 1.05rem;
  line-height: 1.75;
  letter-spacing: 0.01em;
  padding: 1.2rem 1.4rem;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05), 0 10px 20px rgba(0,0,0,.04);
  border: 1px solid rgba(2, 6, 23, .06);
}
.readable h1, .readable h2, .readable h3 {
  line-height: 1.3; margin: 0.4rem 0 0.6rem; font-weight: 700;
}
.readable h1 { font-size: 1.55rem; }
.readable h2 { font-size: 1.35rem; }
.readable h3 { font-size: 1.2rem; }
.readable p { margin: 0 0 0.9rem; }
.readable ul, .readable ol { margin: 0 0 1rem 1.2rem; }
.readable li { margin: 0.25rem 0; }
.readable .hr { height: 1px; background: rgba(2,6,23,.08); margin: 1rem 0; }
</style>
"""

def _is_heading(txt: str) -> int:
    """Return 1 for h1-ish, 2 for h2-ish, 0 otherwise."""
    s = txt.strip()
    if not s:
        return 0
    if len(s) <= 80 and (s.isupper() or s.endswith(":") or re.match(r"^(chapter|section|unit|topic|lesson)\b", s, re.I)):
        return 1
    if len(s) <= 100 and re.match(r"^(\d+[\.)]\s+.+|[IVXLC]+\.\s+.+)$", s):
        return 2
    return 0

def to_readable_html(text: str) -> str:
    """
    Convert plain text to readable HTML:
    - Split by blank lines into paragraphs
    - Detect headings
    - Convert bullet-like lines to lists
    - Preserve single newlines inside paragraphs with <br>
    """
    parts = re.split(r"\n\s*\n", text.strip())
    html_parts = []
    for block in parts:
        raw = block.strip("\n")
        if not raw:
            continue

        # Heading?
        h = _is_heading(raw)
        if h == 1:
            html_parts.append(f"<h1>{html.escape(raw.title())}</h1>")
            continue
        elif h == 2:
            html_parts.append(f"<h2>{html.escape(raw)}</h2>")
            continue

        # Bullets or numbered list?
        lines = [ln.strip() for ln in raw.split("\n")]
        if all(re.match(r"^([*•\-–]\s+|\d+[\.\)]\s+)", ln) for ln in lines if ln):
            items = []
            ordered = all(re.match(r"^\d+[\.\)]\s+", ln) for ln in lines if ln)
            for ln in lines:
                if not ln: 
                    continue
                ln = re.sub(r"^([*•\-–]\s+|\d+[\.\)]\s+)", "", ln)
                items.append(f"<li>{html.escape(ln)}</li>")
            tag = "ol" if ordered else "ul"
            html_parts.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
        else:
            # regular paragraph; preserve single newlines
            html_parts.append("<p>" + html.escape(raw).replace("\n", "<br>") + "</p>")

    return READABLE_CSS + '<div class="readable">' + "\n".join(html_parts) + "</div>"

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
if "last_target_chars" not in st.session_state:
    st.session_state.last_target_chars = DEFAULT_PAGE_CHARS
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 1.5  # default speed

# ---------- Sidebar (URL + pagination settings first) ----------
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

# ---------- Load & paginate (re-run on URL or target_chars change) ----------
needs_reload = (url != st.session_state.loaded_url) or (target_chars != st.session_state.last_target_chars)

if needs_reload:
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
            st.session_state.last_target_chars = target_chars
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# ---------- Sidebar (page selector after pages are available) ----------
with st.sidebar:
    total_pages = len(st.session_state.pages)
    options = [f"Page {i+1}" for i in range(total_pages)]
    current_idx = min(st.session_state.page_idx, total_pages - 1)
    chosen = st.selectbox("Go to page", options, index=current_idx)
    st.session_state.page_idx = options.index(chosen)

# ---------- TOP: Speed & Read Aloud ----------
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("0.75×"): st.session_state.playback_rate = 0.75
if c2.button("1.0×"):  st.session_state.playback_rate = 1.0
if c3.button("1.5× (default)"):  st.session_state.playback_rate = 1.5
if c4.button("2.0×"):  st.session_state.playback_rate = 2.0
if c5.button("2.5×"):  st.session_state.playback_rate = 2.5
if c6.button("3.0×"):  st.session_state.playback_rate = 3.0
st.caption(f"Current speed: **{st.session_state.playback_rate}×**")

# Read Aloud button at the top (autoplay)
if st.button("🔊 Read this page aloud", use_container_width=True):
    try:
        page_text_top = st.session_state.pages[st.session_state.page_idx]
        with st.spinner("Generating audio..."):
            audio_buf = tts_mp3(page_text_top)
        render_speedy_audio(audio_buf, rate=st.session_state.playback_rate, autoplay=True)
    except Exception as e:
        st.error(f"TTS failed: {e}")

st.markdown("---")

# ---------- Current page (readable formatting) ----------
page_text = st.session_state.pages[st.session_state.page_idx]
st.markdown(to_readable_html(page_text), unsafe_allow_html=True)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=page_text.encode("utf-8"),
    file_name=f"page_{st.session_state.page_idx+1}.txt",
    mime="text/plain",
)
