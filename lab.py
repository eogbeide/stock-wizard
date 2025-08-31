# lab.py — Read ALL pages & passages (no exclusions) + TTS, with styled HTML display
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
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labbook.docx"
DEFAULT_PAGE_CHARS = 1600  # used only if no explicit page breaks are present

# ---------- Global styles (readability) ----------
STYLE = """
<style>
  :root {
    --page-max: 980px;
    --card-bg: #ffffff;
    --card-border: #e9eef3;
    --ink: #0f172a;              /* slate-900 */
    --ink-subtle: #334155;        /* slate-600 */
    --accent: #2563eb;            /* blue-600 */
    --shadow: 0 10px 25px rgba(15,23,42,0.06);
  }
  /* Tighten Streamlit's max width a bit */
  .main > div { max-width: var(--page-max); margin-left: auto; margin-right: auto; }
  /* Top title tweak */
  .stApp h1, .stApp h2, .stApp h3 { letter-spacing: 0.2px; }
  /* Page card */
  .page-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 22px 22px;
    margin: 10px 0 18px 0;
  }
  .page-meta {
    font-size: 13px;
    color: var(--ink-subtle);
    margin-bottom: 4px;
  }
  .page-content {
    color: var(--ink);
    line-height: 1.75;
    white-space: pre-wrap;      /* preserve newlines from source text */
    word-wrap: break-word;
    hyphens: auto;
  }
  /* Optional nicer list spacing if present in text */
  .page-content ul, .page-content ol { margin: 0.5rem 1.25rem; }
  /* Subtle hr */
  .soft-hr {
    height: 1px;
    border: none;
    background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    margin: 6px 0 14px 0;
  }
  /* Top controls spacing */
  .controls-wrap { margin-top: 6px; margin-bottom: 8px; }
  .speed-note { font-size: 13px; color: var(--ink-subtle); margin-top: 4px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

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
    """Custom HTML5 audio with adjustable playbackRate (default 1.5×)."""
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto = "autoplay" if autoplay else ""
    elem_id = "tts_player"
    st.components.v1.html(
        f"""
        <div class="page-card">
          <div style="font-weight:600;margin-bottom:6px;">Audio Player</div>
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
        height=120,
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
if "last_target_chars" not in st.session_state:
    st.session_state.last_target_chars = DEFAULT_PAGE_CHARS
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 1.5  # default speed
if "font_px" not in st.session_state:
    st.session_state.font_px = 18  # comfortable default text size

# ---------- Sidebar (URL + pagination settings first) ----------
with st.sidebar:
    st.markdown("### Settings")
    url = st.text_input("GitHub RAW .docx URL", value=DEFAULT_URL)
    target_chars = st.slider(
        "Target characters per page (fallback when no page breaks)",
        800, 3200, DEFAULT_PAGE_CHARS, 100
    )
    st.session_state.font_px = st.slider("Text size (px)", 14, 24, st.session_state.font_px, 1)
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
    st.markdown("### Navigate")
    total_pages = len(st.session_state.pages)
    options = [f"Page {i+1}" for i in range(total_pages)]
    current_idx = min(st.session_state.page_idx, total_pages - 1)
    chosen = st.selectbox("Go to page", options, index=current_idx)
    st.session_state.page_idx = options.index(chosen)

# ---------- TOP: Speed & Read Aloud ----------
st.markdown('<div class="controls-wrap">', unsafe_allow_html=True)
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("0.75×"): st.session_state.playback_rate = 0.75
if c2.button("1.0×"):  st.session_state.playback_rate = 1.0
if c3.button("1.5× (default)"):  st.session_state.playback_rate = 1.5
if c4.button("2.0×"):  st.session_state.playback_rate = 2.0
if c5.button("2.5×"):  st.session_state.playback_rate = 2.5
if c6.button("3.0×"):  st.session_state.playback_rate = 3.0
st.markdown(f'<div class="speed-note">Current speed: <b>{st.session_state.playback_rate}×</b></div>', unsafe_allow_html=True)

# Read Aloud button at the top
if st.button("🔊 Read this page aloud", use_container_width=True):
    try:
        page_text_top = st.session_state.pages[st.session_state.page_idx]
        with st.spinner("Generating audio..."):
            audio_buf = tts_mp3(page_text_top)
        render_speedy_audio(audio_buf, rate=st.session_state.playback_rate, autoplay=True)
    except Exception as e:
        st.error(f"TTS failed: {e}")
st.markdown("</div>", unsafe_allow_html=True)  # end controls-wrap

st.markdown('<hr class="soft-hr" />', unsafe_allow_html=True)

# ---------- Navigation buttons (optional; still available) ----------
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

# ---------- Current page (styled HTML, preserved newlines) ----------
page_text = st.session_state.pages[st.session_state.page_idx]
escaped = html.escape(page_text)  # prevent accidental HTML injection from source
st.markdown(
    f"""
    <section class="page-card">
      <div class="page-meta">Viewing <b>Page {st.session_state.page_idx + 1}</b> of <b>{len(st.session_state.pages)}</b></div>
      <div class="page-content" style="font-size:{st.session_state.font_px}px;">{escaped}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=page_text.encode("utf-8"),
    file_name=f"page_{st.session_state.page_idx+1}.txt",
    mime="text/plain",
)
