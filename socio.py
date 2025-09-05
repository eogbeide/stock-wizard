# lab.py — Read ALL pages & passages (no exclusions) + TTS
# - Sidebar page navigation + default TTS speed 1.5×
# - Displays ONLY: Question, Correct Answer and Explanation (removes options)
import re
from io import BytesIO
from base64 import b64encode

import requests
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS

# .docx extraction
try:
    from docx import Document
    from docx.enum.text import WD_BREAK
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# ---------- App config ----------
st.set_page_config(page_title="📖 Lab Reader (Q • Correct Answer • Explanation) → TTS", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Socio.docx"
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

def render_audio_player(audio_bytes: BytesIO, rate: float = 1.5):
    """
    Render an HTML5 audio player with a specified playbackRate.
    """
    b64 = b64encode(audio_bytes.getvalue()).decode("utf-8")
    html = f"""
    <audio id="tts_player" controls style="width:100%;">
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    <script>
      const audio = document.getElementById('tts_player');
      audio.addEventListener('loadedmetadata', () => {{
          audio.playbackRate = {rate};
      }});
      audio.playbackRate = {rate};
    </script>
    """
    components.html(html, height=80, scrolling=False)

# ---------- Condense to "Question, Correct Answer and Explanation" ----------
# Patterns
_Q_START_RE = re.compile(r'^\s*(\d+)[\).\s]+(.*\S)?\s*$')               # "1. ..." or "1) ..."
_Q_LABEL_RE = re.compile(r'^\s*question\s*[:\-]\s*(.*)$', re.I)         # "Question: ..."
_A_LABEL_RE = re.compile(r'^\s*answer\s*[:\-]\s*(.*)$', re.I)           # "Answer: ..."
_E_LABEL_RE = re.compile(r'^\s*explanation\s*[:\-]\s*(.*)$', re.I)      # "Explanation: ..."
# Options like "- text", "* text", "a) text", "A. text", "1) text", possibly with ✅/✓ or "(correct)" marker.
_OPT_RE     = re.compile(r'^\s*(?:[\-\*•]|\(?[A-Ea-e1-9]\)?)[\.\)]?\s+(.*?)(\s*(?:✅|✓|\(correct\)|\[correct\]))?\s*$')

def _is_qstart(s: str) -> bool:
    return bool(_Q_START_RE.match(s) or _Q_LABEL_RE.match(s))

def condense_to_qce(page: str) -> str:
    """
    Scan a page, remove MCQ options, and keep:
      - Question:
      - Correct Answer:
      - Explanation:
    Correct Answer is taken from an explicit "Answer:" label or from an option
    line carrying ✅ / ✓ / (correct) / [correct].
    """
    lines = [ln.rstrip() for ln in page.splitlines()]
    i, N = 0, len(lines)
    out = []

    while i < N:
        line = lines[i]
        qmatch_num = _Q_START_RE.match(line)
        qmatch_lbl = _Q_LABEL_RE.match(line)

        if qmatch_num or qmatch_lbl:
            # ---- Extract question text ----
            if qmatch_num:
                qnum = qmatch_num.group(1)
                qtext = (qmatch_num.group(2) or "").strip()
                if not qtext:
                    j = i + 1
                    while j < N and not lines[j].strip():
                        j += 1
                    if j < N:
                        qtext = lines[j].strip()
                        i = j
                q_prefix = f"{qnum}) "
            else:
                qtext = (qmatch_lbl.group(1) or "").strip()
                q_prefix = ""

            i += 1
            # ---- Gather the rest of the block until next question ----
            block = []
            while i < N and not _is_qstart(lines[i]):
                block.append(lines[i])
                i += 1

            # ---- Parse for Correct Answer + Explanation ----
            correct_answer = None
            explanation_lines = []
            k = 0
            while k < len(block):
                cur = block[k].strip()
                if not cur:
                    k += 1
                    continue

                mA = _A_LABEL_RE.match(cur)
                if mA:
                    # Take everything after "Answer:" up to either blank or another label
                    ans = mA.group(1).strip()
                    k += 1
                    cont = []
                    while k < len(block):
                        nxt = block[k].strip()
                        if not nxt or _E_LABEL_RE.match(nxt) or _A_LABEL_RE.match(nxt):
                            break
                        cont.append(nxt)
                        k += 1
                    if cont:
                        ans = (ans + " " + " ".join(cont)).strip()
                    if ans:
                        correct_answer = ans
                    continue

                mE = _E_LABEL_RE.match(cur)
                if mE:
                    exp = mE.group(1).strip()
                    k += 1
                    cont = []
                    while k < len(block):
                        nxt = block[k]
                        if _A_LABEL_RE.match(nxt) or _E_LABEL_RE.match(nxt):
                            break
                        cont.append(nxt.strip())
                        k += 1
                    expl = " ".join([s for s in [exp] + cont if s]).strip()
                    if expl:
                        explanation_lines.append(expl)
                    continue

                # Options: capture only if marked as correct (✅/✓ or "(correct)"/"[correct]")
                mOpt = _OPT_RE.match(cur)
                if mOpt:
                    opt_text = (mOpt.group(1) or "").strip()
                    marked = bool(mOpt.group(2))
                    if marked and not correct_answer and opt_text:
                        correct_answer = opt_text
                    k += 1
                    continue

                k += 1

            # ---- Emit Q / Correct Answer / Explanation ----
            if qtext:
                out.append(f"{q_prefix}Question: {qtext}")
            out.append(f"Correct Answer: {correct_answer if correct_answer else '(not specified)'}")
            if explanation_lines:
                out.append(f"Explanation: {' '.join(explanation_lines).strip()}")
            out.append("")  # blank line between items

            continue  # move to next question block

        i += 1  # not a question start, keep scanning

    cleaned = "\n".join(out).strip()
    # If we couldn't parse anything, just return the original page text
    return cleaned if cleaned else page

# ---------- UI ----------
st.title("📖 Lab Reader → **Question, Correct Answer and Explanation**")
st.caption(
    "Shows only **Question, Correct Answer and Explanation** from each page. "
    "Multiple-choice options are removed. Use the sidebar to navigate pages and set playback speed."
)

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 1.5  # default 1.5×

# ---------- Sidebar: inputs + navigation + speed ----------
with st.sidebar:
    url = st.text_input("GitHub RAW .docx URL", value=DEFAULT_URL)
    target_chars = st.slider("Target characters per page (fallback when no page breaks)", 800, 3200, DEFAULT_PAGE_CHARS, 100)

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
            st.session_state.last_audio = None  # clear previous audio on new load
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# ---------- Sidebar: page navigation & playback speed ----------
with st.sidebar:
    total_pages = len(st.session_state.pages)
    # Page dropdown (1-based)
    current_display_idx = st.session_state.page_idx + 1
    page_choice = st.selectbox(
        "Jump to page",
        options=list(range(1, total_pages + 1)),
        index=current_display_idx - 1,
        help="Navigate directly to a page.",
        key="page_select_sidebar"
    )
    if page_choice != current_display_idx:
        st.session_state.page_idx = page_choice - 1

    # Playback speed selector (default 1.5×)
    speed_options = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    default_index = speed_options.index(1.5)
    playback_rate = st.selectbox(
        "Playback speed",
        options=speed_options,
        index=default_index if st.session_state.playback_rate not in speed_options
              else speed_options.index(st.session_state.playback_rate),
        format_func=lambda x: f"{x}×",
        help="Controls audio playback speed."
    )
    st.session_state.playback_rate = playback_rate

# ---------- Top navigation buttons (optional) ----------
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
raw_page = st.session_state.pages[st.session_state.page_idx]
clean_page = condense_to_qce(raw_page)

# Main viewer shows cleaned "Question, Correct Answer and Explanation"
st.text_area("Question, Correct Answer and Explanation", clean_page, height=520)

# Optional: show original for reference
with st.expander("Original page (raw)"):
    st.text_area("Raw content", raw_page, height=320)

# ---------- Audio: read the CLEANED page ----------
col_play, col_dl = st.columns([2, 1])
with col_play:
    if st.button("🔊 Read this page aloud"):
        try:
            with st.spinner("Generating audio..."):
                audio_buf = tts_mp3(clean_page)
                st.session_state.last_audio = audio_buf
        except Exception as e:
            st.error(f"TTS failed: {e}")

    # If we have audio, render player at selected speed (updates when speed changes)
    if st.session_state.last_audio is not None:
        render_audio_player(st.session_state.last_audio, rate=st.session_state.playback_rate)

with col_dl:
    st.download_button(
        "⬇️ Download this page (Question, Correct Answer and Explanation)",
        data=clean_page.encode("utf-8"),
        file_name=f"page_{st.session_state.page_idx+1}_Q-Ans-Expl.txt",
        mime="text/plain",
    )

# ---------- Jump (numeric input, optional) ----------
with st.expander("Jump to page (numeric)"):
    total = max(1, len(st.session_state.pages))
    idx = st.number_input("Go to page #", min_value=1, max_value=total,
                          value=min(st.session_state.page_idx + 1, total), step=1)
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
