# lab.py — Read ALL pages & passages (no exclusions) + TTS
# - Sidebar page navigation
# - Default TTS speed 1.5× with selectable speeds
# - Condenses MCQs to "Question, Answer and Explanation" (removes options)
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
st.set_page_config(page_title="📖 Lab Reader (All Pages + Passages → TTS)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Socio.docx"
DEFAULT_PAGE_CHARS = 1600  # used only if no explicit page breaks are present

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text_with_breaks(data: bytes) -> str:
    if not DOCX_OK:
        raise RuntimeError("python-docx not available. Add 'python-docx' to requirements.txt.")
    doc = Document(BytesIO(data))
    out = []
    for para in doc.paragraphs:
        out.append(para.text)
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                out.append("\f")
    text = "\n".join(out)
    text = text.replace("\f\n", "\f").replace("\n\f", "\f")
    return normalize_text(text)

def split_by_formfeed(text: str):
    parts = [p for p in text.split("\f")]
    parts = [p.strip("\n") for p in parts]
    return parts if len(parts) > 1 else None

def smart_paginate(text: str, max_chars: int):
    paragraphs = [p for p in re.split(r"\n\s*\n", text)]
    sent_pat = re.compile(r'(?<=\S[.?!])\s+(?=[A-Z0-9"“(])')
    sentences = []
    for p in paragraphs:
        if not p.strip():
            sentences.append("")
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

# ---------- Condense to "Question, Answer and Explanation" ----------
_Q_START_RE = re.compile(r'^\s*(\d+)[\).\s]+(.*\S)?\s*$')               # "1. ..." or "1) ..."
_Q_LABEL_RE = re.compile(r'^\s*question\s*[:\-]\s*(.*)$', re.I)         # "Question: ..."
_A_LABEL_RE = re.compile(r'^\s*answer\s*[:\-]\s*(.*)$', re.I)           # "Answer: ..."
_E_LABEL_RE = re.compile(r'^\s*explanation\s*[:\-]\s*(.*)$', re.I)      # "Explanation: ..."
_OPT_RE     = re.compile(r'^\s*(?:[\-\*•]|\(?[A-Ea-e1-9]\)?)[\.\)]?\s+(.*?)(\s*✅.*)?\s*$')

def condense_to_qae(page: str) -> str:
    """
    Removes multiple-choice options and returns 'Question, Answer and Explanation'.
    Keeps labels as "Question:", "Answer:", "Explanation:".
    """
    lines = [ln.rstrip() for ln in page.splitlines()]
    i, N = 0, len(lines)
    out = []

    def is_qstart(s: str) -> bool:
        return bool(_Q_START_RE.match(s) or _Q_LABEL_RE.match(s))

    while i < N:
        line = lines[i]
        qmatch_num = _Q_START_RE.match(line)
        qmatch_lbl = _Q_LABEL_RE.match(line)

        if qmatch_num or qmatch_lbl:
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
            block = []
            while i < N and not is_qstart(lines[i]):
                block.append(lines[i])
                i += 1

            answer = None
            explanation_lines = []
            k = 0
            while k < len(block):
                cur = block[k].strip()
                mA = _A_LABEL_RE.match(cur)
                mE = _E_LABEL_RE.match(cur)

                if mA:
                    ans = mA.group(1).strip()
                    k += 1
                    cont = []
                    while k < len(block) and block[k].strip() and not _E_LABEL_RE.match(block[k]) and not _A_LABEL_RE.match(block[k]):
                        cont.append(block[k].strip())
                        k += 1
                    if cont:
                        ans = (ans + " " + " ".join(cont)).strip()
                    answer = ans or answer
                    continue

                if mE:
                    exp = mE.group(1).strip()
                    k += 1
                    cont = []
                    while k < len(block):
                        nxt = block[k]
                        if not nxt.strip():
                            cont.append("")
                            k += 1
                            continue
                        if _A_LABEL_RE.match(nxt) or _E_LABEL_RE.match(nxt):
                            break
                        cont.append(nxt.strip())
                        k += 1
                    explanation_lines.append(exp)
                    if cont:
                        explanation_lines.append(" ".join([c for c in cont if c != ""]))
                    continue

                mOpt = _OPT_RE.match(cur)
                if mOpt:
                    opt_text = mOpt.group(1).strip()
                    has_check = bool(mOpt.group(2)) or ("(correct" in cur.lower())
                    if has_check and not answer:
                        answer = opt_text
                    k += 1
                    continue

                k += 1

            if qtext:
                out.append(f"{q_prefix}Question: {qtext}")
            if answer:
                out.append(f"Answer: {answer}")
            if explanation_lines:
                joined_exp = " ".join([s for s in explanation_lines if s]).strip()
                if joined_exp:
                    out.append(f"Explanation: {joined_exp}")
            out.append("")
            continue

        i += 1

    cleaned = "\n".join(out).strip()
    return cleaned if cleaned else page

# ---------- UI ----------
st.title("📖 Lab Reader (All Content) → Text-to-Speech")
st.caption(
    "Reads **everything** from the document, including all pages and passages. "
    "Multiple-choice options are removed; only **Question, Answer and Explanation** are shown."
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
    target_chars = st.slider("Target characters per page (no page breaks)", 800, 3200, DEFAULT_PAGE_CHARS, 100)
    total_pages = len(st.session_state.pages) if st.session_state.pages else 1
    current_display_idx = (st.session_state.page_idx + 1) if st.session_state.pages else 1
    page_choice = st.selectbox(
        "Jump to page",
        options=list(range(1, total_pages + 1)),
        index=(current_display_idx - 1) if current_display_idx - 1 >= 0 else 0,
        help="Navigate directly to a page.",
        key="page_select_sidebar"
    )
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
            st.session_state.last_audio = None
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# Apply page choice from sidebar
if page_choice != (st.session_state.page_idx + 1):
    st.session_state.page_idx = page_choice - 1

# ---------- Top navigation (optional) ----------
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
clean_page = condense_to_qae(raw_page)

# Main viewer shows cleaned "Question, Answer and Explanation"
st.text_area("Question, Answer and Explanation", clean_page, height=520)

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
    if st.session_state.last_audio is not None:
        render_audio_player(st.session_state.last_audio, rate=st.session_state.playback_rate)

with col_dl:
    st.download_button(
        "⬇️ Download this page (Question, Answer and Explanation, txt)",
        data=clean_page.encode("utf-8"),
        file_name=f"page_{st.session_state.page_idx+1}_QAE.txt",
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
