# psychology.py — MCQs (Question + Explanation only; Options & Answers removed)
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
st.set_page_config(page_title="📖 MCQs Reader — Question + Explanation", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.docx"

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text(data: bytes) -> str:
    if not DOCX_OK:
        raise RuntimeError("python-docx not available. Add 'python-docx' to requirements.txt.")
    doc = Document(BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                parts.append("\f")
    text = "\n".join(parts).replace("\f\n", "\f")
    return normalize_text(text)

# -------- Patterns --------
MCQ_START_PAT = re.compile(
    r"""^\s*(?:
            (?:Q(?:uestion)?\s*\d*)[.)\s:-]* |
            \d+\s*[.)-]\s+
        )""",
    re.IGNORECASE | re.VERBOSE,
)
OPTION_PAT = re.compile(r"""^\s*([A-Ha-h])\s*[\).:,-]\s+""", re.VERBOSE)
ANSWER_PAT = re.compile(r"""^\s*(?:answer|answers?|ans|correct\s*answer|key|solution)\s*[:\-]?\s*(.*)""", re.IGNORECASE)
EXPL_PAT   = re.compile(r"""^\s*(?:explanation|rationale|why|reason(?:ing)?)\s*[:\-]?\s*(.*)""", re.IGNORECASE)

PASSAGE_HEADER_PAT = re.compile(r"^\s*passage(\s*[ivx]+|\s*\d+|\s*[a-z])?\b", re.IGNORECASE)
QUESTIONS_HEADER_PAT = re.compile(r"^\s*questions?\b", re.IGNORECASE)

def looks_like_question(line: str) -> bool:
    return bool(MCQ_START_PAT.match(line))

def parse_answer(line: str):
    m = ANSWER_PAT.match(line)
    return m.group(1).strip() if m else None

def parse_expl(line: str):
    m = EXPL_PAT.match(line)
    return m.group(1).strip() if m else None

def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()

def remove_passages(full_text: str) -> list[str]:
    """
    Remove passage blocks. If a line starts with 'Passage...' we skip lines
    until we hit a question line or a 'Questions' header.
    """
    lines = [l.rstrip() for l in full_text.split("\n")]
    filtered = []
    in_passage = False
    for raw in lines:
        line = raw.strip()

        if PASSAGE_HEADER_PAT.match(line):
            in_passage = True
            continue
        if in_passage:
            if not line:
                continue
            if QUESTIONS_HEADER_PAT.match(line) or looks_like_question(line):
                in_passage = False
            else:
                continue

        filtered.append(raw)
    return filtered

# ---------- Core extraction: Question + Explanation ONLY ----------
def extract_mcqs_qe_only(full_text: str):
    """
    Build items with ONLY:
      - Question: <stem>
      - Explanation: <text> (when present)
    Options and Answers are ignored/omitted entirely.
    Explanation paragraphs may follow an 'Explanation:' line, or come after an 'Answer:' line.
    """
    lines = remove_passages(full_text)

    items = []
    cur_stem_parts: list[str] = []
    cur_expl_parts: list[str] = []

    in_mcq = False
    options_started = False
    capturing_expl = False  # turn on when we see Explanation: OR after Answer: line

    def flush():
        nonlocal cur_stem_parts, cur_expl_parts, in_mcq, options_started, capturing_expl
        stem_text = " ".join([clean_line(s) for s in cur_stem_parts if s is not None]).strip()
        expl_text = " ".join([clean_line(s) for s in cur_expl_parts if s is not None]).strip()
        if stem_text:
            block = [f"Question: {stem_text}"]
            if expl_text:
                block.append(f"Explanation: {expl_text}")
            items.append("\n".join(block).strip())
        # reset
        cur_stem_parts = []
        cur_expl_parts = []
        in_mcq = False
        options_started = False
        capturing_expl = False

    for raw in lines:
        line = raw.strip()
        if not line:
            # keep blank lines inside stem (pre-options) and explanation (when capturing)
            if in_mcq:
                if not options_started and not capturing_expl and cur_stem_parts:
                    cur_stem_parts.append("")
                elif capturing_expl and cur_expl_parts:
                    cur_expl_parts.append("")
            continue

        # New question
        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq = True
            options_started = False
            capturing_expl = False
            cur_stem_parts = [clean_line(line)]
            cur_expl_parts = []
            continue

        if not in_mcq:
            continue

        # Option? (ignored but toggles options_started)
        if OPTION_PAT.match(line):
            options_started = True
            continue

        # Answer? (ignored, but explanations often follow; start capturing expl after this)
        if parse_answer(line) is not None:
            capturing_expl = True
            continue

        # Explanation line?
        expl = parse_expl(line)
        if expl is not None:
            capturing_expl = True
            if expl:
                cur_expl_parts.append(expl)
            continue

        # Free text
        if capturing_expl:
            cur_expl_parts.append(clean_line(line))
        elif not options_started:
            cur_stem_parts.append(clean_line(line))
        else:
            # text after options but before explicit Explanation/Answer:
            # treat as explanation if we've already started capturing
            # (otherwise ignore to avoid mixing with option blocks)
            pass

    if in_mcq:
        flush()

    return items

def mcqs_to_pages(items, per_page: int):
    pages = []
    for i in range(0, len(items), per_page):
        pages.append("\n\n".join(items[i:i+per_page]))
    return pages or ["No MCQs found."]

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

def render_speedy_audio(audio_bytes: BytesIO, rate: float = 2.5, autoplay: bool = False):
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
        height=80,
    )

# ---------- UI ----------
st.title("📖 MCQs — Question + Explanation (Options/Answers removed)")
st.caption("Shows only the question stem and any explanation text. Passages, options, and answer lines are omitted from display.")

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 2.5  # default 2.5×

# ---------- Sidebar ----------
with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    mcqs_per_page = st.slider("MCQs per page", 1, 10, 3, 1)
    st.markdown("- Use a **raw** URL (`https://raw.githubusercontent.com/...`). Best with **.docx** or **.txt**.")

# ---------- Load → decode → extract ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching file..."):
            data = fetch_bytes(url)
        lower = url.lower()
        if lower.endswith(".docx"):
            full_text = extract_docx_text(data)
        elif lower.endswith(".txt") or lower.endswith(".md"):
            full_text = normalize_text(best_effort_bytes_to_text(data))
        elif lower.endswith(".doc"):
            st.warning("Legacy **.doc** detected. Convert to **.docx**/**.txt** for clean parsing.")
            use_fallback = st.toggle("Try fallback decode (may be messy)", value=False)
            if not use_fallback:
                st.stop()
            full_text = normalize_text(best_effort_bytes_to_text(data))
        else:
            full_text = normalize_text(best_effort_bytes_to_text(data))

        items = extract_mcqs_qe_only(full_text)
        pages = mcqs_to_pages(items, mcqs_per_page)

        st.session_state.pages = pages
        st.session_state.page_idx = 0
        st.session_state.loaded_url = url

        if pages and pages[0].startswith("No MCQs"):
            st.warning("No MCQs detected. Make sure questions start with 'Q...' or '1.' and explanations contain 'Explanation:'.")
    except Exception as e:
        st.error(f"Could not load/parse: {e}")
        st.stop()

# ---------- Sidebar dropdown page selector ----------
with st.sidebar:
    total_pages = len(st.session_state.pages)
    if total_pages:
        page_labels = [f"Page {i}" for i in range(1, total_pages + 1)]
        selected_label = st.selectbox(
            "Select page",
            options=page_labels,
            index=min(st.session_state.page_idx, total_pages - 1),
        )
        st.session_state.page_idx = page_labels.index(selected_label)
    else:
        st.selectbox("Select page", options=["(no pages yet)"], index=0, disabled=True)

# ---------- TOP: Speed & Play Controls ----------
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("1.0×"): st.session_state.playback_rate = 1.0
if c2.button("1.5×"): st.session_state.playback_rate = 1.5
if c3.button("2.0×"): st.session_state.playback_rate = 2.0
if c4.button("2.5× (default)"): st.session_state.playback_rate = 2.5
if c5.button("3.0×"): st.session_state.playback_rate = 3.0
if c6.button("0.75×"): st.session_state.playback_rate = 0.75
st.caption(f"Current speed: **{st.session_state.playback_rate}×**")

if st.button("🔊 Generate & Play at selected speed", use_container_width=True):
    try:
        page_text_top = st.session_state.pages[st.session_state.page_idx] if st.session_state.pages else ""
        if not page_text_top:
            st.warning("Nothing to read on this page.")
        else:
            with st.spinner("Generating audio..."):
                audio_buf = tts_mp3(page_text_top)
            render_speedy_audio(audio_buf, rate=st.session_state.playback_rate, autoplay=True)
    except Exception as e:
        st.error(f"TTS failed: {e}")

st.markdown("---")

# ---------- Navigation ----------
left, mid, right = st.columns([1, 3, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.page_idx == 0):
        st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
with mid:
    total = len(st.session_state.pages)
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state.page_idx + 1 if total else 0} / {total}</div>",
        unsafe_allow_html=True,
    )
with right:
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.page_idx >= total - 1):
        st.session_state.page_idx = min(total - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx] if st.session_state.pages else ""
st.text_area("MCQs (Question + Explanation only)", page_text, height=480)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=(page_text or "").encode("utf-8"),
    file_name=f"mcqs_page_{(st.session_state.page_idx + 1) if st.session_state.pages else 0}.txt",
    mime="text/plain",
    disabled=not page_text,
)
