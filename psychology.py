# psychology.py — Question + Explanation only (Passages removed) + Sidebar question selector
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
st.set_page_config(page_title="📖 Question + Explanation (GitHub → TTS)", page_icon="🎧", layout="wide")
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

PASSAGE_HEADER_PAT   = re.compile(r"^\s*passage(\s*[ivx]+|\s*\d+|\s*[a-z])?\b", re.IGNORECASE)
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

def remove_passages(full_text: str):
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

# ---------- Core extraction: Question + Explanation (no Correct Option in output) ----------
LETTER_ONLY = re.compile(r"\b([A-Ha-h])\b")
LETTER_PARENS = re.compile(r"^\s*[\(\[]?([A-Ha-h])[\)\]]?\s*$")

def strip_option_label(line: str) -> tuple[str, str]:
    m = OPTION_PAT.match(line)
    if not m:
        return "", clean_line(line)
    label = m.group(1).upper()
    text = clean_line(line[m.end():])
    return label, text

def extract_q_expl_only(full_text: str):
    """
    Items contain:
      - Question: <stem>
      - Explanation: <text> (optional)
    Options/answers are parsed to help capture explanations, but NOT shown.
    """
    lines = remove_passages(full_text)

    items = []
    cur_stem_parts = []
    cur_answer_raw, cur_expl = None, None
    in_mcq = False
    options_started = False

    def flush():
        nonlocal cur_stem_parts, cur_answer_raw, cur_expl, in_mcq, options_started
        stem_text = " ".join([clean_line(s) for s in cur_stem_parts if s is not None]).strip()
        if stem_text:
            block = [f"Question: {stem_text}"]
            if cur_expl:
                block.append(f"Explanation: {cur_expl}")
            # keep items that have at least a Question (and optionally Explanation)
            if len(block) >= 1:
                items.append("\n".join(block).strip())

        cur_stem_parts = []
        cur_answer_raw, cur_expl = None, None
        in_mcq = False
        options_started = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_mcq and cur_stem_parts and not options_started:
                cur_stem_parts.append("")  # preserve paragraph spacing
            continue

        # New question
        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq = True
            options_started = False
            cur_stem_parts = [clean_line(line)]
            cur_answer_raw, cur_expl = None, None
            continue

        if not in_mcq:
            continue

        # Option? (ignored in output)
        if OPTION_PAT.match(line):
            options_started = True
            # we still parse/consume, but don't store
            _lbl, _txt = strip_option_label(line)
            continue

        # Answer? (ignored in output)
        ans = parse_answer(line)
        if ans is not None:
            cur_answer_raw = ans if ans != "" else cur_answer_raw
            continue

        # Explanation? (kept)
        expl = parse_expl(line)
        if expl is not None:
            cur_expl = (cur_expl + " " + expl).strip() if cur_expl else expl
            continue

        # Extra prose
        if not options_started:
            cur_stem_parts.append(clean_line(line))
        else:
            # after options, treat extra prose as extension of explanation if present
            if (cur_answer_raw is not None) or (cur_expl is not None):
                cur_expl = (cur_expl + " " + clean_line(line)).strip() if cur_expl else clean_line(line)

    if in_mcq:
        flush()

    return items

def pages_from_items(items, per_page: int):
    pages = []
    sep = "\n\n"
    for i in range(0, len(items), per_page):
        pages.append(sep.join(items[i:i+per_page]))
    return pages or ["No Questions found."]

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
st.title("📖 Question + Explanation")
st.caption("Passages and options are removed from the display. Shows only the Question and its Explanation.")

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "items" not in st.session_state:
    st.session_state.items = []          # individual questions
if "pages" not in st.session_state:
    st.session_state.pages = []          # paginated view
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 2.5
if "per_page" not in st.session_state:
    st.session_state.per_page = 3
if "selected_question_idx" not in st.session_state:
    st.session_state.selected_question_idx = 0

# ---------- Sidebar (top controls) ----------
with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    per_page = st.slider("Items per page", 1, 10, st.session_state.per_page, 1)
    st.markdown("- Use a **raw** URL (`https://raw.githubusercontent.com/...`). Best with **.docx** or **.txt**.")

# ---------- Load → decode → extract ----------
need_parse = (url != st.session_state.loaded_url)
if need_parse:
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

        items = extract_q_expl_only(full_text)
        st.session_state.items = items
        st.session_state.loaded_url = url
        # reset paging to start
        st.session_state.page_idx = 0
    except Exception as e:
        st.error(f"Could not load/parse: {e}")
        st.stop()

# Always (re)build pages from items & current per_page
st.session_state.per_page = per_page
st.session_state.pages = pages_from_items(st.session_state.items, st.session_state.per_page)

# ---------- Sidebar (question selector) ----------
def _shorten(s: str, n: int = 80) -> str:
    s = s.splitlines()[0] if s else ""
    s = s.replace("Question:", "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s

with st.sidebar:
    if st.session_state.items:
        labels = [f"{i+1}. {_shorten(it)}" for i, it in enumerate(st.session_state.items)]
        st.session_state.selected_question_idx = st.selectbox(
            "Pick a specific question",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=min(st.session_state.selected_question_idx, max(0, len(labels)-1))
        )
        if st.button("📌 Show selected question"):
            # compute the page that contains this question
            target_q = st.session_state.selected_question_idx
            st.session_state.page_idx = target_q // st.session_state.per_page

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
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state.page_idx + 1} / {len(st.session_state.pages)}</div>",
        unsafe_allow_html=True
    )
with right:
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx] if st.session_state.pages else ""
st.text_area("Question + Explanation", page_text, height=520)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=(page_text or "").encode("utf-8"),
    file_name=f"qe_page_{st.session_state.page_idx+1}.txt",
    mime="text/plain",
    disabled=not page_text,
)

# ---------- Jump ----------
with st.expander("Jump to page"):
    total = max(1, len(st.session_state.pages))
    idx = st.number_input(
        "Go to page #",
        min_value=1,
        max_value=total,
        value=min(st.session_state.page_idx + 1, total),
        step=1
    )
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
