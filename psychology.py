# psychology.py — Question + Explanation only (Passages & Correct Option removed)
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

# ---------- Core extraction: Question + Explanation (no Correct Option output) ----------
def extract_q_expl_only(full_text: str):
    """
    Items contain:
      - Question: <stem>
      - Explanation: <text> (optional)
    Options and 'Correct Option' are omitted from output entirely.
    """
    lines = remove_passages(full_text)
    items = []
    cur_stem_parts = []
    cur_expl = None
    in_mcq = False
    options_started = False

    def flush():
        nonlocal cur_stem_parts, cur_expl, in_mcq, options_started
        stem_text = " ".join([clean_line(s) for s in cur_stem_parts if s is not None]).strip()
        if stem_text:
            block = [f"Question: {stem_text}"]
            if cur_expl:
                block.append(f"Explanation: {cur_expl}")
            items.append("\n".join(block).strip())
        # reset
        cur_stem_parts = []
        cur_expl = None
        in_mcq = False
        options_started = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_mcq and cur_stem_parts and not options_started:
                cur_stem_parts.append("")
            continue

        # new question
        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq = True
            options_started = False
            cur_stem_parts = [clean_line(line)]
            cur_expl = None
            continue

        if not in_mcq:
            continue

        # Option? (skip from output, but mark that options started so further prose isn't appended to stem)
        if OPTION_PAT.match(line):
            options_started = True
            continue

        # Answer? (ignore in output)
        if parse_answer(line) is not None:
            continue

        # Explanation?
        expl = parse_expl(line)
        if expl is not None:
            cur_expl = (cur_expl + " " + expl).strip() if cur_expl else expl
            continue

        # Extra prose: if before options -> part of stem; after options -> treat as continuation of explanation
        if not options_started:
            cur_stem_parts.append(clean_line(line))
        else:
            cur_expl = (cur_expl + " " + clean_line(line)).strip() if cur_expl else clean_line(line)

    if in_mcq:
        flush()

    # Filter out any empties (paranoia)
    return [x for x in items if x.strip()]

# ---------- Pagination ----------
def pages_from_items(items, per_page: int):
    try:
        per_page = int(per_page)
    except Exception:
        per_page = 3
    per_page = max(1, per_page)
    pages = []
    sep = "\n\n"
    for i in range(0, len(items), per_page):
        pages.append(sep.join(items[i:i+per_page]))
    return pages or ["No Questions found."]

def clamp_page_idx():
    """Ensure page_idx is a valid index into pages."""
    pages = st.session_state.get("pages", [])
    n = len(pages)
    if n == 0:
        st.session_state["page_idx"] = 0
    else:
        st.session_state["page_idx"] = max(0, min(st.session_state.get("page_idx", 0), n - 1))

# ---------- TTS ----------
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
    """HTML5 audio with adjustable playbackRate."""
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
st.caption("Passages and the *Correct Option* line are removed. Shows each Question with its Explanation (if present).")

# ---------- Session state (safe keys) ----------
if "loaded_url" not in st.session_state:
    st.session_state["loaded_url"] = ""
if "items" not in st.session_state:
    st.session_state["items"] = []
if "pages" not in st.session_state:
    st.session_state["pages"] = []
if "page_idx" not in st.session_state:
    st.session_state["page_idx"] = 0
if "playback_rate" not in st.session_state:
    st.session_state["playback_rate"] = 2.5
if "per_page" not in st.session_state:
    st.session_state["per_page"] = 3

# ---------- Sidebar ----------
with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    per_page = st.slider("Items per page", 1, 10, st.session_state["per_page"], 1)
    st.markdown("- Use a **raw** URL (`https://raw.githubusercontent.com/...`). Best with **.docx** or **.txt**.")
    st.divider()
    # New: direct question-number jump
    total_q = max(1, len(st.session_state.get("items", [])))
    qnum = st.number_input("Go to question #", min_value=1, max_value=total_q, value=1, step=1)
    if st.button("📌 Show selected question"):
        # Convert question number to page index
        q_index = int(qnum) - 1
        pp = max(1, int(per_page))
        st.session_state["page_idx"] = q_index // pp
        clamp_page_idx()
        st.experimental_rerun()

# ---------- Load → decode → extract (when URL changes) ----------
if url != st.session_state["loaded_url"]:
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

        st.session_state["items"] = extract_q_expl_only(full_text) or []
        st.session_state["loaded_url"] = url
        st.session_state["page_idx"] = 0
    except Exception as e:
        st.session_state["items"] = []
        st.error(f"Could not load/parse: {e}")
        st.stop()

# ---------- Build pages & clamp ----------
st.session_state["per_page"] = int(per_page) if per_page else 3
st.session_state["pages"] = pages_from_items(st.session_state["items"], st.session_state["per_page"])
clamp_page_idx()

# ---------- TOP: Speed & Play Controls ----------
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)
if c1.button("1.0×"): st.session_state["playback_rate"] = 1.0
if c2.button("1.5×"): st.session_state["playback_rate"] = 1.5
if c3.button("2.0×"): st.session_state["playback_rate"] = 2.0
if c4.button("2.5× (default)"): st.session_state["playback_rate"] = 2.5
if c5.button("3.0×"): st.session_state["playback_rate"] = 3.0
if c6.button("0.75×"): st.session_state["playback_rate"] = 0.75
st.caption(f"Current speed: **{st.session_state['playback_rate']}×**")

if st.button("🔊 Generate & Play at selected speed", use_container_width=True):
    try:
        clamp_page_idx()
        page_text_top = st.session_state["pages"][st.session_state["page_idx"]] if st.session_state["pages"] else ""
        if not page_text_top:
            st.warning("Nothing to read on this page.")
        else:
            with st.spinner("Generating audio..."):
                audio_buf = tts_mp3(page_text_top)
            render_speedy_audio(audio_buf, rate=st.session_state["playback_rate"], autoplay=True)
    except Exception as e:
        st.error(f"TTS failed: {e}")

st.markdown("---")

# ---------- Navigation ----------
left, mid, right = st.columns([1, 3, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state["page_idx"] == 0):
        st.session_state["page_idx"] = max(0, st.session_state["page_idx"] - 1)
with mid:
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state['page_idx'] + 1} / {len(st.session_state['pages'])}</div>",
        unsafe_allow_html=True
    )
with right:
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state["page_idx"] >= len(st.session_state["pages"]) - 1):
        st.session_state["page_idx"] = min(len(st.session_state["pages"]) - 1, st.session_state["page_idx"] + 1)

# ---------- Current page ----------
clamp_page_idx()
page_text = st.session_state["pages"][st.session_state["page_idx"]] if st.session_state["pages"] else ""
st.text_area("Question + Explanation", page_text, height=520)

# ---------- Download ----------
st.download_button(
    "⬇️ Download this page (txt)",
    data=(page_text or "").encode("utf-8"),
    file_name=f"qe_page_{st.session_state['page_idx']+1}.txt",
    mime="text/plain",
    disabled=not page_text,
)

# ---------- Jump ----------
with st.expander("Jump to page"):
    total = max(1, len(st.session_state["pages"]))
    idx = st.number_input(
        "Go to page #",
        min_value=1,
        max_value=total,
        value=min(st.session_state["page_idx"] + 1, total),
        step=1
    )
    if st.button("Go"):
        st.session_state["page_idx"] = int(idx) - 1
        clamp_page_idx()
        st.experimental_rerun()
