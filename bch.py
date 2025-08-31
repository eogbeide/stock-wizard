# psychology.py — Questions + Answers/Explanations only (Passages removed) + Sidebar page selector
import re
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
st.set_page_config(page_title="📖 MCQs + Answers Reader (GitHub → TTS)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/bchbooks.docx"

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

def looks_like_option(line: str) -> bool:
    return bool(OPTION_PAT.match(line))

def parse_answer(line: str):
    m = ANSWER_PAT.match(line);  return m.group(1).strip() if m else None

def parse_expl(line: str):
    m = EXPL_PAT.match(line);    return m.group(1).strip() if m else None

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

# ---------- Extract only Question + Answer + Explanation (NO options) ----------
def extract_qae_only(full_text: str):
    """
    Keep only:
      - Question (stem text)
      - Answer (whatever follows 'Answer:' etc.)
      - Explanation (optional)
    Options are skipped entirely.
    """
    lines = remove_passages(full_text)

    items = []
    cur_stem_parts = []
    cur_answer, cur_expl = None, None
    in_mcq = False
    options_started = False

    def flush():
        nonlocal cur_stem_parts, cur_answer, cur_expl, in_mcq, options_started
        stem_text = " ".join([clean_line(s) for s in cur_stem_parts if s is not None]).strip()
        if stem_text and (cur_answer or cur_expl):
            block = [stem_text]
            if cur_answer:
                block.append(f"Answer: {cur_answer}")
            if cur_expl:
                block.append(f"Explanation: {cur_expl}")
            items.append("\n".join(block).strip())
        # reset
        cur_stem_parts = []
        cur_answer, cur_expl = None, None
        in_mcq = False
        options_started = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_mcq and cur_stem_parts and not options_started:
                cur_stem_parts.append("")  # paragraph spacing within stem
            continue

        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq = True
            options_started = False
            cur_stem_parts = [clean_line(line)]
            cur_answer, cur_expl = None, None
            continue

        if not in_mcq:
            continue

        # Skip options entirely
        if looks_like_option(line):
            options_started = True
            continue

        # Capture Answer / Explanation
        ans = parse_answer(line)
        if ans is not None:
            cur_answer = ans or cur_answer or ""
            continue

        expl = parse_expl(line)
        if expl is not None:
            cur_expl = (cur_expl + " " + expl).strip() if cur_expl else expl
            continue

        # Extra prose: join to stem before options; after options, treat as continuation of explanation
        if not options_started:
            cur_stem_parts.append(clean_line(line))
        else:
            if cur_answer is not None or cur_expl is not None:
                cur_expl = (cur_expl + " " + clean_line(line)).strip() if cur_expl else clean_line(line)

    if in_mcq:
        flush()

    return items

def mcqs_to_pages(items, per_page: int):
    pages = []
    for i in range(0, len(items), per_page):
        pages.append("\n\n".join(items[i:i+per_page]))
    return pages or ["No questions with answers/explanations were detected."]

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

# ---------- Sidebar (input controls) ----------
st.title("📖 MCQs + Answers Reader (no options)")
st.caption("Shows only **Question**, **Answer**, and **Explanation** — options are removed.")

with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    mcqs_per_page = st.slider("Items per page", 1, 10, 3, 1)
    st.markdown("- Use a **raw** URL (`https://raw.githubusercontent.com/...`). Best with **.docx** or **.txt**.")

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

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

        items = extract_qae_only(full_text)       # <-- options removed here
        pages = mcqs_to_pages(items, mcqs_per_page)

        st.session_state.pages = pages
        st.session_state.page_idx = 0
        st.session_state.loaded_url = url

        if pages and pages[0].startswith("No questions"):
            st.warning("No questions with answers detected. Ensure questions start with 'Q...' or '1.' and answer lines contain 'Answer:'.")
    except Exception as e:
        st.error(f"Could not load/parse: {e}")
        st.stop()

# ---------- Sidebar page selector (dropdown) ----------
if st.session_state.pages:
    with st.sidebar:
        total_pages = len(st.session_state.pages)
        labels = [f"Page {i}" for i in range(1, total_pages + 1)]
        current = min(st.session_state.page_idx, total_pages - 1)
        selected = st.selectbox("Go to page", options=labels, index=current)
        st.session_state.page_idx = labels.index(selected)

# ---------- Navigation ----------
left, mid, right = st.columns([1, 3, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.page_idx == 0):
        st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
with mid:
    total = len(st.session_state.pages) if st.session_state.pages else 0
    pos = (st.session_state.page_idx + 1) if total else 0
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {pos} / {total}</div>",
        unsafe_allow_html=True
    )
with right:
    if st.button("Next ➡️", use_container_width=True, disabled=not st.session_state.pages or st.session_state.page_idx >= len(st.session_state.pages) - 1):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx] if st.session_state.pages else ""
st.text_area("Questions + Answers + Explanations (options removed)", page_text, height=480)

# ---------- TTS & Download ----------
col_play, col_dl = st.columns([2, 1])
with col_play:
    if st.button("🔊 Read this page aloud", disabled=not page_text):
        try:
            with st.spinner("Generating audio..."):
                st.audio(tts_mp3(page_text), format="audio/mp3")
        except Exception as e:
            st.error(f"TTS failed: {e}")

with col_dl:
    st.download_button(
        "⬇️ Download this page (txt)",
        data=(page_text or "").encode("utf-8"),
        file_name=f"mcqs_page_{st.session_state.page_idx+1}.txt",
        mime="text/plain",
        disabled=not page_text,
    )

# ---------- Bottom jump (optional) ----------
with st.expander("Jump to page"):
    total = max(1, len(st.session_state.pages))
    idx = st.number_input("Go to page #", min_value=1, max_value=total, value=min(st.session_state.page_idx + 1, total), step=1)
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
