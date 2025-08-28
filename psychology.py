# psychology.py — MCQs + Options + Answers/Explanations only (GitHub → TTS)
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
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.docx"

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines
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
        # explicit page breaks
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                parts.append("\f")
    text = "\n".join(parts).replace("\f\n", "\f")
    return normalize_text(text)

# -------- MCQ extraction (stem + options + answers/explanations) --------
MCQ_START_PAT = re.compile(
    r"""^\s*(?:                                # start of line
            (?:Q(?:uestion)?\s*\d*)[.)\s:-]*   # 'Q' or 'Question' with optional number
          | \d+\s*[.)-]\s+                     # '12.' or '12)' or '12 -'
          )""",
    re.IGNORECASE | re.VERBOSE,
)

OPTION_PAT = re.compile(
    r"""^\s*([A-Ha-h])\s*[\).:,-]\s+""",  # A)  A.  A-  A:
    re.VERBOSE,
)

ANSWER_PAT = re.compile(
    r"""^\s*(?:answer|answers?|ans|correct\s*answer|key|solution)\s*[:\-]?\s*(.*)""",
    re.IGNORECASE | re.VERBOSE,
)

EXPL_PAT = re.compile(
    r"""^\s*(?:explanation|rationale|why|reason(?:ing)?)\s*[:\-]?\s*(.*)""",
    re.IGNORECASE | re.VERBOSE,
)

def looks_like_question(line: str) -> bool:
    return bool(MCQ_START_PAT.match(line))

def looks_like_option(line: str) -> bool:
    return bool(OPTION_PAT.match(line))

def parse_answer(line: str):
    m = ANSWER_PAT.match(line)
    return m.group(1).strip() if m else None

def parse_expl(line: str):
    m = EXPL_PAT.match(line)
    return m.group(1).strip() if m else None

def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()

def extract_mcqs_with_answers(full_text: str):
    """
    Returns a list of formatted blocks:
      Q... (stem)
      A) ...
      B) ...
      ...
      Answer: ...
      Explanation: ...
    """
    lines = [l.rstrip() for l in full_text.split("\n")]
    mcqs = []
    cur_stem = []
    cur_opts = []
    cur_answer = None
    cur_expl = None
    in_mcq = False
    options_started = False

    def flush():
        nonlocal cur_stem, cur_opts, cur_answer, cur_expl, in_mcq, options_started
        # keep only real MCQs (need stem + >=2 options)
        stem_text = " ".join([clean_line(s) for s in cur_stem]).strip()
        if stem_text and len(cur_opts) >= 2:
            block_lines = []
            block_lines.append(stem_text)
            block_lines.extend(cur_opts)
            if cur_answer:
                block_lines.append(f"Answer: {cur_answer}")
            if cur_expl:
                block_lines.append(f"Explanation: {cur_expl}")
            mcqs.append("\n".join(block_lines).strip())

        # reset
        cur_stem, cur_opts = [], []
        cur_answer, cur_expl = None, None
        in_mcq = False
        options_started = False

    for raw in lines:
        line = raw.strip()
        if not line:
            # blank line inside MCQ: separate paragraphs in stem
            if in_mcq and cur_stem and not options_started:
                cur_stem.append("")  # minor spacing
            continue

        if looks_like_question(line):
            # new MCQ begins
            if in_mcq:
                flush()
            in_mcq = True
            options_started = False
            cur_stem = [clean_line(line)]
            cur_opts = []
            cur_answer, cur_expl = None, None
            continue

        if not in_mcq:
            continue  # ignore non-MCQ text

        # within an MCQ:
        # options
        if looks_like_option(line):
            options_started = True
            cur_opts.append(clean_line(line))
            continue

        # answers/explanations
        ans = parse_answer(line)
        if ans is not None:
            cur_answer = ans or cur_answer or ""  # allow "Answer: A"
            continue

        expl = parse_expl(line)
        if expl is not None:
            # accumulate multiple explanation lines if they appear separated
            if cur_expl:
                cur_expl = (cur_expl + " " + expl).strip()
            else:
                cur_expl = expl
            continue

        # extra prose lines:
        if not options_started:
            # still part of stem (multi-line stem)
            cur_stem.append(clean_line(line))
        else:
            # after options: could be continued explanation without the word "Explanation"
            # if line is long and not a new question, treat as explanation continuation
            if cur_answer is not None or cur_expl is not None:
                cur_expl = (cur_expl + " " + clean_line(line)).strip() if cur_expl else clean_line(line)
            # else ignore unrelated prose after options

    # flush tail
    if in_mcq:
        flush()

    return mcqs

# -------- Pagination over MCQs --------
def mcqs_to_pages(mcqs, per_page: int):
    pages = []
    for i in range(0, len(mcqs), per_page):
        pages.append("\n\n".join(mcqs[i : i + per_page]))
    return pages or ["No MCQs (with options) found."]

# -------- TTS --------
def tts_mp3(text: str) -> BytesIO:
    step = 4500
    combined = BytesIO()
    for i in range(0, len(text), step):
        chunk = text[i : i + step]
        buf = BytesIO()
        gTTS(chunk, lang="en").write_to_fp(buf)
        buf.seek(0)
        combined.write(buf.read())
    combined.seek(0)
    return combined

# ---------- Sidebar ----------
st.title("📖 MCQs + Answers Reader (GitHub → TTS)")
st.caption("Reads only Multiple-Choice Questions, Options, Answers, and Explanations.")
with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    mcqs_per_page = st.slider("MCQs per page", 1, 10, 3, 1)
    st.markdown(
        "- Use a **raw** URL (`https://raw.githubusercontent.com/...`).\n"
        "- Works best with **.docx** or **.txt**."
    )

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

        mcqs = extract_mcqs_with_answers(full_text)
        pages = mcqs_to_pages(mcqs, mcqs_per_page)

        st.session_state.pages = pages
        st.session_state.page_idx = 0
        st.session_state.loaded_url = url

        if pages and pages[0].startswith("No MCQs"):
            st.warning("No MCQs with options detected. Ensure questions start with 'Q...' or '1.' and options like 'A) text'.")
    except Exception as e:
        st.error(f"Could not load/parse: {e}")
        st.stop()

# ---------- Navigation ----------
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
    if st.button(
        "Next ➡️",
        use_container_width=True,
        disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1,
    ):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("MCQs + Answers (Page)", page_text, height=480)

col_play, col_dl = st.columns([2, 1])
with col_play:
    if st.button("🔊 Read this page aloud"):
        try:
            with st.spinner("Generating audio..."):
                audio = tts_mp3(page_text)
            st.audio(audio, format="audio/mp3")
        except Exception as e:
            st.error(f"TTS failed: {e}")

with col_dl:
    st.download_button(
        "⬇️ Download this page (txt)",
        data=page_text.encode("utf-8"),
        file_name=f"mcqs_page_{st.session_state.page_idx+1}.txt",
        mime="text/plain",
    )

# ---------- Jump ----------
with st.expander("Jump to page"):
    idx = st.number_input(
        "Go to page #",
        min_value=1,
        max_value=len(st.session_state.pages),
        value=st.session_state.page_idx + 1,
        step=1,
    )
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
