# lab.py — Show only Questions + Answers + Explanations (Options excluded) + Sidebar page dropdown + TTS
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
st.set_page_config(page_title="📖 Lab Reader — Q + A + Explanation (Options removed)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/BIO.docx"
DEFAULT_ITEMS_PER_PAGE = 5

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    # keep ALL content; just standardize newlines and compress long blank gaps a bit
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # cap at max 3 consecutive blanks
    return text.strip()

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text_with_breaks(data: bytes) -> str:
    """
    Extract ALL text from .docx, inserting \f for explicit page breaks.
    We do not drop passages; we only filter later to exclude options.
    """
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

# ---------- Patterns ----------
MCQ_START_PAT = re.compile(
    r"""^\s*(?:
            (?:Q(?:uestion)?\s*\d*)[.)\s:-]* |   # Question, Question 12) etc
            \d+\s*[.)-]\s+                       # 1)  2.  3-
        )""",
    re.IGNORECASE | re.VERBOSE,
)
OPTION_PAT = re.compile(r"""^\s*([A-Ha-h])\s*[\).:,-]\s+""", re.VERBOSE)  # A)  B.  c:
ANSWER_PAT = re.compile(r"""^\s*(?:answer|answers?|ans|correct\s*answer|key|solution)\s*[:\-]?\s*(.*)""", re.IGNORECASE)
EXPL_PAT   = re.compile(r"""^\s*(?:explanation|rationale|why|reason(?:ing)?)\s*[:\-]?\s*(.*)""", re.IGNORECASE)

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

# ---------- Extract ONLY Question + Answer + Explanation (Options are excluded) ----------
def extract_qae_only(full_text: str):
    """
    Build items with ONLY:
      - Question: <stem text>
      - Answer: <answer text as written>  (no options shown)
      - Explanation: <free text>          (optional)
    We do not remove passages; we simply ignore option lines like "A) ...".
    """
    lines = [l.rstrip() for l in full_text.split("\n")]

    items = []
    cur_stem_parts = []
    cur_answer = None
    cur_expl_parts = []

    in_mcq = False
    options_seen = False
    capturing_expl = False  # on after Explanation: or after Answer:

    def flush():
        nonlocal cur_stem_parts, cur_answer, cur_expl_parts, in_mcq, options_seen, capturing_expl
        stem_text = " ".join([clean_line(s) for s in cur_stem_parts if s is not None]).strip()
        expl_text = " ".join([clean_line(s) for s in cur_expl_parts if s is not None]).strip()

        if stem_text:
            block = [f"Question: {stem_text}"]
            if cur_answer:
                block.append(f"Answer: {cur_answer}")
            if expl_text:
                block.append(f"Explanation: {expl_text}")
            items.append("\n".join(block).strip())

        # reset
        cur_stem_parts = []
        cur_answer = None
        cur_expl_parts = []
        in_mcq = False
        options_seen = False
        capturing_expl = False

    for raw in lines:
        line = raw.strip()

        # preserve paragraph breaks within stem/expl (as blank lines)
        if not line:
            if in_mcq:
                if not options_seen and not capturing_expl and cur_stem_parts:
                    cur_stem_parts.append("")
                elif capturing_expl and cur_expl_parts:
                    cur_expl_parts.append("")
            continue

        # New question starts
        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq = True
            options_seen = False
            capturing_expl = False
            cur_stem_parts = [clean_line(line)]
            cur_answer = None
            cur_expl_parts = []
            continue

        if not in_mcq:
            # outside questions → ignore
            continue

        # Option line? (we EXCLUDE options entirely)
        if OPTION_PAT.match(line):
            options_seen = True
            continue

        # Answer line?
        ans = parse_answer(line)
        if ans is not None:
            cur_answer = ans if ans != "" else cur_answer
            capturing_expl = True   # often explanation follows answer
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
        elif not options_seen:
            cur_stem_parts.append(clean_line(line))
        else:
            # Text after options but before explicit Explanation: may be distractors; skip
            pass

    if in_mcq:
        flush()

    return items

def items_to_pages(items, per_page: int):
    pages = []
    for i in range(0, len(items), per_page):
        pages.append("\n\n".join(items[i:i+per_page]))
    return pages or ["No questions found. Ensure questions begin with 'Q...' or '1.' and answers use 'Answer:'."]

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

# ---------- UI ----------
st.title("📖 Lab Reader — Question + Answer + Explanation (Options removed)")
st.caption("Parses your document and shows only Questions, Answers, and Explanations. Options are excluded. Passages are kept.")

# ---------- Sidebar ----------
with st.sidebar:
    url = st.text_input("GitHub RAW .docx/.txt URL", value=DEFAULT_URL)
    per_page = st.slider("Items per page", 1, 12, DEFAULT_ITEMS_PER_PAGE, 1)
    st.markdown(
        "- Use a **raw** URL (`https://raw.githubusercontent.com/...`).\n"
        "- Best with **.docx** or plain text that uses lines like **Answer:** and **Explanation:**."
    )

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# ---------- Load & extract ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching and parsing document..."):
            data = fetch_bytes(url)
            if url.lower().endswith(".docx"):
                full_text = extract_docx_text_with_breaks(data)
            else:
                full_text = normalize_text(best_effort_bytes_to_text(data))

            items = extract_qae_only(full_text)
            pages = items_to_pages(items, per_page)

            st.session_state.pages = pages
            st.session_state.page_idx = 0
            st.session_state.loaded_url = url
    except Exception as e:
        st.error(f"Could not load or parse the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found after parsing.")
    st.stop()

# ---------- Sidebar dropdown page selector ----------
with st.sidebar:
    total_pages = len(st.session_state.pages)
    page_labels = [f"Page {i}" for i in range(1, total_pages + 1)]
    current_index = min(st.session_state.page_idx, total_pages - 1)
    selected_label = st.selectbox("Select page", options=page_labels, index=current_index)
    st.session_state.page_idx = page_labels.index(selected_label)

# ---------- Navigation (optional quick buttons) ----------
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
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("Questions + Answers + Explanations (Options removed)", page_text, height=520)

# ---------- TTS & Download ----------
col_play, col_dl = st.columns([2, 1])
with col_play:
    if st.button("🔊 Read this page aloud"):
        try:
            with st.spinner("Generating audio..."):
                st.audio(tts_mp3(page_text), format="audio/mp3")
        except Exception as e:
            st.error(f"TTS failed: {e}")

with col_dl:
    st.download_button(
        "⬇️ Download this page (txt)",
        data=page_text.encode("utf-8"),
        file_name=f"page_{st.session_state.page_idx+1}.txt",
        mime="text/plain",
    )
