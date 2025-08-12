import streamlit as st
import requests
import io
import re
import random
import zlib
from urllib.parse import urlsplit, urlunsplit, quote
from docx import Document  # pip install python-docx

# =========================
# Config
# =========================
st.set_page_config(page_title="Psych 180 MCQs", page_icon="🧠", layout="centered")

# Your DOCX URL (spaces auto-encoded by encode_url):
DOC_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Psych 180 Pages.docx"

# Mobile-friendly sizing
st.markdown("""
<style>
button, .stButton>button {padding: 0.65rem 1.1rem; font-size: 1rem;}
div[role="radiogroup"] label {padding: 0.25rem 0; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def encode_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, quote(parts.path), parts.query, parts.fragment))

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s).strip())

def is_question_start(line: str) -> bool:
    # e.g., "1. ...", "1) ...", "Q1 ...", "Question 1 ..."
    return bool(re.match(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*\S', line, flags=re.I))

def parse_option(line: str):
    """
    Return (letter, text) if line looks like an option, else None.
    Accepts formats: "A) text", "A. text", "(A) text", "A - text", "A: text"
    """
    m = re.match(r'^\s*\(?([A-Ea-e])\)?\s*[\.\):\-]\s*(\S.*)$', line)
    if m:
        return m.group(1).upper(), _norm(m.group(2))
    return None

def parse_answer_line(line: str):
    """Return answer value (letter or text) if line starts with Answer/Correct/Key."""
    m = re.match(r'^\s*(?:Answer|Ans|Key|Correct)\s*[:\-]\s*(\S.*)$', line, flags=re.I)
    if m:
        return _norm(m.group(1))
    return None

def parse_expl_start(line: str):
    """Return initial explanation text if line starts with Explanation/Rationale/Why/Exp."""
    m = re.match(r'^\s*(?:Explanation|Rationale|Why|Exp)\s*[:\-]\s*(.*)$', line, flags=re.I)
    if m:
        return _norm(m.group(1))
    return None

def figure_out_correct_idx(correct_val, options):
    """Map 'A'..'E' or exact text to an index within options."""
    if correct_val is None:
        return 0
    cv = _norm(correct_val)
    letter_map = {"A":0,"B":1,"C":2,"D":3,"E":4}
    if cv.upper() in letter_map and letter_map[cv.upper()] < len(options):
        return letter_map[cv.upper()]
    # exact match
    for i, opt in enumerate(options):
        if _norm(opt).lower() == cv.lower():
            return i
    # contains match
    for i, opt in enumerate(options):
        if cv.lower() in _norm(opt).lower():
            return i
    return 0

def stable_shuffle(n: int, seed_key: str):
    """Stable permutation for each question (so options don't jump around)."""
    seed = zlib.adler32(seed_key.encode("utf-8")) & 0xffffffff
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order

# =========================
# Load & Parse DOCX
# =========================
@st.cache_data(show_spinner=True)
def load_questions(doc_url: str):
    resp = requests.get(encode_url(doc_url), timeout=30)
    resp.raise_for_status()
    doc = Document(io.BytesIO(resp.content))

    # Flatten text lines (paragraphs first)
    lines = []
    for p in doc.paragraphs:
        t = _norm(p.text)
        if t or (lines and lines[-1] != ""):
            lines.append(t if t else "")

    # If nothing usable found, also scan tables
    if not any(lines):
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        t = _norm(p.text)
                        if t or (lines and lines[-1] != ""):
                            lines.append(t if t else "")

    # Parse state
    questions = []
    q_text_parts = []
    options = []
    correct_val = None
    explanation_parts = []
    collecting_expl = False

    def flush_current():
        if q_text_parts and options:
            q_text = _norm(" ".join(q_text_parts))
            correct_idx = figure_out_correct_idx(correct_val, options)
            questions.append({
                "question": q_text,
                "options": options[:],
                "correct_idx": max(0, min(correct_idx, len(options)-1)),
                "explanation": _norm(" ".join(explanation_parts)) if explanation_parts else ""
            })

    for raw in lines:
        line = raw.strip()

        # Explanation collection mode
        if collecting_expl:
            if is_question_start(line):
                flush_current()
                q_text_parts = [re.sub(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*', '', line, flags=re.I)]
                options, correct_val, explanation_parts = [], None, []
                collecting_expl = False
            else:
                explanation_parts.append(line)
            continue

        # New question?
        if is_question_start(line):
            if q_text_parts or options:
                flush_current()
                q_text_parts, options, correct_val, explanation_parts = [], [], None, []
            q_text_parts = [re.sub(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*', '', line, flags=re.I)]
            continue

        # Option line?
        opt = parse_option(line)
        if opt:
            _, text = opt
            options.append(text)
            continue

        # Answer line?
        ans = parse_answer_line(line)
        if ans is not None:
            correct_val = ans
            continue

        # Explanation start?
        expl0 = parse_expl_start(line)
        if expl0 is not None:
            collecting_expl = True
            explanation_parts = [expl0] if expl0 else []
            continue

        # Empty line — soft separator
        if line == "":
            continue

        # Otherwise: continuation of stem or wrapped option
        if options:
            options[-1] = _norm(options[-1] + " " + line)
        else:
            q_text_parts.append(line)

    # Flush last one
    flush_current()

    if not questions:
        raise ValueError(
            "No usable questions were parsed. Ensure numbered questions (e.g., '1.'), "
            "options like 'A) ...', an 'Answer:' line, and optional 'Explanation:'."
        )

    return questions

# =========================
# App UI
# =========================
st.title("🧠 Psych 180 MCQs (from .docx)")

doc_url = st.text_input("DOCX URL", DOC_URL)
if not doc_url:
    st.stop()

try:
    qs = load_questions(doc_url)
except Exception as e:
    st.error(f"Failed to load/parse DOCX: {e}")
    st.stop()

# Session state
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "shuffle_map" not in st.session_state:
    st.session_state.shuffle_map = {}
if "selection" not in st.session_state:
    # stores the selected VALUE per question:
    #   -1 for "no selection yet", or 0..n-1 for chosen option index (in shuffled space)
    st.session_state.selection = {}

total = len(qs)
st.caption(f"Loaded {total} question(s) from Psych 180 Pages")

# Navigation
cols = st.columns([1,2,1])
with cols[0]:
    st.button("◀ Back", disabled=(st.session_state.q_idx == 0),
              on_click=lambda: st.session_state.update(q_idx=max(0, st.session_state.q_idx - 1)))
with cols[2]:
    st.button("Next ▶", disabled=(st.session_state.q_idx >= total - 1),
              on_click=lambda: st.session_state.update(q_idx=min(total - 1, st.session_state.q_idx + 1)))

idx = st.session_state.q_idx
q = qs[idx]

# Stable shuffle of options per question
if idx not in st.session_state.shuffle_map:
    st.session_state.shuffle_map[idx] = stable_shuffle(len(q["options"]), seed_key=q["question"])
order = st.session_state.shuffle_map[idx]
shuffled_opts = [q["options"][i] for i in order]
correct_shuffled_idx = order.index(q["correct_idx"])

st.subheader(f"Q{idx+1}. {q['question']}")

# ----- RADIO WIDGET (fixed) -----
# Use a sentinel option at POSITION 0, but store/compare using its VALUE (-1).
SENTINEL_VALUE = -1
options_for_radio = [SENTINEL_VALUE] + list(range(len(shuffled_opts)))

def format_opt(val: int) -> str:
    return "— Select an answer —" if val == SENTINEL_VALUE else shuffled_opts[val]

# Determine the RADIO INDEX (position) from the previously saved VALUE
prev_value = st.session_state.selection.get(idx, SENTINEL_VALUE)
try:
    default_radio_index = options_for_radio.index(prev_value)
except ValueError:
    default_radio_index = 0  # fallback safely inside [0, len(options)-1]

selected_value = st.radio(
    "Choose one:",
    options=options_for_radio,           # values: [-1, 0, 1, 2, ...]
    format_func=format_opt,
    index=default_radio_index,           # <-- POSITION within options list (0..n-1)
    key=f"radio_{idx}"
)

# Persist the VALUE (-1 or 0..n-1) for this question
st.session_state.selection[idx] = selected_value

# Feedback (only once user has actually selected an option)
if selected_value != SENTINEL_VALUE:
    if selected_value == correct_shuffled_idx:
        st.success("✅ Correct!")
        if q.get("explanation"):
            st.info(q["explanation"])
    else:
        st.error("❌ Incorrect — try again.")

with st.expander("Jump to question"):
    jump = st.slider("Question number", 1, total, idx+1)
    if jump - 1 != idx:
        st.session_state.q_idx = jump - 1
        st.rerun()
