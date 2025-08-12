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

# Replace with your real DOCX raw URL (spaces auto-encoded):
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

def _norm(s):
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
                # next question begins
                flush_current()
                # start new question
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
            # strip the numbering (e.g., "1. ", "Q1) ")
            q_text_parts = [re.sub(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*', '', line, flags=re.I)]
            continue

        # Option line?
        opt = parse_option(line)
        if opt:
            letter, text = opt
            # If options already collected and we see another letter that jumps (OK), just append.
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

        # Empty line — treat as soft separator
        if line == "":
            # If we have question + options but no explanation, keep going;
            # some docs separate blocks with blank lines.
            # Also allow multi-line stems/options before/after empties.
            continue

        # Otherwise: it's either part of the stem or a wrapped option; decide:
        if options:
            # Likely continuation of the previous option (wrapped line)
            options[-1] = _norm(options[-1] + " " + line)
        else:
            q_text_parts.append(line)

    # Flush last one
    flush_current()

    if not questions:
        raise ValueError("No usable questions were parsed. Make sure your DOCX uses numbered questions and options like 'A) ...', and includes an 'Answer:' line (plus optional 'Explanation:').")

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
    st.session_state.selection = {}  # q_idx -> selected shuffled index (or -1)

total = len(qs)
st.caption(f"Loaded {total} question(s) from Psych 180 Pages")

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

# Add a "select prompt" sentinel so we don't show error before the user chooses
sentinel = -1
options_for_radio = [sentinel] + list(range(len(shuffled_opts)))

def format_opt(i):
    return "— Select an answer —" if i == sentinel else shuffled_opts[i]

selected = st.radio(
    "Choose one:",
    options=options_for_radio,
    format_func=format_opt,
    index=st.session_state.selection.get(idx, sentinel),
    key=f"radio_{idx}"
)

# Persist selection
st.session_state.selection[idx] = selected

# Feedback (only once user has selected something)
if selected != sentinel:
    if selected == correct_shuffled_idx:
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
