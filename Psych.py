import streamlit as st
import requests
import io
import re
from urllib.parse import urlsplit, urlunsplit, quote
from docx import Document  # pip install python-docx

# =========================
# Config
# =========================
st.set_page_config(page_title="Psych 180 MCQs", page_icon="🧠", layout="centered")

# Default DOCX URL (spaces auto-encoded)
DEFAULT_DOC_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Psych 180 Pages.docx"

# Mobile-friendly tweaks
st.markdown("""
<style>
button, .stButton>button {padding: 0.65rem 1.1rem; font-size: 1rem;}
div[role="radiogroup"] label {padding: 0.25rem 0; line-height: 1.6;}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def encode_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, quote(parts.path), parts.query, parts.fragment))

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s or "").strip())

def is_question_start(line: str) -> bool:
    # "1. ...", "1) ...", "Q1 ...", "Question 1 ..."
    return bool(re.match(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*\S', line, flags=re.I))

def strip_qnum(line: str) -> str:
    return re.sub(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*', '', line, flags=re.I)

def parse_multiple_options_inline(line: str):
    """
    Parse a line like: "A. text B) text C: text D - text"
    Returns list of tuples: [(letter, text), ...]
    Also captures any leading segment *before* the first marker as option A.
    """
    results = []
    pat = re.compile(r'\(?([A-Ea-e])\)?\s*[\.\):\-]\s*')
    matches = list(pat.finditer(line))
    if not matches:
        return results

    # Leading text before first marker -> treat as first option if present
    if matches[0].start() > 0:
        lead = _norm(line[:matches[0].start()])
        if lead:
            results.append((None, lead))  # letter assigned later (likely "A")

    # Subsequent marked options
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(line)
        text = _norm(line[start:end])
        if text:
            results.append((letter, text))
    return results

def parse_option_line(line: str):
    """Parse a single option line: 'A) text' / 'A. text' / '(A) text' / 'A - text' / 'A: text'."""
    m = re.match(r'^\s*\(?([A-Ea-e])\)?\s*[\.\):\-]\s*(\S.*)$', line)
    if m:
        return [(m.group(1).upper(), _norm(m.group(2)))]
    return []

def figure_out_correct_idx(correct_val, letters, texts):
    """
    Map 'B' or exact option text to an index within current letters/texts.
    letters: ['A','B','C','D',...]
    texts:   ['opt text', ...]
    """
    if correct_val is None:
        return 0
    cv = _norm(correct_val)

    # Letter key?
    if re.fullmatch(r'[A-Ea-e]', cv):
        letter = cv.upper()
        if letter in letters:
            return letters.index(letter)

    # Exact text match?
    for i, t in enumerate(texts):
        if _norm(t).lower() == cv.lower():
            return i

    # Contains match
    for i, t in enumerate(texts):
        if cv.lower() in _norm(t).lower():
            return i

    return 0

def assign_missing_letters(options):
    """Fill in missing letters sequentially A, B, C, ... keeping given letters if present."""
    out_letters, out_texts = [], []
    next_code = ord('A')
    for letter, text in options:
        if letter is None or not re.fullmatch(r'[A-E]', letter):
            letter = chr(next_code)
        out_letters.append(letter)
        out_texts.append(text)
        next_code = ord('A') + len(out_letters)
    return out_letters, out_texts

# =========================
# Load & Parse DOCX
# =========================
@st.cache_data(show_spinner=True)
def load_questions(doc_url: str):
    resp = requests.get(encode_url(doc_url), timeout=30)
    resp.raise_for_status()
    doc = Document(io.BytesIO(resp.content))

    # Gather lines from paragraphs
    lines = []
    for p in doc.paragraphs:
        t = _norm(p.text)
        if t or (lines and lines[-1] != ""):
            lines.append(t if t else "")

    # If nothing, also scan tables
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
    opts_accum = []            # list[(letter or None, text)]
    correct_val = None
    explanation_parts = []
    rationale_parts = []
    collecting_expl = False
    collecting_rat = False

    def flush_current():
        if q_text_parts and opts_accum:
            letters, texts = assign_missing_letters(opts_accum)
            correct_idx = figure_out_correct_idx(correct_val, letters, texts)
            questions.append({
                "question": _norm(" ".join(q_text_parts)),
                "letters": letters,
                "options": texts,
                "correct_idx": max(0, min(correct_idx, len(texts)-1)),
                "explanation": _norm(" ".join(explanation_parts)) if explanation_parts else "",
                "rationale": _norm(" ".join(rationale_parts)) if rationale_parts else ""
            })

    for raw in lines:
        line = raw.strip()

        # Collecting Explanation
        if collecting_expl:
            # Switch to next question?
            if is_question_start(line):
                flush_current()
                q_text_parts = [strip_qnum(line)]
                opts_accum, correct_val = [], None
                explanation_parts, rationale_parts = [], []
                collecting_expl = collecting_rat = False
                continue
            # Switch to Rationale?
            m_rat0 = re.match(r'^\s*(?:Rationale)\s*[:\-]\s*(.*)$', line, flags=re.I)
            if m_rat0:
                collecting_expl = False
                collecting_rat = True
                first = _norm(m_rat0.group(1))
                rationale_parts = [first] if first else []
                continue
            # Continue explanation
            if line != "":
                explanation_parts.append(line)
            continue

        # Collecting Rationale
        if collecting_rat:
            if is_question_start(line):
                flush_current()
                q_text_parts = [strip_qnum(line)]
                opts_accum, correct_val = [], None
                explanation_parts, rationale_parts = [], []
                collecting_expl = collecting_rat = False
                continue
            if line != "":
                rationale_parts.append(line)
            continue

        # Start of a new question?
        if is_question_start(line):
            if q_text_parts or opts_accum:
                flush_current()
                q_text_parts, opts_accum = [], []
                correct_val = None
                explanation_parts, rationale_parts = [], []
            q_text_parts = [strip_qnum(line)]
            continue

        # Options inline on a single line?
        inline_opts = parse_multiple_options_inline(line)
        if inline_opts:
            opts_accum.extend(inline_opts)
            continue

        # Single option line?
        one_opt = parse_option_line(line)
        if one_opt:
            opts_accum.extend(one_opt)
            continue

        # Answer line
        m_ans = re.match(r'^\s*(?:Answer|Ans|Key|Correct)\s*[:\-]\s*(\S.*)$', line, flags=re.I)
        if m_ans:
            correct_val = _norm(m_ans.group(1))
            continue

        # Explanation start
        m_exp = re.match(r'^\s*(?:Explanation|Why|Exp)\s*[:\-]\s*(.*)$', line, flags=re.I)
        if m_exp:
            collecting_expl = True
            first = _norm(m_exp.group(1))
            explanation_parts = [first] if first else []
            continue

        # Rationale start
        m_rat = re.match(r'^\s*(?:Rationale)\s*[:\-]\s*(.*)$', line, flags=re.I)
        if m_rat:
            collecting_rat = True
            first = _norm(m_rat.group(1))
            rationale_parts = [first] if first else []
            continue

        # Empty line → soft separator
        if line == "":
            continue

        # Otherwise: continuation of stem or wrapped option continuation
        if opts_accum:
            last_letter, last_text = opts_accum[-1]
            opts_accum[-1] = (last_letter, _norm(last_text + " " + line))
        else:
            q_text_parts.append(line)

    # Flush last block
    flush_current()

    if not questions:
        raise ValueError(
            "No usable questions parsed. Ensure numbered questions (e.g., '1.'), "
            "options like 'A) ...' OR inline 'A. ... B. ...', an 'Answer:' line, "
            "and optional 'Explanation:' and 'Rationale:' lines."
        )

    return questions

# =========================
# App UI
# =========================
st.title("🧠 Psych 180 MCQs (from .docx)")
doc_url = st.text_input("DOCX URL", DEFAULT_DOC_URL)
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
if "selection" not in st.session_state:
    # per-question selected VALUE: -1 for none, else 0..n-1
    st.session_state.selection = {}
if "submitted" not in st.session_state:
    # per-question submission boolean
    st.session_state.submitted = {}
if "checked_value" not in st.session_state:
    # per-question last submitted VALUE (for stable feedback)
    st.session_state.checked_value = {}

total = len(qs)
st.caption(f"Loaded {total} question(s) from Psych 180 Pages")

# Navigation
c1, c2, c3 = st.columns([1,2,1])
with c1:
    st.button("◀ Back", disabled=(st.session_state.q_idx == 0),
              on_click=lambda: st.session_state.update(q_idx=max(0, st.session_state.q_idx - 1)))
with c3:
    st.button("Next ▶", disabled=(st.session_state.q_idx >= total - 1),
              on_click=lambda: st.session_state.update(q_idx=min(total - 1, st.session_state.q_idx + 1)))

idx = st.session_state.q_idx
q = qs[idx]

st.subheader(f"Q{idx+1}. {q['question']}")

# Build radio with A/B/C/D labels
labels = [f"{q['letters'][i]}. {q['options'][i]}" for i in range(len(q['options']))]

# Radio with a placeholder sentinel so the user must choose
SENTINEL = -1
options_values = [SENTINEL] + list(range(len(labels)))  # values: -1, 0..n-1

prev_value = st.session_state.selection.get(idx, SENTINEL)
try:
    default_radio_index = options_values.index(prev_value)
except ValueError:
    default_radio_index = 0

selected_value = st.radio(
    "Choose one:",
    options=options_values,
    format_func=lambda v: "— Select an answer —" if v == SENTINEL else labels[v],
    index=default_radio_index,
    key=f"radio_{idx}"
)

# Persist current choice
st.session_state.selection[idx] = selected_value

# If selection changed after submitting, require resubmission
if st.session_state.submitted.get(idx, False):
    last_checked = st.session_state.checked_value.get(idx, SENTINEL)
    if last_checked != selected_value:
        st.session_state.submitted[idx] = False
        st.info("Selection changed — click Submit to check.")

# Submit button
if st.button("Submit", key=f"submit_{idx}"):
    st.session_state.submitted[idx] = True
    st.session_state.checked_value[idx] = st.session_state.selection.get(idx, SENTINEL)

# Show result ONLY after submission
if st.session_state.submitted.get(idx, False):
    checked = st.session_state.checked_value.get(idx, SENTINEL)
    if checked == SENTINEL:
        st.warning("Please select an option before submitting.")
    else:
        if checked == q["correct_idx"]:
            st.success("✅ Correct!")
            if q.get("explanation"):
                st.markdown(f"**Explanation:** {q['explanation']}")
            if q.get("rationale"):
                st.markdown(f"**Rationale:** {q['rationale']}")
        else:
            st.error("❌ Incorrect — try again.")

with st.expander("Jump to question"):
    jump = st.slider("Question number", 1, total, idx+1)
    if jump - 1 != idx:
        st.session_state.q_idx = jump - 1
        st.rerun()
