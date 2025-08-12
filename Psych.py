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

# Default DOCX URL (auto-encodes spaces)
DEFAULT_DOC_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Psych 180 Pages.docx"

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
    return re.sub(r'\s+', ' ', str(s or "").strip())

def is_question_start(line: str) -> bool:
    # "1. ...", "1) ...", "Q1 ...", "Question 1 ..."
    return bool(re.match(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*\S', line, flags=re.I))

def strip_qnum(line: str) -> str:
    return re.sub(r'^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\)]\s*', '', line, flags=re.I)

def parse_multiple_options_inline(line: str):
    """
    Parse a single line that may contain multiple options like:
    "A. text B) text C: text D - text"
    Returns list of tuples: [(letter, text), ...]
    """
    results = []
    # find occurrences of an option marker (A-E + punctuation)
    pat = re.compile(r'\(?([A-Ea-e])\)?\s*[\.\):\-]\s*')
    matches = list(pat.finditer(line))
    if not matches:
        return results
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(line)
        text = _norm(line[start:end])
        if text:
            results.append((letter, text))
    return results

def parse_option_line(line: str):
    """
    Parse a line that's just one option, e.g.:
    "A) text" / "A. text" / "(A) text" / "A - text" / "A: text"
    """
    m = re.match(r'^\s*\(?([A-Ea-e])\)?\s*[\.\):\-]\s*(\S.*)$', line)
    if m:
        return [(m.group(1).upper(), _norm(m.group(2)))]
    return []

def figure_out_correct_idx(correct_val, letters, texts):
    """
    Map 'B' or exact option text to an index within current letters/texts.
    letters: ['A','B','C','D',...]
    texts:   ['opt text', ...]   (same length/order)
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
    """Fill in missing letters in sequence A, B, C, ... keeping given letters if present."""
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

    # Collect text lines from paragraphs first
    lines = []
    for p in doc.paragraphs:
        t = _norm(p.text)
        if t or (lines and lines[-1] != ""):
            lines.append(t if t else "")

    # If nothing, also check tables
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
    opts_accum = []           # list of (letter or None, text)
    correct_val = None
    explanation_parts = []
    collecting_expl = False

    def flush_current():
        if q_text_parts and opts_accum:
            letters, texts = assign_missing_letters(opts_accum)
            correct_idx = figure_out_correct_idx(correct_val, letters, texts)
            questions.append({
                "question": _norm(" ".join(q_text_parts)),
                "letters": letters,
                "options": texts,
                "correct_idx": max(0, min(correct_idx, len(texts)-1)),
                "explanation": _norm(" ".join(explanation_parts)) if explanation_parts else ""
            })

    for raw in lines:
        line = raw.strip()

        # If we're collecting explanation and a new question starts, flush & start over
        if collecting_expl:
            if is_question_start(line):
                flush_current()
                q_text_parts = [strip_qnum(line)]
                opts_accum, correct_val, explanation_parts = [], None, []
                collecting_expl = False
            else:
                if line != "":
                    explanation_parts.append(line)
            continue

        # New question start?
        if is_question_start(line):
            if q_text_parts or opts_accum:
                flush_current()
                q_text_parts, opts_accum, correct_val, explanation_parts = [], [], None, []
            q_text_parts = [strip_qnum(line)]
            continue

        # Options inline on a single line?
        inline_opts = parse_multiple_options_inline(line)
        if inline_opts:
            opts_accum.extend(inline_opts)
            continue

        # Single option on this line?
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
        m_exp = re.match(r'^\s*(?:Explanation|Rationale|Why|Exp)\s*[:\-]\s*(.*)$', line, flags=re.I)
        if m_exp:
            collecting_expl = True
            first = _norm(m_exp.group(1))
            explanation_parts = [first] if first else []
            continue

        # Empty line → soft separator
        if line == "":
            continue

        # Otherwise it's probably stem continuation or wrapped option continuation
        if opts_accum:
            # append to last option's text
            last_letter, last_text = opts_accum[-1]
            opts_accum[-1] = (last_letter, _norm(last_text + " " + line))
        else:
            q_text_parts.append(line)

    # flush last
    flush_current()

    if not questions:
        raise ValueError(
            "No usable questions parsed. Ensure numbered questions (e.g., '1.'), "
            "options like 'A) ...' OR inline 'A. ... B. ...', an 'Answer:' line, "
            "and optional 'Explanation:'."
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
    # For each question, store selected index (0..n-1) or -1 for none
    st.session_state.selection = {}

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

# Build radio choices with letter prefixes (A., B., C., D.)
labels = [f"{q['letters'][i]}. {q['options'][i]}" for i in range(len(q['options']))]

# Sentinel handling so we don't error before the user picks
SENTINEL = -1
saved_value = st.session_state.selection.get(idx, SENTINEL)

# Compute default index (0..n) for radio widget: if none selected, default to 0 but show a placeholder
radio_options = list(range(len(labels)))  # values: 0..n-1
default_index = saved_value if saved_value in radio_options else 0

# A top placeholder label so users must actively choose? (Optional)
# If you prefer a visible placeholder, use a selectbox instead.
selected_idx = st.radio(
    "Choose one:",
    options=radio_options,
    format_func=lambda i: labels[i],
    index=default_index,
    key=f"radio_{idx}"
)

# Persist selection
st.session_state.selection[idx] = selected_idx

# Feedback
if selected_idx != SENTINEL:
    if selected_idx == q["correct_idx"]:
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
