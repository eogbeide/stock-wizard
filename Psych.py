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

def classify_meta(text: str):
    """
    Detect 'Answer:', 'Explanation:'/'Why:'/'Exp:', 'Rationale:' or 'Eliminate ...'
    Return ('answer', value) | ('explanation', txt) | ('rationale', txt) | None
    """
    t = _norm(text)
    m = re.match(r'^(?:Answer|Ans|Key|Correct)\s*[:\-]\s*(\S.*)$', t, flags=re.I)
    if m:
        return ('answer', _norm(m.group(1)))
    m = re.match(r'^(?:Explanation|Why|Exp)\s*[:\-]\s*(.*)$', t, flags=re.I)
    if m:
        return ('explanation', _norm(m.group(1)))
    m = re.match(r'^(?:Rationale)\s*[:\-]\s*(.*)$', t, flags=re.I)
    if m:
        return ('rationale', _norm(m.group(1)))
    # Many docs write rationale without the word "Rationale", e.g., "Eliminate A and D because ..."
    if re.match(r'^(Eliminate|Because)\b', t, flags=re.I):
        return ('rationale', t)
    return None

def parse_inline_options(line: str):
    """
    Parse a line that may contain inline options:
      "Question: ... A. text B) text C: text D - text F. Answer: C Explanation: ..."

    Returns: (leading_text, parts)
      - leading_text: text before first option marker (belongs to the question stem)
      - parts: list of tuples [('A', '...'), ('B','...'), ...]  (letters kept if present)
              meta segments like 'Answer:' / 'Explanation:' / 'Rationale:' are returned as
              [('META', 'answer: C')] etc. We'll classify later.
    """
    parts = []
    pat = re.compile(r'\(?([A-Fa-f])\)?\s*[\.\):\-]\s*')
    matches = list(pat.finditer(line))
    if not matches:
        return (_norm(line), [])  # no options here

    # Leading stem before first marker
    lead = _norm(line[:matches[0].start()])

    # Segments from each marker to the next
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(line)
        seg = _norm(line[start:end])
        if seg:
            parts.append((letter, seg))
    return (lead, parts)

def parse_single_option(line: str):
    """Parse a single option line like 'A) text' or '(B) text'."""
    m = re.match(r'^\s*\(?([A-Fa-f])\)?\s*[\.\):\-]\s*(\S.*)$', line)
    if m:
        return (m.group(1).upper(), _norm(m.group(2)))
    return None

def figure_out_correct_idx(correct_val, letters, texts):
    """Map 'B' or exact/contains text to an index among texts."""
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

def finalize_letters_texts(collected):
    """
    Keep ONLY the first four real options in A–D order.
    'collected' is list like [('A','...'), ('B','...'), ...].
    Drop anything with letters beyond 'D'.
    If letters missing, fill sequentially A..D.
    """
    real = [(L, T) for (L, T) in collected if L in ('A','B','C','D')]
    # If some A-D are missing but we have unlabeled options (None), fill them in order.
    out_letters, out_texts = [], []
    next_code = ord('A')
    for L, T in real[:4]:
        # enforce A..D order by their natural letter order
        out_letters.append(L)
        out_texts.append(T)
    # If fewer than 4 found, stop with what we have
    return out_letters, out_texts

# =========================
# Load & Parse DOCX
# =========================
@st.cache_data(show_spinner=True)
def load_questions(doc_url: str):
    resp = requests.get(encode_url(doc_url), timeout=30)
    resp.raise_for_status()
    doc = Document(io.BytesIO(resp.content))

    # Gather lines from paragraphs (fallback to tables if needed)
    lines = []
    for p in doc.paragraphs:
        t = _norm(p.text)
        if t or (lines and lines[-1] != ""):
            lines.append(t if t else "")
    if not any(lines):
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        t = _norm(p.text)
                        if t or (lines and lines[-1] != ""):
                            lines.append(t if t else "")

    questions = []
    q_text_parts = []
    option_pairs = []  # [('A','text'), ('B','text'), ...]
    correct_val = None
    explanation_parts = []
    rationale_parts = []
    collecting_expl = False
    collecting_rat = False

    def flush():
        if q_text_parts and option_pairs:
            letters, texts = finalize_letters_texts(option_pairs)
            if not letters or not texts:
                return
            q_text = _norm(" ".join(q_text_parts))
            # common doc style: "Question: ..." — strip that tag from stem
            q_text = re.sub(r'^\s*(?:Question)\s*[:\-]\s*', '', q_text, flags=re.I)
            correct_idx = figure_out_correct_idx(correct_val, letters, texts)
            questions.append({
                "question": q_text,
                "letters": letters,
                "options": texts,
                "correct_idx": max(0, min(correct_idx, len(texts)-1)),
                "explanation": _norm(" ".join(explanation_parts)) if explanation_parts else "",
                "rationale": _norm(" ".join(rationale_parts)) if rationale_parts else ""
            })

    for raw in lines:
        line = raw.strip()

        # Explanation and Rationale collection blocks
        if collecting_expl:
            if is_question_start(line):
                flush()
                q_text_parts, option_pairs = [strip_qnum(line)], []
                correct_val, explanation_parts, rationale_parts = None, [], []
                collecting_expl = collecting_rat = False
                continue
            # Switch to rationale midstream?
            meta = classify_meta(line)
            if meta and meta[0] == 'rationale':
                collecting_expl = False
                collecting_rat = True
                if meta[1]:
                    rationale_parts.append(meta[1])
                continue
            if line:
                explanation_parts.append(line)
            continue

        if collecting_rat:
            if is_question_start(line):
                flush()
                q_text_parts, option_pairs = [strip_qnum(line)], []
                correct_val, explanation_parts, rationale_parts = None, [], []
                collecting_expl = collecting_rat = False
                continue
            if line:
                rationale_parts.append(line)
            continue

        # Start of a new question?
        if is_question_start(line):
            if q_text_parts or option_pairs:
                flush()
                q_text_parts, option_pairs = [], []
                correct_val, explanation_parts, rationale_parts = None, [], []
            q_text_parts = [strip_qnum(line)]
            continue

        # Inline options?
        lead, parts = parse_inline_options(line)
        if parts:
            if lead:
                q_text_parts.append(lead)
            for L, seg in parts:
                # If the "option" is actually meta (Answer/Explanation/Rationale), capture it
                meta = classify_meta(seg)
                if meta:
                    kind, val = meta
                    if kind == 'answer':
                        correct_val = val
                    elif kind == 'explanation':
                        collecting_expl = True
                        if val:
                            explanation_parts.append(val)
                    elif kind == 'rationale':
                        collecting_rat = True
                        if val:
                            rationale_parts.append(val)
                    continue
                # Keep only A-D as options
                if L in ('A','B','C','D'):
                    option_pairs.append((L, seg))
            continue

        # Single option per line?
        opt = parse_single_option(line)
        if opt:
            L, seg = opt
            meta = classify_meta(seg)
            if meta:
                kind, val = meta
                if kind == 'answer':
                    correct_val = val
                elif kind == 'explanation':
                    collecting_expl = True
                    if val:
                        explanation_parts.append(val)
                elif kind == 'rationale':
                    collecting_rat = True
                    if val:
                        rationale_parts.append(val)
            else:
                if L in ('A','B','C','D'):
                    option_pairs.append((L, seg))
            continue

        # Explicit Answer / Explanation / Rationale lines (no A/B/C prefix)
        meta = classify_meta(line)
        if meta:
            kind, val = meta
            if kind == 'answer':
                correct_val = val
            elif kind == 'explanation':
                collecting_expl = True
                if val:
                    explanation_parts.append(val)
            elif kind == 'rationale':
                collecting_rat = True
                if val:
                    rationale_parts.append(val)
            continue

        # Empty line → soft separator
        if line == "":
            continue

        # Otherwise, more stem text OR option continuation (we only continue stem here to avoid mis-capturing)
        q_text_parts.append(line)

    # Flush last block
    flush()

    if not questions:
        raise ValueError(
            "No usable questions parsed. Use numbered questions (e.g., '1.'), "
            "A–D options (inline or one per line), and an 'Answer:' line. "
            "Optional 'Explanation:' and 'Rationale:' supported."
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
    st.session_state.selection = {}       # per-question selected VALUE: -1 or 0..3
if "submitted" not in st.session_state:
    st.session_state.submitted = {}       # per-question submitted flag
if "checked_value" not in st.session_state:
    st.session_state.checked_value = {}   # per-question last submitted VALUE

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

# Question text (separate from options)
st.subheader(f"Q{idx+1}. {q['question']}")

# Build radio list with exactly A–D that were parsed
labels = [f"{q['letters'][i]}. {q['options'][i]}" for i in range(len(q['options']))]

# Radio with sentinel until user chooses
SENTINEL = -1
values = [SENTINEL] + list(range(len(labels)))  # [-1, 0, 1, 2, 3]
prev = st.session_state.selection.get(idx, SENTINEL)
try:
    default_index = values.index(prev)
except ValueError:
    default_index = 0

selected_value = st.radio(
    "Choose one:",
    options=values,
    format_func=lambda v: "— Select an answer —" if v == SENTINEL else labels[v],
    index=default_index,
    key=f"radio_{idx}"
)

# Persist current choice
st.session_state.selection[idx] = selected_value

# If user changes selection after submit, require re-submit
if st.session_state.submitted.get(idx, False):
    if st.session_state.checked_value.get(idx, SENTINEL) != selected_value:
        st.session_state.submitted[idx] = False
        st.info("Selection changed — click Submit to check.")

# Submit
if st.button("Submit", key=f"submit_{idx}"):
    st.session_state.submitted[idx] = True
    st.session_state.checked_value[idx] = st.session_state.selection.get(idx, SENTINEL)

# Result after submission
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
