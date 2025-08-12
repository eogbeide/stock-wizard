import streamlit as st
import pandas as pd
import io
import requests
import random
import zlib
from urllib.parse import urlsplit, urlunsplit, quote

# --------------------------------------------------------
# Config
# --------------------------------------------------------
st.set_page_config(page_title="Psych 180 MCQs", page_icon="🧠", layout="centered")

DOC_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Psych 180 Pages.doc"

# Simple mobile-friendly tweaks
st.markdown("""
<style>
/* Bigger touch targets on mobile */
button, .stButton>button {padding: 0.6rem 1rem; font-size: 1rem;}
label[data-baseweb="radio"] {line-height: 1.6;}
div[role="radiogroup"] label {padding: 0.25rem 0;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def encode_url(url: str) -> str:
    """URL-encode path parts (for spaces, etc.)."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, quote(parts.path), parts.query, parts.fragment))

def _norm(s):
    return str(s).strip()

def _lower(s):
    return _norm(s).lower()

def pick_first_present(names, cols_lower_map):
    """Return the first existing column name (original case) from candidate names (case-insensitive)."""
    for name in names:
        if name.lower() in cols_lower_map:
            return cols_lower_map[name.lower()]
    return None

def get_option_columns(df):
    """Detect option columns in order A..E, Option A..Option E, etc."""
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # Try A, B, C, D, E
    letters = ["a", "b", "c", "d", "e"]
    opts = []
    for l in letters:
        if l in lower_map:
            opts.append(lower_map[l])

    # If not found, try Option A..E
    if not opts:
        for l in letters:
            key = f"option {l}"
            if key in lower_map:
                opts.append(lower_map[key])

    # If still nothing, try any column that starts with "option"
    if not opts:
        for c in cols:
            if _lower(c).startswith("option"):
                opts.append(c)

    return opts

def figure_out_correct_idx(correct_val, options):
    """Return index of the correct option given a 'Correct' value that may be a letter or exact text."""
    if correct_val is None:
        return 0  # default fallback
    cv = _norm(correct_val)
    cv_lower = cv.lower()

    # Letter?
    letter_map = {"a":0, "b":1, "c":2, "d":3, "e":4}
    if cv_lower in letter_map and letter_map[cv_lower] < len(options):
        return letter_map[cv_lower]

    # Exact text match?
    for i, opt in enumerate(options):
        if _norm(opt).lower() == cv_lower:
            return i

    # Partial/loose match (contains)
    for i, opt in enumerate(options):
        if cv_lower in _norm(opt).lower():
            return i

    # Fallback to first option
    return 0

@st.cache_data(show_spinner=False)
def load_questions(url: str):
    """Load and normalize questions from Excel at URL."""
    # Fetch bytes (handles spaces in URL robustly)
    resp = requests.get(encode_url(url), timeout=30)
    resp.raise_for_status()
    data = io.BytesIO(resp.content)

    # Read Excel (needs xlrd for .xls)
    df = pd.read_excel(data)

    # Normalize column names
    orig_cols = list(df.columns)
    cols_lower_map = {c.lower(): c for c in orig_cols}

    q_col = pick_first_present(["Question", "Prompt", "Stem", "Item"], cols_lower_map)
    if not q_col:
        raise ValueError("Could not find a 'Question/Prompt/Stem' column.")

    correct_col = pick_first_present(["Correct", "Answer", "Key"], cols_lower_map)
    expl_col = pick_first_present(["Explanation", "Rationale", "Reason"], cols_lower_map)

    opt_cols = get_option_columns(df)
    if not opt_cols:
        raise ValueError("Could not find any option columns (e.g., A, B, C, D or Option A..).")

    questions = []
    for _, row in df.iterrows():
        q_text = _norm(row.get(q_col, ""))
        if not q_text:
            continue
        options = [row.get(c, "") for c in opt_cols]
        options = [o if pd.notna(o) else "" for o in options]
        options = [str(o).strip() for o in options if str(o).strip()]

        if len(options) < 2:
            # skip malformed rows
            continue

        correct_val = row.get(correct_col, None) if correct_col else None
        correct_idx = figure_out_correct_idx(correct_val, options)
        explanation = _norm(row.get(expl_col, "")) if expl_col else ""

        questions.append({
            "question": q_text,
            "options": options,
            "correct_idx": max(0, min(correct_idx, len(options)-1)),
            "explanation": explanation
        })

    if not questions:
        raise ValueError("No usable questions were found in the sheet.")

    return questions

def get_stable_shuffle(n: int, seed_key: str):
    """Return a stable permutation (list of indices 0..n-1) derived from seed_key."""
    seed = zlib.adler32(seed_key.encode("utf-8")) & 0xffffffff
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order

# --------------------------------------------------------
# App
# --------------------------------------------------------
st.title("🧠 Psych 180 MCQs")

try:
    qs = load_questions(DOC_URL)
except Exception as e:
    st.error(f"Failed to load questions: {e}")
    st.stop()

# Session state
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "shuffle_map" not in st.session_state:
    st.session_state.shuffle_map = {}   # q_idx -> order list
if "selection" not in st.session_state:
    st.session_state.selection = {}     # q_idx -> selected option index (in shuffled space)

total = len(qs)
st.caption(f"{total} questions loaded from: Psych 180 Pages")

# Navigation
c1, c2, c3 = st.columns([1,2,1])
with c1:
    if st.button("◀ Back", disabled=(st.session_state.q_idx == 0)):
        st.session_state.q_idx = max(0, st.session_state.q_idx - 1)
with c3:
    if st.button("Next ▶", disabled=(st.session_state.q_idx >= total - 1)):
        st.session_state.q_idx = min(total - 1, st.session_state.q_idx + 1)

# Current question
idx = st.session_state.q_idx
q = qs[idx]

# Stable per-question shuffle so options don't jump around on reruns
if idx not in st.session_state.shuffle_map:
    st.session_state.shuffle_map[idx] = get_stable_shuffle(len(q["options"]), seed_key=q["question"])

order = st.session_state.shuffle_map[idx]
shuffled_options = [q["options"][i] for i in order]
correct_shuffled_idx = order.index(q["correct_idx"])

st.subheader(f"Q{idx+1}. {q['question']}")
selected = st.radio(
    "Choose one:",
    options=list(range(len(shuffled_options))),
    format_func=lambda i: shuffled_options[i],
    index=st.session_state.selection.get(idx, 0) if idx in st.session_state.selection else 0,
    key=f"radio_{idx}"
)

# Persist selection
st.session_state.selection[idx] = selected

# Feedback
if selected == correct_shuffled_idx:
    st.success("✅ Correct!")
    if q["explanation"]:
        st.info(q["explanation"])
else:
    st.error("❌ Incorrect — try again.")

# Quick jump
with st.expander("Jump to question"):
    jump = st.slider("Question number", 1, total, idx+1)
    if jump - 1 != idx:
        st.session_state.q_idx = jump - 1
        st.rerun()
