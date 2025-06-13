import streamlit as st
import pandas as pd
import re
import streamlit.components.v1 as components

# --- Load & cache data ---
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"
    try:
        return pd.read_excel(url)
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.sidebar.warning("No data available.")
    st.stop()

# --- Global CSS ---
components.html("""
<style>
  .passage {
    padding: 1em;
    background: #f9f9f9;
    border-left: 4px solid #0078d4;
    margin-bottom: 1em;
  }
  .question {
    margin: 1em 0 0.5em;
    font-weight: bold;
    font-size: 1.1em;
  }
  .options {
    margin: 0.5em 0 1em 1em;
    list-style-type: disc;
  }
  .explanation {
    margin: 1em 0;
    padding: 1em;
    background: #eef8e8;
    border-left: 4px solid #40a860;
  }
  .explanation p {
    margin-bottom: 1em;
  }
</style>
""", height=0)

# --- Sidebar: Subject selector ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
subset = df[df["Subject"] == subject].reset_index(drop=True)

# --- Sidebar: Item selector with clamped default index ---
count = len(subset)
if count == 0:
    st.sidebar.warning("No items for this subject.")
    st.stop()

labels = [f"Item {n+1}" for n in range(count)]
default_idx = st.session_state.get("idx", 0)
# Clamp to valid range
default_idx = max(0, min(default_idx, count - 1))

sel = st.sidebar.selectbox("Go to item", labels, index=default_idx)
st.session_state.idx = labels.index(sel)
i = st.session_state.idx

# --- Helper: split text into <p> paragraphs ---
def to_para_html(text: str) -> str:
    paras = re.split(r"\n\s*\n", text.strip())
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paras if p)

# --- Render selected item ---
row = subset.iloc[i]

# Optional Passage
passage = str(row.get("Passage","") or "").strip()
if passage:
    st.markdown(f'<div class="passage">{to_para_html(passage)}</div>', unsafe_allow_html=True)

# Question
question = str(row.get("Question","") or "").strip()
if question:
    st.markdown(f'<div class="question">{to_para_html(question)}</div>', unsafe_allow_html=True)

# Options
answers = [a.strip() for a in str(row.get("Answer","") or "").split(";")]
if answers and answers != ['']:
    opts_html = "".join(f"<li>{opt}</li>" for opt in answers)
    st.markdown(f'<ul class="options">{opts_html}</ul>', unsafe_allow_html=True)

# Explanation
exp = str(row.get("Explanation","") or "").strip()
if exp:
    if st.checkbox("Show Explanation"):
        st.markdown(f'<div class="explanation">{to_para_html(exp)}</div>', unsafe_allow_html=True)

# --- Navigation buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx = i - 1
        st.experimental_rerun()
with col2:
    if st.button("Next ▶") and i < count - 1:
        st.session_state.idx = i + 1
        st.experimental_rerun()
