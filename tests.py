import streamlit as stcdd
import pandas as pd
import re
import streamlit.components.v1 as components

# --- Load & cache data ---
@st.cache_data
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/tests.xlsx"
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
  .question-block {
    margin: 1em 0;
    padding: 0.75em;
    background: #eef6ff;
    border-left: 4px solid #0050b3;
  }
  .options-block {
    margin: 0.5em 0 1em 1em;
  }
  .options-block li {
    margin-bottom: 0.5em;
  }
  .explanation-block {
    margin: 1em 0;
    padding: 1em;
    background: #f5f5f5;
    border-left: 4px solid #0078d4;
  }
  .explanation-block p {
    margin-bottom: 1em;
  }
</style>
""", height=0)

# --- Sidebar: Subject & Item selector ---
st.sidebar.title("Quiz Navigation")
subject = st.sidebar.selectbox("Subject", df["Subject"].unique())
subset = df[df["Subject"] == subject].reset_index(drop=True)

count = len(subset)
if count == 0:
    st.sidebar.warning("No items for this subject.")
    st.stop()

labels = [f"Item {n+1}" for n in range(count)]
default = st.session_state.get("idx", 0)
default = max(0, min(default, count - 1))
sel = st.sidebar.selectbox("Go to item", labels, index=default)
st.session_state.idx = labels.index(sel)
i = st.session_state.idx

# --- Helper: split text into paragraphs ---
def to_para_html(text: str) -> str:
    parts = re.split(r"\n\s*\n", text.strip())
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in parts if p)

# --- Render selected item ---
row = subset.iloc[i]

# Question
question = str(row.get("Question","") or "").strip()
if question:
    st.markdown(f'<div class="question-block">{to_para_html(question)}</div>', unsafe_allow_html=True)

# Options
answers = [a.strip() for a in str(row.get("Answer","") or "").split(";")]
if answers and answers != ['']:
    opts_html = "".join(f"<li>{opt}</li>" for opt in answers)
    st.markdown(f'<ul class="options-block">{opts_html}</ul>', unsafe_allow_html=True)

# Explanation
exp = str(row.get("Explanation","") or "").strip()
if exp:
    st.markdown(f'<div class="explanation-block">{to_para_html(exp)}</div>', unsafe_allow_html=True)

# --- Navigation buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("◀ Back") and i > 0:
        st.session_state.idx = i - 1
with col2:
    if st.button("Next ▶") and i < count - 1:
        st.session_state.idx = i + 1
