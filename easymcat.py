# app.py
# Streamlit DOCX Reader + Subject/Topic sidebar + Text-to-Speech + Next/Back
#
# Run:
#   streamlit run app.py
#
# Notes:
# - Subjects/Topics are inferred from DOCX headings:
#     Heading 1 => Subject
#     Heading 2 => Topic
#   If your doc doesn't use headings, it will fall back to a single Subject/Topic.
# - TTS uses the browser via embedded Web Speech API (no external keys required).

import io
import re
import requests
import streamlit as st
from docx import Document

DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Exam_Crackers.docx"


# -----------------------------
# DOCX parsing
# -----------------------------
def _is_heading(style_name: str) -> int | None:
    """
    Return heading level if style_name looks like 'Heading 1', 'Heading 2', etc.
    Otherwise return None.
    """
    if not style_name:
        return None
    m = re.match(r"Heading\s+(\d+)", str(style_name).strip(), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def fetch_docx_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def parse_docx_to_structure(docx_bytes: bytes):
    """
    Returns a nested structure:

    subjects = [
      {
        "subject": "Subject Name",
        "topics": [
          {
            "topic": "Topic Name",
            "chunks": [ "para1", "para2", ... ],
            "full_text": "..."
          },
          ...
        ]
      },
      ...
    ]
    """
    doc = Document(io.BytesIO(docx_bytes))

    subjects = []
    cur_subject = None
    cur_topic = None

    def ensure_subject(name: str):
        nonlocal subjects, cur_subject
        subj = {"subject": name.strip() or "Untitled Subject", "topics": []}
        subjects.append(subj)
        cur_subject = subj

    def ensure_topic(name: str):
        nonlocal cur_subject, cur_topic
        if cur_subject is None:
            ensure_subject("General")
        top = {
            "topic": name.strip() or "Untitled Topic",
            "chunks": [],
            "full_text": "",
        }
        cur_subject["topics"].append(top)
        cur_topic = top

    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue

        lvl = _is_heading(getattr(p.style, "name", ""))
        if lvl == 1:
            ensure_subject(text)
            cur_topic = None
        elif lvl == 2:
            ensure_topic(text)
        else:
            # regular text goes into current topic; if none, create a default topic
            if cur_topic is None:
                ensure_topic("Overview")
            cur_topic["chunks"].append(text)

    # finalize full_text
    for subj in subjects:
        for top in subj["topics"]:
            top["full_text"] = "\n\n".join(top["chunks"]).strip()

    # Fallback if doc had no headings and no paragraphs captured
    if not subjects:
        subjects = [
            {
                "subject": "Document",
                "topics": [{"topic": "Content", "chunks": [], "full_text": ""}],
            }
        ]

    # If headings existed but a subject has zero topics, make a placeholder
    for subj in subjects:
        if not subj["topics"]:
            subj["topics"] = [{"topic": "Overview", "chunks": [], "full_text": ""}]

    return subjects


def flatten_topics(subjects):
    """
    Create a flat ordered list of (subj_idx, topic_idx).
    """
    flat = []
    for si, subj in enumerate(subjects):
        for ti, _ in enumerate(subj["topics"]):
            flat.append((si, ti))
    return flat


# -----------------------------
# Browser TTS (Web Speech API)
# -----------------------------
def tts_component(text: str, voice_lang: str = "en-US", rate: float = 1.0, pitch: float = 1.0):
    """
    Uses a small HTML/JS snippet that runs in the user's browser.
    """
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <button id="tts_play" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
        ▶️ Play
      </button>
      <button id="tts_stop" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
        ⏹ Stop
      </button>
      <span style="color:#666; font-size: 0.9rem;">(Uses your browser's text-to-speech)</span>
    </div>

    <script>
      const text = `{safe_text}`;
      const playBtn = document.getElementById("tts_play");
      const stopBtn = document.getElementById("tts_stop");

      function speak(t) {{
        if (!("speechSynthesis" in window)) {{
          alert("Sorry—your browser doesn't support speech synthesis.");
          return;
        }}
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(t);
        utter.lang = "{voice_lang}";
        utter.rate = {rate};
        utter.pitch = {pitch};
        window.speechSynthesis.speak(utter);
      }}

      playBtn.addEventListener("click", () => speak(text));
      stopBtn.addEventListener("click", () => {{
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
      }});
    </script>
    """
    st.components.v1.html(html, height=70)


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="DOCX Study Reader", layout="wide")

st.title("DOCX Study Reader")
st.caption("Sidebar: Subject → Topic • Main page: reading view + Text-to-Speech + Next/Back")

# Sidebar: document source
with st.sidebar:
    st.header("Document Source")
    url = st.text_input("DOCX URL", value=DEFAULT_URL)
    st.write("Tip: Use headings in the DOCX (Heading 1/2) for best Subject/Topic grouping.")

# Load / cache document
@st.cache_data(show_spinner=True)
def load_structure(url_: str):
    b = fetch_docx_bytes(url_)
    return parse_docx_to_structure(b)

try:
    subjects = load_structure(url)
except Exception as e:
    st.error(f"Could not load DOCX from URL.\n\nError: {e}")
    st.stop()

flat = flatten_topics(subjects)
if not flat:
    st.warning("No topics found in the document.")
    st.stop()

# Session state for navigation
if "flat_index" not in st.session_state:
    st.session_state.flat_index = 0

# Sidebar navigation controls
with st.sidebar:
    st.header("Navigate")

    # Build options for subjects and topics
    subject_names = [s["subject"] for s in subjects]
    # Determine current subject/topic from flat_index
    cur_si, cur_ti = flat[st.session_state.flat_index]

    selected_subject = st.selectbox("Subject", subject_names, index=cur_si)

    # When subject changes, default to first topic in that subject
    new_si = subject_names.index(selected_subject)
    topic_names = [t["topic"] for t in subjects[new_si]["topics"]]
    # pick current topic index if same subject, else 0
    default_topic_index = cur_ti if new_si == cur_si and cur_ti < len(topic_names) else 0
    selected_topic = st.selectbox("Topic", topic_names, index=default_topic_index)

    # Jump button
    if st.button("Go", use_container_width=True):
        new_ti = topic_names.index(selected_topic)
        # find corresponding flat index
        for idx, (si, ti) in enumerate(flat):
            if si == new_si and ti == new_ti:
                st.session_state.flat_index = idx
                break
        st.rerun()

    st.divider()

    # TTS settings
    st.subheader("Text-to-Speech")
    voice_lang =_
