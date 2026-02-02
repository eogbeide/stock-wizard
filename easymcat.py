# easymcat.py
# Streamlit DOCX Reader + Subject/Topic sidebar + Text-to-Speech + Next/Back
#
# Run:
#   streamlit run easymcat.py
#
# Default DOCX URL:
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Exam_Crackers.docx"

import io
import re
from typing import Optional, List, Dict, Tuple

import requests
import streamlit as st
from docx import Document


# -----------------------------
# DOCX parsing helpers
# -----------------------------
SUBJECT_RE = re.compile(r"^\s*subject\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)
TOPIC_RE = re.compile(r"^\s*topic\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)


def heading_level(style_name: str) -> Optional[int]:
    """Return heading level if style_name looks like 'Heading 1', 'Heading 2', etc."""
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


def parse_docx_to_structure(docx_bytes: bytes) -> List[Dict]:
    """
    Build structure:
    subjects = [
      {"subject": str, "topics": [{"topic": str, "chunks": [str], "full_text": str}, ...]},
      ...
    ]

    Priority rules:
      1) If a paragraph line looks like "Subject: ..." -> Subject marker
      2) If a paragraph line looks like "Topic: ..."   -> Topic marker
      3) Else fall back to styles:
         - Heading 1 => Subject
         - Heading 2 => Topic
      4) Otherwise, append paragraph text to current Topic.
    """
    doc = Document(io.BytesIO(docx_bytes))

    subjects: List[Dict] = []
    cur_subject: Optional[Dict] = None
    cur_topic: Optional[Dict] = None

    def ensure_subject(name: str):
        nonlocal cur_subject, cur_topic
        subj = {"subject": (name.strip() or "Untitled Subject"), "topics": []}
        subjects.append(subj)
        cur_subject = subj
        cur_topic = None

    def ensure_topic(name: str):
        nonlocal cur_subject, cur_topic
        if cur_subject is None:
            ensure_subject("General")
        top = {"topic": (name.strip() or "Untitled Topic"), "chunks": [], "full_text": ""}
        cur_subject["topics"].append(top)
        cur_topic = top

    for p in doc.paragraphs:
        raw = (p.text or "").strip()
        if not raw:
            continue

        # 1) Explicit markers inside the document text (highest priority)
        sm = SUBJECT_RE.match(raw)
        if sm:
            ensure_subject(sm.group(1))
            continue

        tm = TOPIC_RE.match(raw)
        if tm:
            ensure_topic(tm.group(1))
            continue

        # 2) Fallback to DOCX heading styles
        lvl = heading_level(getattr(p.style, "name", ""))
        if lvl == 1:
            ensure_subject(raw)
            continue
        if lvl == 2:
            ensure_topic(raw)
            continue

        # 3) Regular content lines -> attach to current topic
        if cur_topic is None:
            ensure_topic("Overview")
        cur_topic["chunks"].append(raw)

    # finalize full_text
    if not subjects:
        subjects = [{"subject": "Document", "topics": [{"topic": "Content", "chunks": [], "full_text": ""}]}]

    for subj in subjects:
        if not subj.get("topics"):
            subj["topics"] = [{"topic": "Overview", "chunks": [], "full_text": ""}]
        for top in subj["topics"]:
            top["full_text"] = "\n\n".join(top.get("chunks", [])).strip()

    return subjects


def flatten_topics(subjects: List[Dict]) -> List[Tuple[int, int]]:
    flat: List[Tuple[int, int]] = []
    for si, subj in enumerate(subjects):
        for ti, _ in enumerate(subj["topics"]):
            flat.append((si, ti))
    return flat


# -----------------------------
# Browser TTS (Web Speech API)
# -----------------------------
def tts_component(text: str, voice_lang: str = "en-US", rate: float = 1.0, pitch: float = 1.0):
    """Text-to-speech in the user's browser (no API keys) using Web Speech API."""
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

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
      const text = `{safe}`;
      const playBtn = document.getElementById("tts_play");
      const stopBtn = document.getElementById("tts_stop");

      function speak(t) {{
        if (!("speechSynthesis" in window)) {{
          alert("Your browser doesn't support speech synthesis.");
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
    st.components.v1.html(html, height=80)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="DOCX Study Reader", layout="wide")
st.title("DOCX Study Reader")
st.caption("Sidebar: Subject → Topic • Page: content + Text-to-Speech + Next/Back")


@st.cache_data(show_spinner=True)
def load_structure_from_url(url: str) -> List[Dict]:
    b = fetch_docx_bytes(url)
    return parse_docx_to_structure(b)


with st.sidebar:
    st.header("Document Source")
    url = st.text_input("DOCX URL", value=DEFAULT_URL)
    st.write("This app will scan for lines like **Subject:** and **Topic:** in the DOCX.")


# Load document
try:
    subjects = load_structure_from_url(url)
except Exception as e:
    st.error(f"Could not load DOCX.\n\nError: {e}")
    st.stop()

flat = flatten_topics(subjects)
if not flat:
    st.warning("No topics found in the document.")
    st.stop()

# Session state
if "flat_index" not in st.session_state:
    st.session_state.flat_index = 0

# Current indices
cur_si, cur_ti = flat[st.session_state.flat_index]

# Sidebar navigation + TTS settings
with st.sidebar:
    st.header("Navigate")

    subject_names = [s["subject"] for s in subjects]
    selected_subject = st.selectbox("Subject", subject_names, index=cur_si)
    new_si = subject_names.index(selected_subject)

    topic_names = [t["topic"] for t in subjects[new_si]["topics"]]
    default_topic_index = cur_ti if new_si == cur_si and cur_ti < len(topic_names) else 0
    selected_topic = st.selectbox("Topic", topic_names, index=default_topic_index)

    if st.button("Go", use_container_width=True):
        new_ti = topic_names.index(selected_topic)
        for idx, (si, ti) in enumerate(flat):
            if si == new_si and ti == new_ti:
                st.session_state.flat_index = idx
                break
        st.rerun()

    st.divider()
    st.subheader("Text-to-Speech")

    voice_lang = st.selectbox("Voice language", ["en-US", "en-GB", "en", "es-ES", "fr-FR"], index=0)
    rate = st.slider("Rate", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)

# Refresh current content after potential jump
cur_si, cur_ti = flat[st.session_state.flat_index]
cur_subject = subjects[cur_si]["subject"]
cur_topic = subjects[cur_si]["topics"][cur_ti]["topic"]
cur_text = (subjects[cur_si]["topics"][cur_ti].get("full_text") or "").strip()

# Layout
col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    st.subheader(f"{cur_subject}  →  {cur_topic}")

    if cur_text:
        st.write(cur_text)
    else:
        st.info("No paragraph text under this topic.")

    st.divider()

    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        if st.button("⬅️ Back", disabled=(st.session_state.flat_index == 0), use_container_width=True):
            st.session_state.flat_index -= 1
            st.rerun()
    with b2:
        if st.button("Next ➡️", disabled=(st.session_state.flat_index == len(flat) - 1), use_container_width=True):
            st.session_state.flat_index += 1
            st.rerun()

with col_right:
    st.subheader("Listen")
    if cur_text:
        tts_component(cur_text, voice_lang=voice_lang, rate=rate, pitch=pitch)
    else:
        st.caption("Nothing to read for this topic.")

    st.divider()
    st.caption("Progress")
    st.write(f"Topic {st.session_state.flat_index + 1} of {len(flat)}")
