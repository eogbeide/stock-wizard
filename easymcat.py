# easymcat.py
# Streamlit DOCX Reader + Subject dropdown + Topic dropdown + Subtopic dropdown
# + Text-to-Speech + Next/Back
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
SUBTOPIC_RE = re.compile(r"^\s*sub\s*topic\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)
SUBTOPIC_RE2 = re.compile(r"^\s*subtopic\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)


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
    Structure:
    subjects = [
      {
        "subject": str, "real": bool,
        "topics": [
          {
            "topic": str, "real": bool,
            "subtopics": [
              {"subtopic": str, "real": bool, "chunks": [str], "full_text": str}
            ]
          }
        ]
      }
    ]

    Preference:
      - If the doc contains ANY Subject:/Topic:/Subtopic: markers, those define navigation.
      - Otherwise fallback to Heading 1/2/3 as Subject/Topic/Subtopic.
    """
    doc = Document(io.BytesIO(docx_bytes))

    # detect explicit markers anywhere
    has_markers = False
    for p in doc.paragraphs:
        raw = (p.text or "").strip()
        if not raw:
            continue
        if SUBJECT_RE.match(raw) or TOPIC_RE.match(raw) or SUBTOPIC_RE.match(raw) or SUBTOPIC_RE2.match(raw):
            has_markers = True
            break

    subjects: List[Dict] = []
    cur_subject: Optional[Dict] = None
    cur_topic: Optional[Dict] = None
    cur_subtopic: Optional[Dict] = None

    def ensure_subject(name: str, real: bool = True):
        nonlocal cur_subject, cur_topic, cur_subtopic
        subj = {"subject": (name.strip() or "Untitled Subject"), "topics": [], "real": real}
        subjects.append(subj)
        cur_subject = subj
        cur_topic = None
        cur_subtopic = None

    def ensure_topic(name: str, real: bool = True):
        nonlocal cur_subject, cur_topic, cur_subtopic
        if cur_subject is None:
            ensure_subject("General", real=False)
        top = {"topic": (name.strip() or "Untitled Topic"), "subtopics": [], "real": real}
        cur_subject["topics"].append(top)
        cur_topic = top
        cur_subtopic = None

    def ensure_subtopic(name: str, real: bool = True):
        nonlocal cur_subject, cur_topic, cur_subtopic
        if cur_subject is None:
            ensure_subject("General", real=False)
        if cur_topic is None:
            ensure_topic("Overview", real=False)
        sub = {"subtopic": (name.strip() or "Untitled Subtopic"), "chunks": [], "full_text": "", "real": real}
        cur_topic["subtopics"].append(sub)
        cur_subtopic = sub

    for p in doc.paragraphs:
        raw = (p.text or "").strip()
        if not raw:
            continue

        if has_markers:
            sm = SUBJECT_RE.match(raw)
            if sm:
                ensure_subject(sm.group(1), real=True)
                continue

            tm = TOPIC_RE.match(raw)
            if tm:
                ensure_topic(tm.group(1), real=True)
                continue

            stm = SUBTOPIC_RE.match(raw) or SUBTOPIC_RE2.match(raw)
            if stm:
                ensure_subtopic(stm.group(1), real=True)
                continue

            # Attach content only when we're inside a real subtopic (or at least an existing one)
            if cur_subtopic is not None:
                cur_subtopic["chunks"].append(raw)
            # If there's no subtopic yet, we ignore content in markers-mode (keeps sidebar "document-only")
            continue

        # fallback: headings
        lvl = heading_level(getattr(p.style, "name", ""))
        if lvl == 1:
            ensure_subject(raw, real=True)
            continue
        if lvl == 2:
            ensure_topic(raw, real=True)
            continue
        if lvl == 3:
            ensure_subtopic(raw, real=True)
            continue

        # Regular content lines: attach to current subtopic; create minimal placeholders in fallback mode
        if cur_subtopic is None:
            ensure_subtopic("Overview", real=False)
        cur_subtopic["chunks"].append(raw)

    # If empty doc parsing fallback
    if not subjects:
        subjects = [
            {
                "subject": "Document",
                "real": True,
                "topics": [
                    {
                        "topic": "Content",
                        "real": True,
                        "subtopics": [{"subtopic": "Content", "real": True, "chunks": [], "full_text": ""}],
                    }
                ],
            }
        ]

    # Finalize full_text and ensure minimal structure
    for subj in subjects:
        if not subj.get("topics"):
            subj["real"] = False
            subj["topics"] = [{"topic": "Overview", "real": False, "subtopics": []}]

        for top in subj["topics"]:
            if not top.get("subtopics"):
                top["real"] = False
                top["subtopics"] = [{"subtopic": "Overview", "real": False, "chunks": [], "full_text": ""}]

            for sub in top["subtopics"]:
                sub["full_text"] = "\n\n".join(sub.get("chunks", [])).strip()

    return subjects


def build_navigation(subjects: List[Dict]) -> Tuple[List[Dict], List[Tuple[int, int, int]]]:
    """
    Returns:
      nav: list of dicts with indexes to real subjects/topics/subtopics:
        [
          {
            "si": int, "subject": str,
            "topics": [
              {"ti": int, "topic": str, "subtopics": [{"ui": int, "subtopic": str}, ...]}
            ]
          },
          ...
        ]
      flat: [(si, ti, ui), ...]  # real subtopics only, for Next/Back
    """
    nav: List[Dict] = []
    flat: List[Tuple[int, int, int]] = []

    for si, subj in enumerate(subjects):
        subj_topics = []
        for ti, top in enumerate(subj.get("topics", [])):
            sub_list = []
            for ui, sub in enumerate(top.get("subtopics", [])):
                if sub.get("real", False):
                    sub_list.append({"ui": ui, "subtopic": sub.get("subtopic", f"Subtopic {ui+1}")})
            if sub_list and top.get("real", False):
                subj_topics.append({"ti": ti, "topic": top.get("topic", f"Topic {ti+1}"), "subtopics": sub_list})

        if subj_topics and subj.get("real", False):
            nav.append({"si": si, "subject": subj.get("subject", f"Subject {si+1}"), "topics": subj_topics})
            for t in subj_topics:
                for s in t["subtopics"]:
                    flat.append((si, t["ti"], s["ui"]))

    return nav, flat


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
st.caption("Sidebar: Subject → Topic → Subtopic • Page: content + Text-to-Speech + Next/Back")


@st.cache_data(show_spinner=True)
def load_structure_from_url(url: str) -> List[Dict]:
    return parse_docx_to_structure(fetch_docx_bytes(url))


with st.sidebar:
    st.header("Document Source")
    url = st.text_input("DOCX URL", value=DEFAULT_URL)
    st.write(
        "Navigation comes from **Subject:** / **Topic:** / **Subtopic:** lines (if present), "
        "otherwise Heading 1/2/3."
    )

# Load document
try:
    subjects = load_structure_from_url(url)
except Exception as e:
    st.error(f"Could not load DOCX.\n\nError: {e}")
    st.stop()

nav, flat = build_navigation(subjects)

if not nav or not flat:
    st.warning(
        "No usable Subject/Topic/Subtopic triples found.\n\n"
        "Ensure your DOCX includes lines like:\n"
        "- Subject: ...\n"
        "- Topic: ...\n"
        "- Subtopic: ...\n"
        "with text beneath each Subtopic.\n\n"
        "Or use Heading 1/2/3 styles."
    )
    st.stop()

# Session state
if "flat_index" not in st.session_state:
    st.session_state.flat_index = 0
st.session_state.flat_index = max(0, min(st.session_state.flat_index, len(flat) - 1))

# Current position
cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]

# -----------------------------
# Sidebar: Subject dropdown (1st) + Topic dropdown (2nd) + Subtopic dropdown (3rd)
# -----------------------------
with st.sidebar:
    st.header("Navigate")

    # ---- Subject dropdown ----
    subject_options = [x["subject"] for x in nav]
    cur_subject_nav_idx = next(i for i, x in enumerate(nav) if x["si"] == cur_si)
    selected_subject_name = st.selectbox("Subject", subject_options, index=cur_subject_nav_idx)
    subj_nav_idx = subject_options.index(selected_subject_name)
    subj_node = nav[subj_nav_idx]
    new_si = subj_node["si"]

    # ---- Topic dropdown ----
    topic_options = [t["topic"] for t in subj_node["topics"]]
    # default topic based on current selection if same subject
    if new_si == cur_si:
        cur_topic_nav_idx = next(
            (i for i, t in enumerate(subj_node["topics"]) if t["ti"] == cur_ti),
            0
        )
    else:
        cur_topic_nav_idx = 0

    selected_topic_name = st.selectbox("Topic", topic_options, index=cur_topic_nav_idx)
    topic_nav_idx = topic_options.index(selected_topic_name)
    topic_node = subj_node["topics"][topic_nav_idx]
    new_ti = topic_node["ti"]

    # ---- Subtopic dropdown ----
    subtopic_options = [s["subtopic"] for s in topic_node["subtopics"]]
    if new_si == cur_si and new_ti == cur_ti:
        cur_sub_nav_idx = next(
            (i for i, s in enumerate(topic_node["subtopics"]) if s["ui"] == cur_ui),
            0
        )
    else:
        cur_sub_nav_idx = 0

    selected_subtopic_name = st.selectbox("Subtopic", subtopic_options, index=cur_sub_nav_idx)
    sub_nav_idx = subtopic_options.index(selected_subtopic_name)
    new_ui = topic_node["subtopics"][sub_nav_idx]["ui"]

    if st.button("Go", use_container_width=True):
        for idx, (si, ti, ui) in enumerate(flat):
            if si == new_si and ti == new_ti and ui == new_ui:
                st.session_state.flat_index = idx
                break
        st.rerun()

    st.divider()
    st.subheader("Text-to-Speech")
    voice_lang = st.selectbox("Voice language", ["en-US", "en-GB", "en", "es-ES", "fr-FR"], index=0)
    rate = st.slider("Rate", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)

# Current content
cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]
cur_subject = subjects[cur_si]["subject"]
cur_topic = subjects[cur_si]["topics"][cur_ti]["topic"]
cur_subtopic = subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui]["subtopic"]
cur_text = (subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui].get("full_text") or "").strip()

# Layout
col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    st.subheader(f"{cur_subject}  →  {cur_topic}  →  {cur_subtopic}")

    if cur_text:
        st.write(cur_text)
    else:
        st.info("No paragraph text under this subtopic.")

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
        st.caption("Nothing to read for this subtopic.")

    st.divider()
    st.caption("Progress")
    st.write(f"Section {st.session_state.flat_index + 1} of {len(flat)}")
