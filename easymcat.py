# easymcat.py
# -------------------------------------------------------------------
# Streamlit DOCX Study Reader
#   • Subject dropdown (1) → Topic dropdown (2) → Subtopic dropdown (3)
#   • Clean reading layout + Progress + Next/Back
#   • Browser Text-to-Speech (Web Speech API) — no keys required
#
# Run:
#   streamlit run easymcat.py
#
# Default DOCX URL:
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Exam_Crackers.docx"
# -------------------------------------------------------------------

import io
import re
from typing import Optional, List, Dict, Tuple

import requests
import streamlit as st
from docx import Document


# =========================================================
# Styling (lightweight, Streamlit-safe)
# =========================================================
st.set_page_config(page_title="DOCX Study Reader", layout="wide")

st.markdown(
    """
<style>
/* Make the app feel a bit more "study app" */
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
h1, h2, h3 {letter-spacing: -0.2px;}
/* Subtle “card” look for content */
.study-card {
  background: rgba(255,255,255,0.65);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 18px 18px 12px 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.breadcrumb {
  font-size: 0.92rem;
  color: rgba(0,0,0,0.60);
  margin-bottom: 0.3rem;
}
.small-muted {font-size: 0.9rem; color: rgba(0,0,0,0.55);}
hr {margin: 0.8rem 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📘 DOCX Study Reader")
st.caption("Pick **Subject → Topic → Subtopic** in the sidebar. Read on the left, listen on the right.")


# =========================================================
# DOCX parsing helpers
# =========================================================
SUBJECT_RE = re.compile(r"^\s*subject\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)
TOPIC_RE = re.compile(r"^\s*topic\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)
SUBTOPIC_RE = re.compile(r"^\s*(?:sub\s*topic|subtopic)\s*[:\-]\s*(.+?)\s*$", flags=re.IGNORECASE)


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
        "subject": str,
        "topics": [
          {
            "topic": str,
            "subtopics": [
              {"subtopic": str, "chunks": [str], "full_text": str}
            ],
          }
        ]
      }
    ]

    Preference:
      - If the doc contains ANY Subject:/Topic:/Subtopic: lines, those define navigation.
      - Otherwise fallback to Heading 1/2/3 for Subject/Topic/Subtopic.

    Robustness:
      - If content appears under a Topic before any Subtopic, create implicit Subtopic "Overview"
        so the 3rd dropdown always has something to show.
    """
    doc = Document(io.BytesIO(docx_bytes))

    # detect explicit markers anywhere
    has_markers = False
    for p in doc.paragraphs:
        raw = (p.text or "").strip()
        if not raw:
            continue
        if SUBJECT_RE.match(raw) or TOPIC_RE.match(raw) or SUBTOPIC_RE.match(raw):
            has_markers = True
            break

    subjects: List[Dict] = []
    cur_subject: Optional[Dict] = None
    cur_topic: Optional[Dict] = None
    cur_subtopic: Optional[Dict] = None

    def ensure_subject(name: str):
        nonlocal cur_subject, cur_topic, cur_subtopic
        subj = {"subject": (name.strip() or "Untitled Subject"), "topics": []}
        subjects.append(subj)
        cur_subject = subj
        cur_topic = None
        cur_subtopic = None

    def ensure_topic(name: str):
        nonlocal cur_subject, cur_topic, cur_subtopic
        if cur_subject is None:
            ensure_subject("General")
        top = {"topic": (name.strip() or "Untitled Topic"), "subtopics": []}
        cur_subject["topics"].append(top)
        cur_topic = top
        cur_subtopic = None

    def ensure_subtopic(name: str):
        nonlocal cur_subject, cur_topic, cur_subtopic
        if cur_subject is None:
            ensure_subject("General")
        if cur_topic is None:
            ensure_topic("Overview")
        sub = {"subtopic": (name.strip() or "Untitled Subtopic"), "chunks": [], "full_text": ""}
        cur_topic["subtopics"].append(sub)
        cur_subtopic = sub

    for p in doc.paragraphs:
        raw = (p.text or "").strip()
        if not raw:
            continue

        if has_markers:
            sm = SUBJECT_RE.match(raw)
            if sm:
                ensure_subject(sm.group(1))
                continue

            tm = TOPIC_RE.match(raw)
            if tm:
                ensure_topic(tm.group(1))
                continue

            stm = SUBTOPIC_RE.match(raw)
            if stm:
                ensure_subtopic(stm.group(1))
                continue

            # Content line
            if cur_topic is not None and cur_subtopic is None:
                ensure_subtopic("Overview")
            if cur_subtopic is not None:
                cur_subtopic["chunks"].append(raw)
            continue

        # fallback: headings
        lvl = heading_level(getattr(p.style, "name", ""))
        if lvl == 1:
            ensure_subject(raw)
            continue
        if lvl == 2:
            ensure_topic(raw)
            continue
        if lvl == 3:
            ensure_subtopic(raw)
            continue

        # regular content in fallback mode
        if cur_subtopic is None:
            ensure_subtopic("Overview")
        cur_subtopic["chunks"].append(raw)

    # finalize full_text
    if not subjects:
        subjects = [
            {
                "subject": "Document",
                "topics": [
                    {"topic": "Content", "subtopics": [{"subtopic": "Overview", "chunks": [], "full_text": ""}]}
                ],
            }
        ]

    for subj in subjects:
        if not subj.get("topics"):
            subj["topics"] = [{"topic": "Overview", "subtopics": [{"subtopic": "Overview", "chunks": [], "full_text": ""}]}]
        for top in subj["topics"]:
            if not top.get("subtopics"):
                top["subtopics"] = [{"subtopic": "Overview", "chunks": [], "full_text": ""}]
            for sub in top["subtopics"]:
                sub["full_text"] = "\n\n".join(sub.get("chunks", [])).strip()

    return subjects


def build_navigation(subjects: List[Dict]) -> Tuple[List[Dict], List[Tuple[int, int, int]]]:
    """
    nav = [
      {"si": int, "subject": str, "topics": [
        {"ti": int, "topic": str, "subtopics": [{"ui": int, "subtopic": str}, ...]}
      ]}
    ]
    flat = [(si, ti, ui), ...]  # for Next/Back across subtopics
    """
    nav: List[Dict] = []
    flat: List[Tuple[int, int, int]] = []

    for si, subj in enumerate(subjects):
        topics_nav = []
        for ti, top in enumerate(subj.get("topics", [])):
            subs = top.get("subtopics", [])
            if not subs:
                continue
            subs_nav = [{"ui": ui, "subtopic": s.get("subtopic", f"Subtopic {ui+1}")} for ui, s in enumerate(subs)]
            topics_nav.append({"ti": ti, "topic": top.get("topic", f"Topic {ti+1}"), "subtopics": subs_nav})

        if topics_nav:
            nav.append({"si": si, "subject": subj.get("subject", f"Subject {si+1}"), "topics": topics_nav})
            for t in topics_nav:
                for s in t["subtopics"]:
                    flat.append((si, t["ti"], s["ui"]))

    return nav, flat


# =========================================================
# Browser TTS (Web Speech API)
# =========================================================
def tts_component(text: str, voice_lang: str = "en-US", rate: float = 1.0, pitch: float = 1.0):
    """Text-to-speech in the user's browser (no API keys) using Web Speech API."""
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
      <button id="tts_play"
        style="padding:8px 12px; border-radius:10px; border:1px solid rgba(0,0,0,0.12); cursor:pointer;">
        ▶️ Play
      </button>
      <button id="tts_stop"
        style="padding:8px 12px; border-radius:10px; border:1px solid rgba(0,0,0,0.12); cursor:pointer;">
        ⏹ Stop
      </button>
      <span style="color:rgba(0,0,0,0.55); font-size: 0.9rem;">(Uses your browser’s TTS)</span>
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


# =========================================================
# Data load
# =========================================================
@st.cache_data(show_spinner=True)
def load_structure_from_url(url: str) -> List[Dict]:
    return parse_docx_to_structure(fetch_docx_bytes(url))


with st.sidebar:
    st.header("📄 Document")
    url = st.text_input("DOCX URL", value=DEFAULT_URL)
    st.caption("Uses **Subject / Topic / Subtopic** lines if present, otherwise Heading 1/2/3.")


try:
    subjects = load_structure_from_url(url)
except Exception as e:
    st.error(f"Could not load DOCX.\n\nError: {e}")
    st.stop()

nav, flat = build_navigation(subjects)
if not nav or not flat:
    st.warning(
        "No usable Subject/Topic/Subtopic sections found.\n\n"
        "Add lines like:\n"
        "- Subject: ...\n"
        "- Topic: ...\n"
        "- Subtopic: ...\n"
        "or use Heading 1/2/3 in the DOCX."
    )
    st.stop()


# =========================================================
# Session state + helpers
# =========================================================
if "flat_index" not in st.session_state:
    st.session_state.flat_index = 0
st.session_state.flat_index = max(0, min(st.session_state.flat_index, len(flat) - 1))

cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]


def jump_to(si: int, ti: int, ui: int):
    for idx, (sii, tii, uii) in enumerate(flat):
        if sii == si and tii == ti and uii == ui:
            st.session_state.flat_index = idx
            break
    st.rerun()


# =========================================================
# Sidebar navigation + TTS controls
# =========================================================
with st.sidebar:
    st.divider()
    st.header("🧭 Navigate")

    # 1) Subject dropdown
    subject_options = [x["subject"] for x in nav]
    cur_subject_nav_idx = next(i for i, x in enumerate(nav) if x["si"] == cur_si)
    selected_subject = st.selectbox("Subject", subject_options, index=cur_subject_nav_idx)

    subj_nav_idx = subject_options.index(selected_subject)
    subj_node = nav[subj_nav_idx]
    new_si = subj_node["si"]

    # 2) Topic dropdown (depends on Subject)
    topic_options = [t["topic"] for t in subj_node["topics"]]
    if new_si == cur_si:
        topic_default_idx = next((i for i, t in enumerate(subj_node["topics"]) if t["ti"] == cur_ti), 0)
    else:
        topic_default_idx = 0
    selected_topic = st.selectbox("Topic", topic_options, index=topic_default_idx)

    topic_nav_idx = topic_options.index(selected_topic)
    topic_node = subj_node["topics"][topic_nav_idx]
    new_ti = topic_node["ti"]

    # 3) Subtopic dropdown (depends on Topic)
    subtopic_options = [s["subtopic"] for s in topic_node["subtopics"]]
    if new_si == cur_si and new_ti == cur_ti:
        subtopic_default_idx = next((i for i, s in enumerate(topic_node["subtopics"]) if s["ui"] == cur_ui), 0)
    else:
        subtopic_default_idx = 0
    selected_subtopic = st.selectbox("Subtopic", subtopic_options, index=subtopic_default_idx)

    sub_nav_idx = subtopic_options.index(selected_subtopic)
    new_ui = topic_node["subtopics"][sub_nav_idx]["ui"]

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Go", use_container_width=True):
            jump_to(new_si, new_ti, new_ui)
    with cols[1]:
        if st.button("Reset", use_container_width=True):
            st.session_state.flat_index = 0
            st.rerun()

    st.divider()
    st.subheader("🔊 Text-to-Speech")
    voice_lang = st.selectbox("Voice language", ["en-US", "en-GB", "en", "es-ES", "fr-FR"], index=0)
    rate = st.slider("Rate", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)

    st.caption("Tip: Use **Next/Back** on the page for quick flow.")


# =========================================================
# Current content
# =========================================================
cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]
cur_subject = subjects[cur_si]["subject"]
cur_topic = subjects[cur_si]["topics"][cur_ti]["topic"]
cur_subtopic = subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui]["subtopic"]
cur_text = (subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui].get("full_text") or "").strip()

progress_num = st.session_state.flat_index + 1
progress_den = len(flat)
progress_pct = progress_num / progress_den if progress_den else 0.0


# =========================================================
# Main layout
# =========================================================
left, right = st.columns([2.4, 1.2], vertical_alignment="top")

with left:
    st.markdown(
        f"""
<div class="study-card">
  <div class="breadcrumb">Subject • <b>{cur_subject}</b> &nbsp;&nbsp;→&nbsp;&nbsp;
       Topic • <b>{cur_topic}</b> &nbsp;&nbsp;→&nbsp;&nbsp;
       Subtopic • <b>{cur_subtopic}</b>
  </div>
  <hr />
</div>
""",
        unsafe_allow_html=True,
    )

    if cur_text:
        # Keep content readable; Streamlit handles wrapping well.
        st.markdown(f"<div class='study-card'>{cur_text.replace('\\n', '<br><br>')}</div>", unsafe_allow_html=True)
    else:
        st.info("No paragraph text under this subtopic.")

    st.write("")  # breathing room

    nav_cols = st.columns([1, 1, 2])
    with nav_cols[0]:
        if st.button("⬅️ Back", disabled=(st.session_state.flat_index == 0), use_container_width=True):
            st.session_state.flat_index -= 1
            st.rerun()
    with nav_cols[1]:
        if st.button("Next ➡️", disabled=(st.session_state.flat_index == len(flat) - 1), use_container_width=True):
            st.session_state.flat_index += 1
            st.rerun()
    with nav_cols[2]:
        st.progress(progress_pct, text=f"Progress: {progress_num}/{progress_den}")

with right:
    st.subheader("🎧 Listen")
    st.markdown("<div class='small-muted'>Press play to read the current subtopic aloud.</div>", unsafe_allow_html=True)
    st.write("")

    if cur_text:
        tts_component(cur_text, voice_lang=voice_lang, rate=rate, pitch=pitch)
    else:
        st.caption("Nothing to read for this subtopic.")

    st.divider()
    st.subheader("📌 Quick Info")
    st.metric("Section", f"{progress_num} / {progress_den}")
    st.caption("Use the sidebar dropdowns to jump anywhere instantly.")
