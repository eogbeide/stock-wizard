# easymcat.py
# Streamlit DOCX Reader + Subject dropdown + Topic dropdown + Subtopic dropdown
# + Browser Text-to-Speech (actual installed voices dropdown) + Next/Back (sticky floating controls)
#
# Run:
#   streamlit run easymcat.py
#
# Default DOCX URL:
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Exam_Crackers.docx"

import io
import re
import uuid
from typing import Optional, List, Dict, Tuple

import requests
import streamlit as st
from docx import Document


# -----------------------------
# DOCX parsing helpers
# -----------------------------
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
        "subject": str, "topics": [
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
      - If content appears under a Topic before any Subtopic, we create an implicit Subtopic "Overview"
        so the 3rd dropdown always has something to show.
    """
    doc = Document(io.BytesIO(docx_bytes))

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

            if cur_topic is not None and cur_subtopic is None:
                ensure_subtopic("Overview")
            if cur_subtopic is not None:
                cur_subtopic["chunks"].append(raw)
            continue

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

        if cur_subtopic is None:
            ensure_subtopic("Overview")
        cur_subtopic["chunks"].append(raw)

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


# -----------------------------
# Browser TTS (Web Speech API)
# -----------------------------
def tts_component(
    text: str,
    preferred_lang: str = "en-GB",
    rate: float = 1.0,
    pitch: float = 0.8,
    prefer_deep_male_gb: bool = True,
):
    """Browser TTS with real installed voices dropdown + Play/Pause toggle + Stop."""
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    component_id = f"tts_{uuid.uuid4().hex}"

    html = f"""
    <div id="{component_id}" style="display:flex; flex-direction:column; gap:10px;">
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <button id="{component_id}_playpause" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
          ▶️ Play
        </button>
        <button id="{component_id}_stop" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
          ⏹ Stop
        </button>
        <span id="{component_id}_status" style="color:#666; font-size: 0.9rem;">Ready</span>
      </div>

      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <label style="color:#444; font-size:0.9rem;">Voice</label>
        <select id="{component_id}_voice" style="min-width: 260px; padding:8px 10px; border-radius:8px; border:1px solid #ddd;">
          <option value="">Loading voices…</option>
        </select>
        <button id="{component_id}_resetvoice" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
          Reset default
        </button>
        <span style="color:#888; font-size: 0.85rem;">(Installed voices vary by OS/browser)</span>
      </div>
    </div>

    <script>
      const ROOT_ID = "{component_id}";
      const TEXT = `{safe}`;

      const playPauseBtn = document.getElementById(ROOT_ID + "_playpause");
      const stopBtn = document.getElementById(ROOT_ID + "_stop");
      const voiceSelect = document.getElementById(ROOT_ID + "_voice");
      const resetVoiceBtn = document.getElementById(ROOT_ID + "_resetvoice");
      const statusEl = document.getElementById(ROOT_ID + "_status");

      const preferredLang = "{preferred_lang}";
      const preferredRate = {rate};
      const preferredPitch = {pitch};
      const preferDeepMaleGb = {str(prefer_deep_male_gb).lower()};

      const storageKey = "easymcat_tts_voice_uri";

      function setStatus(msg) {{
        statusEl.textContent = msg;
      }}

      function setPlayPauseLabel() {{
        if (!("speechSynthesis" in window)) {{
          playPauseBtn.textContent = "▶️ Play";
          return;
        }}
        const synth = window.speechSynthesis;
        if (synth.speaking) {{
          playPauseBtn.textContent = synth.paused ? "🔊 Resume" : "⏸ Pause";
        }} else {{
          playPauseBtn.textContent = "▶️ Play";
        }}
      }}

      function ensureSpeechSupport() {{
        if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {{
          alert("Your browser doesn't support speech synthesis.");
          return false;
        }}
        return true;
      }}

      function getVoicesAsync() {{
        return new Promise((resolve) => {{
          const synth = window.speechSynthesis;
          let voices = synth.getVoices();

          if (voices && voices.length) {{
            resolve(voices);
            return;
          }}

          const onVoicesChanged = () => {{
            voices = synth.getVoices();
            synth.removeEventListener("voiceschanged", onVoicesChanged);
            resolve(voices || []);
          }};

          synth.addEventListener("voiceschanged", onVoicesChanged);
          setTimeout(() => {{
            voices = synth.getVoices();
            synth.removeEventListener("voiceschanged", onVoicesChanged);
            resolve(voices || []);
          }}, 1200);
        }});
      }}

      function voiceLabel(v) {{
        const name = v.name || "Unnamed";
        const lang = v.lang || "";
        return lang ? `${{name}} (${{lang}})` : name;
      }}

      function pickDefaultVoice(voices) {{
        if (!voices || !voices.length) return null;

        const normLang = (s) => (s || "").toLowerCase();
        const byLangPrefix = (prefix) =>
          voices.filter(v => normLang(v.lang).startsWith(prefix.toLowerCase()));

        const gbVoices = byLangPrefix("en-gb");
        const enVoices = byLangPrefix("en");
        const preferredPrefix = preferredLang ? preferredLang.toLowerCase() : "";
        const preferredMatches = preferredPrefix ? byLangPrefix(preferredPrefix) : [];

        const maleNamePatterns = [
          /google.*uk.*english.*male/i,
          /microsoft.*(ryan|george|alfie).*online/i,
          /microsoft.*(ryan|george|alfie)/i,
          /daniel/i,
          /male/i
        ];

        if (preferDeepMaleGb) {{
          for (const re of maleNamePatterns) {{
            const found = gbVoices.find(v => re.test(v.name || ""));
            if (found) return found;
          }}
          if (gbVoices.length) return gbVoices[0];
        }}

        if (preferredMatches.length) return preferredMatches[0];
        if (gbVoices.length) return gbVoices[0];
        if (enVoices.length) return enVoices[0];
        return voices[0];
      }}

      function populateVoices(voices) {{
        voiceSelect.innerHTML = "";

        const savedUri = (() => {{
          try {{ return localStorage.getItem(storageKey); }} catch (e) {{ return null; }}
        }})();

        const options = voices.map((v) => ({{
          uri: v.voiceURI || "",
          label: voiceLabel(v),
          voice: v
        }}));

        options.sort((a, b) => a.label.localeCompare(b.label));

        for (const opt of options) {{
          const el = document.createElement("option");
          el.value = opt.uri;
          el.textContent = opt.label;
          voiceSelect.appendChild(el);
        }}

        const hasSaved = savedUri && options.some(o => o.uri === savedUri);
        if (hasSaved) {{
          voiceSelect.value = savedUri;
          setStatus("Voice loaded (saved selection).");
          return;
        }}

        const def = pickDefaultVoice(voices);
        if (def && def.voiceURI) {{
          voiceSelect.value = def.voiceURI;
          setStatus("Voice loaded (default selection).");
        }} else {{
          setStatus("Voice loaded.");
        }}
      }}

      async function initVoices() {{
        if (!ensureSpeechSupport()) {{
          setStatus("Speech not supported.");
          return;
        }}
        const voices = await getVoicesAsync();
        if (!voices || !voices.length) {{
          voiceSelect.innerHTML = '<option value="">No voices found</option>';
          setStatus("No voices found.");
          return;
        }}
        populateVoices(voices);
      }}

      function getSelectedVoice(voices) {{
        const uri = voiceSelect.value || "";
        if (!uri) return null;
        return voices.find(v => (v.voiceURI || "") === uri) || null;
      }}

      function stopAll() {{
        if (!ensureSpeechSupport()) return;
        window.speechSynthesis.cancel();
        setStatus("Stopped.");
        setPlayPauseLabel();
      }}

      async function speakNew() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;
        synth.cancel();

        const voices = await getVoicesAsync();
        const utter = new SpeechSynthesisUtterance(TEXT);

        utter.rate = preferredRate;
        utter.pitch = preferredPitch;
        utter.lang = preferredLang;

        const chosen = getSelectedVoice(voices) || pickDefaultVoice(voices);
        if (chosen) {{
          utter.voice = chosen;
          if (chosen.lang) utter.lang = chosen.lang;
        }}

        utter.onstart = () => {{
          setStatus("Speaking…");
          setPlayPauseLabel();
        }};
        utter.onend = () => {{
          setStatus("Done.");
          setPlayPauseLabel();
        }};
        utter.onerror = () => {{
          setStatus("TTS error.");
          setPlayPauseLabel();
        }};

        synth.speak(utter);
        setPlayPauseLabel();
      }}

      function togglePlayPause() {{
        if (!ensureSpeechSupport()) return;
        const synth = window.speechSynthesis;

        if (!synth.speaking) {{
          speakNew();
          return;
        }}

        if (synth.paused) {{
          synth.resume();
          setStatus("Speaking…");
        }} else {{
          synth.pause();
          setStatus("Paused.");
        }}
        setPlayPauseLabel();
      }}

      voiceSelect.addEventListener("change", () => {{
        try {{ localStorage.setItem(storageKey, voiceSelect.value || ""); }} catch (e) {{}}
        if (ensureSpeechSupport() && window.speechSynthesis.speaking) {{
          stopAll();
        }} else {{
          setStatus("Voice selected.");
          setPlayPauseLabel();
        }}
      }});

      resetVoiceBtn.addEventListener("click", async () => {{
        try {{ localStorage.removeItem(storageKey); }} catch (e) {{}}
        const voices = await getVoicesAsync();
        const def = pickDefaultVoice(voices);
        if (def && def.voiceURI) {{
          voiceSelect.value = def.voiceURI;
          setStatus("Reset to default voice.");
        }} else {{
          setStatus("Reset (no default found).");
        }}
        if (ensureSpeechSupport() && window.speechSynthesis.speaking) stopAll();
        setPlayPauseLabel();
      }});

      playPauseBtn.addEventListener("click", togglePlayPause);
      stopBtn.addEventListener("click", stopAll);

      setInterval(() => {{
        if (!ensureSpeechSupport()) return;
        setPlayPauseLabel();
      }}, 400);

      initVoices();
      setPlayPauseLabel();
    </script>
    """
    st.components.v1.html(html, height=130)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="DOCX Study Reader", layout="wide")
st.title("DOCX Study Reader")
st.caption("Sidebar: Subject (1) → Topic (2) → Subtopic (3) • Page: content + floating Controls (Listen + Next/Back)")

st.markdown(
    """
<style>
/* Make the SECOND column (right rail) sticky for the primary content row. */
div[data-testid="stHorizontalBlock"] {
  align-items: flex-start;
}

/* Keep the right rail floating while scrolling (works well when there's a single main columns row). */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
  position: sticky;
  top: 4.5rem; /* below Streamlit header */
  align-self: flex-start;
  z-index: 5;
}

/* Slight spacing so sticky rail doesn't visually jam into page edges */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
  padding-top: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=True)
def load_structure_from_url(url: str) -> List[Dict]:
    return parse_docx_to_structure(fetch_docx_bytes(url))


with st.sidebar:
    st.header("Document Source")
    url = st.text_input("DOCX URL", value=DEFAULT_URL)
    st.write("Navigation comes from **Subject:** / **Topic:** / **Subtopic:** lines (if present), otherwise Heading 1/2/3.")

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

if "flat_index" not in st.session_state:
    st.session_state.flat_index = 0
st.session_state.flat_index = max(0, min(st.session_state.flat_index, len(flat) - 1))

cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]

with st.sidebar:
    st.header("Navigate")

    subject_options = [x["subject"] for x in nav]
    cur_subject_nav_idx = next(i for i, x in enumerate(nav) if x["si"] == cur_si)
    selected_subject = st.selectbox("Subject", subject_options, index=cur_subject_nav_idx)

    subj_nav_idx = subject_options.index(selected_subject)
    subj_node = nav[subj_nav_idx]
    new_si = subj_node["si"]

    topic_options = [t["topic"] for t in subj_node["topics"]]
    if new_si == cur_si:
        topic_default_idx = next((i for i, t in enumerate(subj_node["topics"]) if t["ti"] == cur_ti), 0)
    else:
        topic_default_idx = 0
    selected_topic = st.selectbox("Topic", topic_options, index=topic_default_idx)

    topic_nav_idx = topic_options.index(selected_topic)
    topic_node = subj_node["topics"][topic_nav_idx]
    new_ti = topic_node["ti"]

    subtopic_options = [s["subtopic"] for s in topic_node["subtopics"]]
    if new_si == cur_si and new_ti == cur_ti:
        subtopic_default_idx = next((i for i, s in enumerate(topic_node["subtopics"]) if s["ui"] == cur_ui), 0)
    else:
        subtopic_default_idx = 0
    selected_subtopic = st.selectbox("Subtopic", subtopic_options, index=subtopic_default_idx)

    sub_nav_idx = subtopic_options.index(selected_subtopic)
    new_ui = topic_node["subtopics"][sub_nav_idx]["ui"]

    if st.button("Go", use_container_width=True):
        for idx, (si, ti, ui) in enumerate(flat):
            if si == new_si and ti == new_ti and ui == new_ui:
                st.session_state.flat_index = idx
                break
        st.rerun()

    st.divider()
    st.subheader("Text-to-Speech")

    preferred_lang = st.selectbox(
        "Preferred language",
        ["en-GB", "en-US", "en", "es-ES", "fr-FR"],
        index=0,
        help="Voice dropdown uses your installed voices. This sets the default preference/fallback.",
    )
    rate = st.slider("Rate", 0.5, 2.0, 1.0, 0.1)
    pitch = st.slider("Pitch", 0.5, 2.0, 0.8, 0.1, help="Lower pitch generally sounds deeper.")
    prefer_deep_male_gb = st.toggle(
        "Prefer deep male UK voice (default)",
        value=True,
        help="Heuristic selection: prefers en-GB male voices when present.",
    )

cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]
cur_subject = subjects[cur_si]["subject"]
cur_topic = subjects[cur_si]["topics"][cur_ti]["topic"]
cur_subtopic = subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui]["subtopic"]
cur_text = (subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui].get("full_text") or "").strip()

col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    st.subheader(f"{cur_subject}  →  {cur_topic}  →  {cur_subtopic}")

    if cur_text:
        st.write(cur_text)
    else:
        st.info("No paragraph text under this subtopic.")

with col_right:
    st.subheader("Controls")

    st.markdown("**Listen**")
    if cur_text:
        tts_component(
            cur_text,
            preferred_lang=preferred_lang,
            rate=rate,
            pitch=pitch,
            prefer_deep_male_gb=prefer_deep_male_gb,
        )
    else:
        st.caption("Nothing to read for this subtopic.")

    st.divider()

    st.markdown("**Navigate**")
    if st.button("⬅️ Back", disabled=(st.session_state.flat_index == 0), use_container_width=True):
        st.session_state.flat_index -= 1
        st.rerun()

    if st.button("Next ➡️", disabled=(st.session_state.flat_index == len(flat) - 1), use_container_width=True):
        st.session_state.flat_index += 1
        st.rerun()

    st.divider()
    st.caption("Progress")
    st.write(f"Section {st.session_state.flat_index + 1} of {len(flat)}")
