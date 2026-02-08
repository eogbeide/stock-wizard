# easymcat.py
# Streamlit DOCX Study Reader
# - Subject/Topic/Subtopic navigation (markers or Heading 1/2/3)
# - Browser Text-to-Speech (installed voices dropdown)
# - Flow reading: sentence/paragraph pauses (optionally clause pauses)
# - Sticky floating Controls (Listen + Next/Back)
#
# Run:
#   streamlit run easymcat.py
#
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
      - If content appears under a Topic before any Subtopic, we create an implicit Subtopic "Overview".
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
            subj["topics"] = [
                {"topic": "Overview", "subtopics": [{"subtopic": "Overview", "chunks": [], "full_text": ""}]}
            ]
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
    rate: float = 0.95,
    pitch: float = 0.78,
    prefer_deep_male_gb: bool = True,
    flow_mode: bool = True,
    sentence_pause_ms: int = 320,
    paragraph_pause_ms: int = 650,
    clause_pause_ms: int = 0,
):
    """
    Browser TTS:
    - Real installed voices dropdown
    - Play/Pause toggle + Stop
    - Flow mode: speaks sentence-by-sentence with pauses (optional clause pauses)
    """
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

      const flowMode = {str(flow_mode).lower()};
      const sentencePauseMs = Math.max(0, {sentence_pause_ms});
      const paragraphPauseMs = Math.max(0, {paragraph_pause_ms});
      const clausePauseMs = Math.max(0, {clause_pause_ms});

      const storageKey = "easymcat_tts_voice_uri";

      let queue = [];
      let idx = 0;

      let betweenTimer = null;
      let betweenDueAt = 0;
      let betweenRemaining = 0;
      let betweenPaused = false;

      function setStatus(msg) {{
        statusEl.textContent = msg;
      }}

      function ensureSpeechSupport() {{
        if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {{
          alert("Your browser doesn't support speech synthesis.");
          return false;
        }}
        return true;
      }}

      function setPlayPauseLabel() {{
        if (!("speechSynthesis" in window)) {{
          playPauseBtn.textContent = "▶️ Play";
          return;
        }}
        const synth = window.speechSynthesis;

        if (betweenPaused) {{
          playPauseBtn.textContent = "🔊 Resume";
          return;
        }}

        if (synth.speaking) {{
          playPauseBtn.textContent = synth.paused ? "🔊 Resume" : "⏸ Pause";
        }} else {{
          playPauseBtn.textContent = "▶️ Play";
        }}
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
          setStatus("Voice loaded (saved).");
          return;
        }}

        const def = pickDefaultVoice(voices);
        if (def && def.voiceURI) {{
          voiceSelect.value = def.voiceURI;
          setStatus("Voice loaded (default).");
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

      function normalizeText(t) {{
        return (t || "")
          .replace(/\\r\\n/g, "\\n")
          .replace(/\\r/g, "\\n")
          .replace(/[ \\t]+/g, " ")
          .replace(/\\n[ \\t]+/g, "\\n")
          .trim();
      }}

      function splitParagraphs(t) {{
        const cleaned = normalizeText(t);
        if (!cleaned) return [];
        return cleaned
          .split(/\\n\\s*\\n+/g)
          .map(p => p.trim())
          .filter(Boolean);
      }}

      function splitSentences(paragraph) {{
        const p = (paragraph || "").trim();
        if (!p) return [];

        // Prefer Intl.Segmenter when available (best sentence boundaries)
        if (typeof Intl !== "undefined" && Intl.Segmenter) {{
          try {{
            const seg = new Intl.Segmenter("en", {{ granularity: "sentence" }});
            const parts = [];
            for (const s of seg.segment(p)) {{
              const chunk = (s.segment || "").trim();
              if (chunk) parts.push(chunk);
            }}
            if (parts.length) return parts;
          }} catch (e) {{}}
        }}

        // Fallback: split after . ! ? followed by whitespace
        return p
          .split(/(?<=[.!?])\\s+/g)
          .map(s => s.trim())
          .filter(Boolean);
      }}

      function splitClauses(sentence) {{
        const s = (sentence || "").trim();
        if (!s) return [];

        // Keep punctuation with the clause.
        const tokens = s.split(/(,|;|:)/g);
        const out = [];
        let buf = "";

        for (let i = 0; i < tokens.length; i++) {{
          const tok = tokens[i];
          if (tok === "," || tok === ";" || tok === ":") {{
            buf = (buf + tok).trim();
            if (buf) out.push(buf);
            buf = "";
          }} else {{
            buf = (buf + " " + tok).trim();
          }}
        }}
        if (buf.trim()) out.push(buf.trim());
        return out.filter(Boolean);
      }}

      function buildQueueFromText(t) {{
        const paragraphs = splitParagraphs(t);
        const items = [];

        for (let pi = 0; pi < paragraphs.length; pi++) {{
          const sentences = splitSentences(paragraphs[pi]);

          for (let si = 0; si < sentences.length; si++) {{
            const sent = sentences[si];

            if (flowMode && clausePauseMs > 0) {{
              const clauses = splitClauses(sent);
              for (let ci = 0; ci < clauses.length; ci++) {{
                const isLastClause = ci === clauses.length - 1;
                const isLastSentence = si === sentences.length - 1;
                const isLastParagraph = pi === paragraphs.length - 1;

                let pauseAfter = 0;
                if (!isLastClause) pauseAfter = clausePauseMs;
                else if (!isLastSentence) pauseAfter = sentencePauseMs;
                else if (!isLastParagraph) pauseAfter = paragraphPauseMs;

                items.push({{ text: clauses[ci], pauseAfter }});
              }}
            }} else if (flowMode) {{
              const isLastSentence = si === sentences.length - 1;
              const isLastParagraph = pi === paragraphs.length - 1;

              let pauseAfter = 0;
              if (!isLastSentence) pauseAfter = sentencePauseMs;
              else if (!isLastParagraph) pauseAfter = paragraphPauseMs;

              items.push({{ text: sent, pauseAfter }});
            }} else {{
              // No flow mode: single utterance (no queue splitting)
              return [{{ text: normalizeText(t), pauseAfter: 0 }}];
            }}
          }}
        }}

        // If nothing parsed, fallback to whole text
        return items.length ? items : [{{ text: normalizeText(t), pauseAfter: 0 }}];
      }}

      function clearBetweenTimer() {{
        if (betweenTimer) {{
          clearTimeout(betweenTimer);
          betweenTimer = null;
        }}
      }}

      function stopAll() {{
        if (!ensureSpeechSupport()) return;

        clearBetweenTimer();
        betweenPaused = false;
        betweenRemaining = 0;

        window.speechSynthesis.cancel();
        queue = [];
        idx = 0;

        setStatus("Stopped.");
        setPlayPauseLabel();
      }}

      function pauseAll() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        // Pausing while in-between utterances (timer)
        if (betweenTimer) {{
          const now = Date.now();
          betweenRemaining = Math.max(0, betweenDueAt - now);
          clearBetweenTimer();
          betweenPaused = true;
          setStatus("Paused.");
          setPlayPauseLabel();
          return;
        }}

        // Pausing while speaking
        if (synth.speaking && !synth.paused) {{
          synth.pause();
          setStatus("Paused.");
          setPlayPauseLabel();
        }}
      }}

      function resumeAll() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        // Resuming while in-between utterances
        if (betweenPaused) {{
          betweenPaused = false;
          const delay = Math.max(0, betweenRemaining);
          betweenRemaining = 0;
          scheduleNext(delay);
          setStatus("Speaking…");
          setPlayPauseLabel();
          return;
        }}

        // Resuming while speaking
        if (synth.paused) {{
          synth.resume();
          setStatus("Speaking…");
          setPlayPauseLabel();
        }}
      }}

      function scheduleNext(delayMs) {{
        clearBetweenTimer();

        if (delayMs <= 0) {{
          speakNext();
          return;
        }}

        betweenDueAt = Date.now() + delayMs;
        betweenTimer = setTimeout(() => {{
          betweenTimer = null;
          speakNext();
        }}, delayMs);
      }}

      async function speakItem(itemText) {{
        const voices = await getVoicesAsync();

        const utter = new SpeechSynthesisUtterance(itemText);
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
          setPlayPauseLabel();
        }};
        utter.onerror = () => {{
          setStatus("TTS error.");
          setPlayPauseLabel();
        }};

        window.speechSynthesis.speak(utter);
      }}

      function speakNext() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        if (idx >= queue.length) {{
          setStatus("Done.");
          setPlayPauseLabel();
          return;
        }}

        const item = queue[idx];
        idx += 1;

        synth.cancel(); // avoid overlap edge-cases
        speakItem(item.text);

        // Schedule next chunk after the current utterance ends + pauseAfter.
        // We can't reliably chain with utter.onend for every browser timing case when cancel() happens,
        // so we poll speaking state and then schedule the next.
        const pauseAfter = Math.max(0, item.pauseAfter || 0);

        const watcher = setInterval(() => {{
          if (!ensureSpeechSupport()) {{
            clearInterval(watcher);
            return;
          }}
          const s = window.speechSynthesis;

          if (!s.speaking && !s.paused) {{
            clearInterval(watcher);
            scheduleNext(pauseAfter);
          }}
        }}, 120);
      }}

      function startFresh() {{
        if (!ensureSpeechSupport()) return;

        clearBetweenTimer();
        betweenPaused = false;
        betweenRemaining = 0;

        window.speechSynthesis.cancel();

        queue = buildQueueFromText(TEXT);
        idx = 0;

        speakNext();
        setPlayPauseLabel();
      }}

      function togglePlayPause() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        if (betweenPaused || synth.paused) {{
          resumeAll();
          return;
        }}

        if (betweenTimer || (synth.speaking && !synth.paused)) {{
          pauseAll();
          return;
        }}

        // Not speaking: resume queued progress if any, otherwise start fresh
        if (queue.length && idx < queue.length) {{
          setStatus("Speaking…");
          speakNext();
          setPlayPauseLabel();
          return;
        }}

        startFresh();
      }}

      voiceSelect.addEventListener("change", () => {{
        try {{ localStorage.setItem(storageKey, voiceSelect.value || ""); }} catch (e) {{}}
        if (ensureSpeechSupport() && (window.speechSynthesis.speaking || window.speechSynthesis.paused)) {{
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
        if (ensureSpeechSupport() && (window.speechSynthesis.speaking || window.speechSynthesis.paused)) stopAll();
        setPlayPauseLabel();
      }});

      playPauseBtn.addEventListener("click", togglePlayPause);
      stopBtn.addEventListener("click", stopAll);

      setInterval(() => {{
        if (!ensureSpeechSupport()) return;
        setPlayPauseLabel();
      }}, 350);

      initVoices();
      setPlayPauseLabel();
    </script>
    """
    st.components.v1.html(html, height=140)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="DOCX Study Reader", layout="wide")
st.title("DOCX Study Reader")
st.caption("Sidebar: Subject (1) → Topic (2) → Subtopic (3) • Page: content + floating Controls (Listen + Next/Back)")

st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"] { align-items: flex-start; }

/* Sticky right rail (2nd column) */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
  position: sticky;
  top: 4.5rem;
  align-self: flex-start;
  z-index: 5;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div { padding-top: 0.25rem; }
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

    rate = st.slider("Rate", 0.5, 2.0, 0.95, 0.05)
    pitch = st.slider("Pitch", 0.5, 2.0, 0.78, 0.05, help="Lower pitch generally sounds deeper.")
    prefer_deep_male_gb = st.toggle(
        "Prefer deep male UK voice (default)",
        value=True,
        help="Heuristic selection: prefers en-GB male voices when present.",
    )

    st.divider()
    st.subheader("Flow (pauses)")

    flow_mode = st.toggle(
        "Flow mode (sentence pacing)",
        value=True,
        help="Speaks sentence-by-sentence with short pauses, for a more 'coach-like' delivery.",
    )
    sentence_pause_ms = st.slider("Pause after sentence (ms)", 0, 1500, 320, 20)
    paragraph_pause_ms = st.slider("Pause after paragraph (ms)", 0, 4000, 650, 50)
    clause_pause_ms = st.slider(
        "Optional pause after clause (ms)",
        0,
        800,
        0,
        10,
        help="If > 0, splits sentences by commas/;/: and adds a light pause for clearer rhythm.",
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
            flow_mode=flow_mode,
            sentence_pause_ms=sentence_pause_ms,
            paragraph_pause_ms=paragraph_pause_ms,
            clause_pause_ms=clause_pause_ms,
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
