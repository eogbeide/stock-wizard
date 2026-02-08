# easymcat.py
# Streamlit DOCX Study Reader
# - Subject/Topic/Subtopic navigation (markers or Heading 1/2/3)
# - Nigerian Pidgin translation (DEFAULT) + Original view
# - Browser Text-to-Speech (installed voices dropdown; Voice A/B for chatty/debate)
# - Conversation narration (Standard/Chatty/Debate) + Flow reading pauses
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
# Nigerian Pidgin (rule-based) translation
# -----------------------------
_PIDGIN_LEVEL_1: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\blet\s+us\b", re.IGNORECASE), "make we"),
    (re.compile(r"\blet's\b", re.IGNORECASE), "make we"),
    (re.compile(r"\bdo\s+not\b", re.IGNORECASE), "no"),
    (re.compile(r"\bdon't\b", re.IGNORECASE), "no"),
    (re.compile(r"\bdoes\s+not\b", re.IGNORECASE), "no"),
    (re.compile(r"\bdid\s+not\b", re.IGNORECASE), "no"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "no fit"),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "no fit"),
    (re.compile(r"\bwon't\b", re.IGNORECASE), "no go"),
    (re.compile(r"\bshould\s+not\b", re.IGNORECASE), "no suppose"),
    (re.compile(r"\bwhat\s+is\b", re.IGNORECASE), "wetin be"),
    (re.compile(r"\bwhat's\b", re.IGNORECASE), "wetin be"),
    (re.compile(r"\bwhat\s+are\b", re.IGNORECASE), "wetin be"),
    (re.compile(r"\btherefore\b", re.IGNORECASE), "so"),
    (re.compile(r"\bhowever\b", re.IGNORECASE), "but"),
    (re.compile(r"\bremember\b", re.IGNORECASE), "no forget"),
    (re.compile(r"\bthis\s+means\b", re.IGNORECASE), "e mean say"),
]

_PIDGIN_LEVEL_2: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bvery\b", re.IGNORECASE), "well well"),
    (re.compile(r"\bbecause\b", re.IGNORECASE), "because say"),
    (re.compile(r"\breally\b", re.IGNORECASE), "for real"),
    (re.compile(r"\btruly\b", re.IGNORECASE), "for real"),
    (re.compile(r"\bin\s+other\s+words\b", re.IGNORECASE), "as e be"),
    (re.compile(r"\bfor\s+that\s+reason\b", re.IGNORECASE), "na why"),
]


def pidgin_translate_text(text: str, intensity: int = 1) -> str:
    """
    Conservative, local, rule-based “pidgin-style” translation.
    - intensity 0: off
    - intensity 1: light (default)
    - intensity 2: stronger
    """
    if not text:
        return ""
    intensity = max(0, min(2, int(intensity)))

    if intensity == 0:
        return text.strip()

    # Preserve paragraph breaks.
    parts = re.split(r"(\n\s*\n+)", text.strip())
    out: List[str] = []
    for part in parts:
        if re.match(r"^\n\s*\n+$", part or ""):
            out.append(part)
            continue

        s = part

        for pat, repl in _PIDGIN_LEVEL_1:
            s = pat.sub(repl, s)

        if intensity >= 2:
            for pat, repl in _PIDGIN_LEVEL_2:
                s = pat.sub(repl, s)

        # Light fillers: keep it subtle.
        if intensity >= 1:
            s = re.sub(r"(?m)^\s*Note:\s*", "Abeg note say: ", s)

        out.append(s)

    return "".join(out).strip()


# -----------------------------
# Browser TTS (Web Speech API)
# -----------------------------
def tts_component(
    text: str,
    preferred_lang: str = "en-NG",
    rate: float = 0.95,
    pitch: float = 0.78,
    prefer_deep_male_gb: bool = False,
    narration_style: str = "Chatty",  # Standard | Chatty | Debate
    flow_mode: bool = True,
    sentence_pause_ms: int = 320,
    paragraph_pause_ms: int = 650,
    clause_pause_ms: int = 0,
    turn_pause_ms: int = 260,
    pidgin_mode: bool = True,
    pidgin_transform_enabled: bool = False,  # we already translate in Python; keep JS transform off
):
    """
    Browser TTS:
    - Installed voices dropdown (Voice A/B)
    - Play/Pause toggle + Stop
    - Flow mode: speaks sentence-by-sentence with pauses (optional clause pauses)
    - Narration style: Standard/Chatty/Debate (A/B alternation)
    - Pidgin mode: prefers en-NG voices when present (no extra JS translation by default)
    """
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    component_id = f"tts_{uuid.uuid4().hex}"

    style_js = narration_style.replace("\\", "").replace("`", "").replace("${", "")

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

      <div style="display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <label style="color:#444; font-size:0.9rem; min-width:72px;">Voice A</label>
          <select id="{component_id}_voice_a" style="min-width: 320px; padding:8px 10px; border-radius:8px; border:1px solid #ddd;">
            <option value="">Loading voices…</option>
          </select>
        </div>

        <div id="{component_id}_voiceb_row" style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <label style="color:#444; font-size:0.9rem; min-width:72px;">Voice B</label>
          <select id="{component_id}_voice_b" style="min-width: 320px; padding:8px 10px; border-radius:8px; border:1px solid #ddd;">
            <option value="">Loading voices…</option>
          </select>
        </div>

        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
          <button id="{component_id}_resetvoice" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
            Reset defaults
          </button>
          <span style="color:#888; font-size: 0.85rem;">
            (A/B alternate in Chatty/Debate. If B not available, it reuses A with a different tone.)
          </span>
        </div>
      </div>
    </div>

    <script>
      const ROOT_ID = "{component_id}";
      const TEXT = `{safe}`;

      const playPauseBtn = document.getElementById(ROOT_ID + "_playpause");
      const stopBtn = document.getElementById(ROOT_ID + "_stop");
      const statusEl = document.getElementById(ROOT_ID + "_status");

      const voiceSelectA = document.getElementById(ROOT_ID + "_voice_a");
      const voiceSelectB = document.getElementById(ROOT_ID + "_voice_b");
      const voiceBRow = document.getElementById(ROOT_ID + "_voiceb_row");
      const resetVoiceBtn = document.getElementById(ROOT_ID + "_resetvoice");

      const preferredLang = "{preferred_lang}";
      const preferredRate = {rate};
      const preferredPitch = {pitch};
      const preferDeepMaleGb = {str(prefer_deep_male_gb).lower()};

      const narrationStyle = "{style_js}";
      const conversationMode = (narrationStyle === "Chatty" || narrationStyle === "Debate");

      const flowMode = {str(flow_mode).lower()};
      const sentencePauseMs = Math.max(0, {sentence_pause_ms});
      const paragraphPauseMs = Math.max(0, {paragraph_pause_ms});
      const clausePauseMs = Math.max(0, {clause_pause_ms});
      const turnPauseMs = Math.max(0, {turn_pause_ms});

      const pidginMode = {str(pidgin_mode).lower()};
      const pidginTransformEnabled = {str(pidgin_transform_enabled).lower()};

      const storageKeyA = "easymcat_tts_voice_uri_a";
      const storageKeyB = "easymcat_tts_voice_uri_b";

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

      function showHideVoiceB() {{
        voiceBRow.style.display = conversationMode ? "flex" : "none";
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

      function normalizeLang(s) {{
        return (s || "").toLowerCase().trim();
      }}

      function byLangPrefix(voices, prefix) {{
        const p = (prefix || "").toLowerCase();
        return (voices || []).filter(v => normalizeLang(v.lang).startsWith(p));
      }}

      function pickDefaultVoice(voices) {{
        if (!voices || !voices.length) return null;

        // Pidgin: prefer en-NG voices first when available
        const ngVoices = byLangPrefix(voices, "en-ng");
        if (pidginMode && ngVoices.length) return ngVoices[0];

        const preferredPrefix = (preferredLang || "").toLowerCase();
        const preferredMatches = preferredPrefix ? byLangPrefix(voices, preferredPrefix) : [];
        if (preferredMatches.length) return preferredMatches[0];

        const gbVoices = byLangPrefix(voices, "en-gb");
        const usVoices = byLangPrefix(voices, "en-us");
        const enVoices = byLangPrefix(voices, "en");

        const maleNamePatterns = [
          /google.*uk.*english.*male/i,
          /microsoft.*(ryan|george|alfie).*online/i,
          /microsoft.*(ryan|george|alfie)/i,
          /daniel/i,
          /male/i
        ];

        if (preferDeepMaleGb && gbVoices.length) {{
          for (const re of maleNamePatterns) {{
            const found = gbVoices.find(v => re.test(v.name || ""));
            if (found) return found;
          }}
          return gbVoices[0];
        }}

        if (pidginMode && enVoices.length) return enVoices[0];
        if (gbVoices.length) return gbVoices[0];
        if (usVoices.length) return usVoices[0];
        if (enVoices.length) return enVoices[0];
        return voices[0];
      }}

      function pickSecondaryVoice(voices, primary) {{
        if (!voices || !voices.length) return null;
        if (!primary) return pickDefaultVoice(voices);

        const primaryUri = primary.voiceURI || "";
        const primaryLang = normalizeLang(primary.lang);

        const sameLang = voices.filter(v => normalizeLang(v.lang) === primaryLang && (v.voiceURI || "") !== primaryUri);
        if (sameLang.length) return sameLang[0];

        if (pidginMode) {{
          const ng = voices.filter(v => normalizeLang(v.lang).startsWith("en-ng") && (v.voiceURI || "") !== primaryUri);
          if (ng.length) return ng[0];
        }}

        const english = voices.filter(v => normalizeLang(v.lang).startsWith("en") && (v.voiceURI || "") !== primaryUri);
        if (english.length) return english[0];

        const anyOther = voices.find(v => (v.voiceURI || "") !== primaryUri);
        return anyOther || primary;
      }}

      function populateSelect(selectEl, voices) {{
        selectEl.innerHTML = "";
        for (const v of voices) {{
          const opt = document.createElement("option");
          opt.value = v.voiceURI || "";
          opt.textContent = voiceLabel(v);
          selectEl.appendChild(opt);
        }}
      }}

      function getSaved(key) {{
        try {{ return localStorage.getItem(key); }} catch (e) {{ return null; }}
      }}

      function setSaved(key, value) {{
        try {{ localStorage.setItem(key, value || ""); }} catch (e) {{}}
      }}

      function removeSaved(key) {{
        try {{ localStorage.removeItem(key); }} catch (e) {{}}
      }}

      function setDefaultsForAandB(voices) {{
        const defA = pickDefaultVoice(voices);
        const defB = pickSecondaryVoice(voices, defA);

        if (defA && defA.voiceURI) voiceSelectA.value = defA.voiceURI;
        if (defB && defB.voiceURI) voiceSelectB.value = defB.voiceURI;

        setSaved(storageKeyA, voiceSelectA.value || "");
        setSaved(storageKeyB, voiceSelectB.value || "");
      }}

      async function initVoices() {{
        showHideVoiceB();

        if (!ensureSpeechSupport()) {{
          setStatus("Speech not supported.");
          return;
        }}

        const voices = await getVoicesAsync();
        if (!voices || !voices.length) {{
          voiceSelectA.innerHTML = '<option value="">No voices found</option>';
          voiceSelectB.innerHTML = '<option value="">No voices found</option>';
          setStatus("No voices found.");
          return;
        }}

        populateSelect(voiceSelectA, voices);
        populateSelect(voiceSelectB, voices);

        const savedA = getSaved(storageKeyA);
        const savedB = getSaved(storageKeyB);

        const hasA = savedA && voices.some(v => (v.voiceURI || "") === savedA);
        const hasB = savedB && voices.some(v => (v.voiceURI || "") === savedB);

        if (hasA) voiceSelectA.value = savedA;
        if (hasB) voiceSelectB.value = savedB;

        if (!hasA || !hasB) {{
          setDefaultsForAandB(voices);
          setStatus("Voices loaded (defaults).");
        }} else {{
          setStatus("Voices loaded (saved).");
        }}
      }}

      function getVoiceByUri(voices, uri) {{
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

        return p
          .split(/(?<=[.!?])\\s+/g)
          .map(s => s.trim())
          .filter(Boolean);
      }}

      function splitClauses(sentence) {{
        const s = (sentence || "").trim();
        if (!s) return [];

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

      function safePidginize(s) {{
        // Off by default: we already translate in Python.
        if (!pidginMode || !pidginTransformEnabled) return s;
        return s;
      }}

      function pickPrefix(speaker, i) {{
        const chatA = pidginMode
          ? ["Oya,", "Make we yarn,", "See eh,", "As e be,", "Now,", "No be lie—"]
          : ["Alright,", "Okay,", "So,", "Now,", "Here’s the thing—", "Let’s be real—"];

        const chatB = pidginMode
          ? ["Ehen,", "True,", "Na so,", "But wait o—", "And see,", "Honestly—"]
          : ["Yeah,", "Right,", "Sure,", "Totally,", "And look—", "Honestly—"];

        const debateA = pidginMode
          ? ["Oya,", "My own take be say—", "Make we reason am—", "As I see am—", "Now—", "Listen—"]
          : ["Alright,", "So,", "Let’s say this plainly—", "Here’s my take—", "Now,", "Okay—"];

        const debateB = pidginMode
          ? ["But hold on o—", "Wait small—", "I no too agree—", "Fair, but—", "Counterpoint—", "True, but—"]
          : ["But hold on—", "Wait a second—", "I’m not sure about that—", "Fair, but—", "Counterpoint—", "Right, but—"];

        const listA = (narrationStyle === "Debate") ? debateA : chatA;
        const listB = (narrationStyle === "Debate") ? debateB : chatB;

        const arr = (speaker === "A") ? listA : listB;
        return arr[i % arr.length];
      }}

      function decorateForConversation(sentence, speaker, globalIndex) {{
        const s0 = (sentence || "").trim();
        if (!s0) return s0;

        const s = safePidginize(s0);
        const prefix = pickPrefix(speaker, globalIndex);

        if (/^(okay|alright|so|now|yeah|right|but|wait|honestly|sure|oya|ehen|na\\s+so|but\\s+wait)\\b/i.test(s)) return s;
        return `${{prefix}} ${{s}}`;
      }}

      function buildQueueFromText(t) {{
        const paragraphs = splitParagraphs(t);
        const items = [];

        if (!paragraphs.length) {{
          const one = normalizeText(t);
          return one ? [{{ text: safePidginize(one), pauseAfter: 0, speaker: "A" }}] : [];
        }}

        let globalSentenceIndex = 0;

        if (conversationMode) {{
          const opener = pidginMode
            ? (narrationStyle === "Debate" ? "Oya—make we test this matter together." : "Oya—make we yarn am small.")
            : (narrationStyle === "Debate" ? "Alright—let’s test this idea together." : "Alright—let’s talk this through.");
          items.push({{
            text: opener,
            pauseAfter: Math.max(120, turnPauseMs),
            speaker: "A"
          }});
        }}

        for (let pi = 0; pi < paragraphs.length; pi++) {{
          const sentences = splitSentences(paragraphs[pi]);

          for (let si = 0; si < sentences.length; si++) {{
            const sentRaw = sentences[si];

            const speaker = conversationMode ? ((globalSentenceIndex % 2 === 0) ? "A" : "B") : "A";
            const isLastSentence = si === sentences.length - 1;
            const isLastParagraph = pi === paragraphs.length - 1;

            let pauseAfter = 0;
            if (!isLastSentence) pauseAfter = sentencePauseMs;
            else if (!isLastParagraph) pauseAfter = paragraphPauseMs;

            if (conversationMode) pauseAfter += turnPauseMs;

            if (flowMode && clausePauseMs > 0) {{
              const clauses = splitClauses(sentRaw);
              if (clauses.length >= 2) {{
                for (let ci = 0; ci < clauses.length; ci++) {{
                  const isLastClause = ci === clauses.length - 1;
                  const firstClause = ci === 0;

                  let clauseText = clauses[ci];
                  if (conversationMode && firstClause) {{
                    clauseText = decorateForConversation(clauses[ci], speaker, globalSentenceIndex);
                  }} else {{
                    clauseText = safePidginize(clauseText);
                  }}

                  let clausePause = 0;
                  if (!isLastClause) clausePause = clausePauseMs;
                  else clausePause = pauseAfter;

                  items.push({{
                    text: clauseText,
                    pauseAfter: clausePause,
                    speaker
                  }});
                }}

                globalSentenceIndex += 1;
                continue;
              }}
            }}

            const finalText = conversationMode
              ? decorateForConversation(sentRaw, speaker, globalSentenceIndex)
              : safePidginize(sentRaw);

            items.push({{
              text: finalText,
              pauseAfter: flowMode ? pauseAfter : 0,
              speaker
            }});

            globalSentenceIndex += 1;
          }}
        }}

        if (!flowMode) {{
          const whole = normalizeText(t);
          if (!whole) return [];
          const spoken = conversationMode ? ("Alright—here it is. " + whole) : whole;
          return [{{ text: safePidginize(spoken), pauseAfter: 0, speaker: "A" }}];
        }}

        return items;
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

        if (betweenTimer) {{
          const now = Date.now();
          betweenRemaining = Math.max(0, betweenDueAt - now);
          clearBetweenTimer();
          betweenPaused = true;
          setStatus("Paused.");
          setPlayPauseLabel();
          return;
        }}

        if (synth.speaking && !synth.paused) {{
          synth.pause();
          setStatus("Paused.");
          setPlayPauseLabel();
        }}
      }}

      function resumeAll() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        if (betweenPaused) {{
          betweenPaused = false;
          const delay = Math.max(0, betweenRemaining);
          betweenRemaining = 0;
          scheduleNext(delay);
          setStatus("Speaking…");
          setPlayPauseLabel();
          return;
        }}

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

      function speakerTone(speaker) {{
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        if (!conversationMode) {{
          return {{ rate: preferredRate, pitch: preferredPitch }};
        }}
        if (speaker === "A") {{
          return {{
            rate: clamp(preferredRate - 0.03, 0.5, 2.0),
            pitch: clamp(preferredPitch - 0.05, 0.5, 2.0)
          }};
        }}
        return {{
          rate: clamp(preferredRate + 0.05, 0.5, 2.0),
          pitch: clamp(preferredPitch + 0.18, 0.5, 2.0)
        }};
      }}

      async function speakItem(item) {{
        const voices = await getVoicesAsync();

        const utter = new SpeechSynthesisUtterance(item.text);

        const tone = speakerTone(item.speaker);
        utter.rate = tone.rate;
        utter.pitch = tone.pitch;

        utter.lang = pidginMode ? "en-NG" : preferredLang;

        const uriA = voiceSelectA.value || "";
        const uriB = voiceSelectB.value || "";
        const preferredUri = (conversationMode && item.speaker === "B") ? uriB : uriA;

        let chosen = getVoiceByUri(voices, preferredUri);
        if (!chosen) {{
          const defA = pickDefaultVoice(voices);
          const defB = pickSecondaryVoice(voices, defA);
          chosen = (conversationMode && item.speaker === "B") ? (defB || defA) : defA;
        }}

        if (chosen) {{
          utter.voice = chosen;
          if (chosen.lang) utter.lang = chosen.lang;
        }}

        utter.onstart = () => {{
          if (conversationMode) {{
            setStatus(item.speaker === "A" ? "Speaking… (A)" : "Speaking… (B)");
          }} else {{
            setStatus("Speaking…");
          }}
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

        synth.cancel();
        speakItem(item);

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

        if (!queue.length) {{
          setStatus("Nothing to read.");
          setPlayPauseLabel();
          return;
        }}

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

        if (queue.length && idx < queue.length) {{
          setStatus("Speaking…");
          speakNext();
          setPlayPauseLabel();
          return;
        }}

        startFresh();
      }}

      function onVoiceChanged() {{
        setSaved(storageKeyA, voiceSelectA.value || "");
        setSaved(storageKeyB, voiceSelectB.value || "");
        if (ensureSpeechSupport() && (window.speechSynthesis.speaking || window.speechSynthesis.paused || betweenTimer || betweenPaused)) {{
          stopAll();
        }} else {{
          setStatus("Voice selected.");
          setPlayPauseLabel();
        }}
      }}

      voiceSelectA.addEventListener("change", onVoiceChanged);
      voiceSelectB.addEventListener("change", onVoiceChanged);

      resetVoiceBtn.addEventListener("click", async () => {{
        removeSaved(storageKeyA);
        removeSaved(storageKeyB);

        const voices = await getVoicesAsync();
        setDefaultsForAandB(voices);

        if (ensureSpeechSupport() && (window.speechSynthesis.speaking || window.speechSynthesis.paused || betweenTimer || betweenPaused)) stopAll();
        setStatus("Reset to defaults.");
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
    st.components.v1.html(html, height=185)


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
    st.subheader("Narration (DEFAULT: Nigerian Pidgin)")

    narration_language = st.selectbox(
        "Language",
        ["Nigerian Pidgin (default)", "English (original)"],
        index=0,
        help="Pidgin uses local, rule-based translation (no API keys). For best accent, install an en-NG voice if available.",
    )
    pidgin_mode = narration_language.startswith("Nigerian Pidgin")

    pidgin_intensity = st.select_slider(
        "Pidgin intensity",
        options=[0, 1, 2],
        value=1,
        help="0 = off, 1 = light, 2 = stronger (still conservative).",
        disabled=not pidgin_mode,
    )

    preferred_lang = st.selectbox(
        "Preferred voice language (hint)",
        ["en-NG", "en-GB", "en-US", "en", "es-ES", "fr-FR"],
        index=0 if pidgin_mode else 1,
        help="This only nudges default voice selection; your browser voices still rule.",
    )

    narration_style = st.selectbox(
        "Style",
        ["Chatty", "Debate", "Standard"],
        index=0,
        help="Chatty/Debate alternate Voice A/B with short ‘turn’ pauses.",
    )

    rate = st.slider("Rate", 0.5, 2.0, 0.95, 0.05)
    pitch = st.slider("Pitch", 0.5, 2.0, 0.78, 0.05, help="Lower pitch generally sounds deeper.")

    prefer_deep_male_gb = st.toggle(
        "Prefer deep male UK voice (Voice A bias)",
        value=False if pidgin_mode else True,
        help="If you want the old deep-UK default, switch language to en-GB and enable this.",
    )

    if narration_style in ("Chatty", "Debate"):
        turn_pause_ms = st.slider("Pause between speakers (ms)", 0, 1200, 260, 20)
    else:
        turn_pause_ms = 0

    st.divider()
    st.subheader("Flow (pauses)")

    flow_mode = st.toggle(
        "Flow mode (sentence pacing)",
        value=True,
        help="Speaks sentence-by-sentence with short pauses for rhythm and clarity.",
    )
    sentence_pause_ms = st.slider("Pause after sentence (ms)", 0, 1500, 320, 20)
    paragraph_pause_ms = st.slider("Pause after paragraph (ms)", 0, 4000, 650, 50)
    clause_pause_ms = st.slider(
        "Optional pause after clause (ms)",
        0,
        800,
        0,
        10,
        help="If > 0, splits sentences by commas/;/: and adds a light pause.",
    )

# Current content
cur_si, cur_ti, cur_ui = flat[st.session_state.flat_index]
cur_subject = subjects[cur_si]["subject"]
cur_topic = subjects[cur_si]["topics"][cur_ti]["topic"]
cur_subtopic = subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui]["subtopic"]
cur_text = (subjects[cur_si]["topics"][cur_ti]["subtopics"][cur_ui].get("full_text") or "").strip()

pidgin_text = pidgin_translate_text(cur_text, intensity=int(pidgin_intensity)) if cur_text else ""
text_to_read = pidgin_text if pidgin_mode else cur_text

col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    st.subheader(f"{cur_subject}  →  {cur_topic}  →  {cur_subtopic}")

    if cur_text:
        tab_pidgin, tab_original = st.tabs(["Pidgin (default)", "Original"])
        with tab_pidgin:
            if pidgin_text:
                st.write(pidgin_text)
            else:
                st.info("No text to translate here.")
        with tab_original:
            st.write(cur_text)
    else:
        st.info("No paragraph text under this subtopic.")

with col_right:
    st.subheader("Controls")

    st.markdown("**Listen**")
    if text_to_read:
        tts_component(
            text_to_read,
            preferred_lang=preferred_lang,
            rate=rate,
            pitch=pitch,
            prefer_deep_male_gb=prefer_deep_male_gb,
            narration_style=narration_style,
            flow_mode=flow_mode,
            sentence_pause_ms=sentence_pause_ms,
            paragraph_pause_ms=paragraph_pause_ms,
            clause_pause_ms=clause_pause_ms,
            turn_pause_ms=turn_pause_ms,
            pidgin_mode=pidgin_mode,
            pidgin_transform_enabled=False,
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
