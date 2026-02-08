# easymcat.py
# Streamlit PDF Study Reader
# - Sidebar navigation by PAGE NUMBER
# - Browser Text-to-Speech (installed voices dropdown)
# - Flow reading: sentence/paragraph pauses (optionally clause pauses)
# - Sticky floating Controls (Listen + Next/Back) on the page
# - Reads mathematical symbols more clearly
# - Reads chemical formulas + simple chemical equations more clearly (H2O, NaCl, (NH4)2SO4, CuSO4·5H2O, Fe3+, etc.)
#
# Run:
#   streamlit run easymcat.py
#
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/ankideck.pdf"

import io
import re
import uuid
from typing import Optional, List, Dict, Tuple

import requests
import streamlit as st

# PDF reader (supports pypdf or PyPDF2)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as e:
        PdfReader = None  # type: ignore


# -----------------------------
# Speakable helpers: numbers, math, chemistry
# -----------------------------
_NUM_WORDS_0_19 = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def number_to_words(n: str) -> str:
    """Best-effort small number-to-words for TTS."""
    try:
        val = int(n)
    except Exception:
        return " ".join(list(n))

    if val < 0:
        return "minus " + number_to_words(str(-val))
    if val <= 19:
        return _NUM_WORDS_0_19[val]
    if val < 100:
        tens = (val // 10) * 10
        ones = val % 10
        if ones == 0:
            return _TENS.get(tens, str(val))
        return f"{_TENS.get(tens, str(tens))} {_NUM_WORDS_0_19.get(ones, str(ones))}"
    # For 100+ just speak digits (safer than odd “one hundred and …”)
    return " ".join(list(str(val)))


_SUPERSCRIPT_UNICODE = {
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
}

_SUBSCRIPT_UNICODE = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
}

_SUPER_SIGNS = {"⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")"}

_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}

# Common element names (fallback is symbol itself)
ELEMENT_NAMES = {
    "H": "hydrogen",
    "He": "helium",
    "Li": "lithium",
    "Be": "beryllium",
    "B": "boron",
    "C": "carbon",
    "N": "nitrogen",
    "O": "oxygen",
    "F": "fluorine",
    "Ne": "neon",
    "Na": "sodium",
    "Mg": "magnesium",
    "Al": "aluminium",
    "Si": "silicon",
    "P": "phosphorus",
    "S": "sulfur",
    "Cl": "chlorine",
    "Ar": "argon",
    "K": "potassium",
    "Ca": "calcium",
    "Cr": "chromium",
    "Mn": "manganese",
    "Fe": "iron",
    "Co": "cobalt",
    "Ni": "nickel",
    "Cu": "copper",
    "Zn": "zinc",
    "Br": "bromine",
    "Ag": "silver",
    "I": "iodine",
    "Sn": "tin",
    "Au": "gold",
    "Hg": "mercury",
    "Pb": "lead",
}

COMMON_ACRONYM_BLACKLIST = {"US", "UK", "EU", "UN", "USA", "UAE", "NATO"}


def make_math_speakable(text: str, style: str = "Natural") -> str:
    if not text:
        return ""

    s = text

    for k, v in _SUPERSCRIPT_UNICODE.items():
        s = s.replace(k, v)

    for sym, name in _GREEK.items():
        s = s.replace(sym, f" {name} ")

    s = s.replace("≤", " less than or equal to ")
    s = s.replace("≥", " greater than or equal to ")
    s = s.replace("≠", " not equal to ")
    s = s.replace("≈", " approximately equal to ")
    s = s.replace("∝", " proportional to ")
    s = s.replace("∞", " infinity ")
    s = s.replace("→", " tends to ")
    s = s.replace("∑", " summation ")
    s = s.replace("∫", " integral ")
    s = s.replace("√", " square root of ")
    s = s.replace("×", " times ")
    s = s.replace("·", " times ")

    def _root_repl(m: re.Match) -> str:
        deg = m.group(1).strip()
        expr = m.group(2).strip()
        if deg == "2":
            return f" square root of {expr} "
        if deg == "3":
            return f" cube root of {expr} "
        if deg == "4":
            return f" fourth root of {expr} "
        return f" {deg}th root of {expr} "

    s = re.sub(r"\broot\s*\(\s*([0-9]+)\s*,\s*([^)]+)\)", _root_repl, s, flags=re.IGNORECASE)
    s = re.sub(r"\bsqrt\s*\(\s*([^)]+)\)", r" square root of \1 ", s, flags=re.IGNORECASE)

    if style.lower().startswith("lit"):
        s = s.replace("^", " caret ")
        s = s.replace("/", " slash ")
        s = s.replace("=", " equals ")
        s = s.replace("+", " plus ")
        s = s.replace("-", " minus ")
        s = s.replace("*", " times ")
    else:
        s = s.replace("=", " equals ")
        s = s.replace("+", " plus ")
        s = re.sub(r"(?<=\s)-(?=\s)", " minus ", s)
        s = s.replace("*", " times ")

        s = re.sub(r"(\b[A-Za-z][A-Za-z0-9]*)\s*\^\s*2\b", r"\1 squared", s)
        s = re.sub(r"(\b[A-Za-z][A-Za-z0-9]*)\s*\^\s*3\b", r"\1 cubed", s)
        s = re.sub(r"(\b[A-Za-z0-9][A-Za-z0-9]*)\s*\^\s*([A-Za-z0-9]+)\b", r"\1 to the power \2", s)

        s = re.sub(r"(\b[A-Za-z0-9]+)\s*_\s*([A-Za-z0-9]+)\b", r"\1 sub \2", s)
        s = re.sub(r"\(\s*([^)]+?)\s*\)\s*/\s*\(\s*([^)]+?)\s*\)", r" \1 over \2 ", s)
        s = s.replace(" / ", " over ")

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


_FORMULA_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])(
      (?:\d+)?                                     # optional coefficient
      (?:
        (?:\([A-Za-z0-9+\-]+\)\d*)                  # (NH4)2
        |
        (?:[A-Z][a-z]?\d*)                          # Na2, Cl, O2
      )+
      (?:[·\.](?:\d+)?(?:[A-Z][a-z]?\d*)+)*         # ·5H2O  (also accepts .)
      (?:\^?\d*[+\-])?                              # optional charge (Fe3+ or ^2-)
    )(?![A-Za-z0-9])
    """,
    re.VERBOSE,
)


def _normalize_chem_unicode(s: str) -> str:
    if not s:
        return ""
    for k, v in _SUBSCRIPT_UNICODE.items():
        s = s.replace(k, v)
    for k, v in _SUPER_SIGNS.items():
        s = s.replace(k, v)
    # keep math superscripts too (sometimes used for charges)
    for k, v in _SUPERSCRIPT_UNICODE.items():
        s = s.replace(k, v.replace("^", ""))  # ² -> 2 in chemistry context
    # normalize hydration dot variants
    s = s.replace("∙", "·")
    return s


def _is_likely_formula(tok: str, include_single_element: bool) -> bool:
    if not tok:
        return False
    if tok in COMMON_ACRONYM_BLACKLIST:
        return False

    t = _normalize_chem_unicode(tok)
    elements = re.findall(r"[A-Z][a-z]?", t)
    if not elements:
        return False

    has_digit = bool(re.search(r"\d", t))
    has_group = "(" in t or ")" in t
    has_dot = "·" in t or "." in t
    has_charge = bool(re.search(r"(\^?\d*[+\-])$", t))

    if len(elements) >= 2:
        return True
    if len(elements) == 1 and (has_digit or has_group or has_dot or has_charge):
        return True
    if include_single_element and len(elements) == 1 and elements[0] in ELEMENT_NAMES:
        return True

    return False


def _speak_element(sym: str, element_mode: str) -> str:
    if element_mode == "Element names":
        return ELEMENT_NAMES.get(sym, sym)
    return " ".join(list(sym))


def _speak_formula(tok: str, element_mode: str) -> str:
    t = _normalize_chem_unicode(tok)

    # Common state symbols
    t = re.sub(r"\(aq\)", " (aq) ", t, flags=re.IGNORECASE)
    t = re.sub(r"\(s\)", " (s) ", t, flags=re.IGNORECASE)
    t = re.sub(r"\(l\)", " (l) ", t, flags=re.IGNORECASE)
    t = re.sub(r"\(g\)", " (g) ", t, flags=re.IGNORECASE)

    out: List[str] = []
    i = 0
    while i < len(t):
        ch = t[i]

        if ch.isspace():
            i += 1
            continue

        # state symbols
        if t[i : i + 4].lower() == "(aq)":
            out.append("aqueous")
            i += 4
            continue
        if t[i : i + 3].lower() == "(s)":
            out.append("solid")
            i += 3
            continue
        if t[i : i + 3].lower() == "(l)":
            out.append("liquid")
            i += 3
            continue
        if t[i : i + 3].lower() == "(g)":
            out.append("gas")
            i += 3
            continue

        if ch == "(":
            out.append("open bracket")
            i += 1
            continue
        if ch == ")":
            out.append("close bracket")
            i += 1
            continue
        if ch in ("·", "."):
            out.append("dot")
            i += 1
            continue

        # caret charge format: ^2- or ^-
        if ch == "^":
            j = i + 1
            num = ""
            while j < len(t) and t[j].isdigit():
                num += t[j]
                j += 1
            sign = ""
            if j < len(t) and t[j] in "+-":
                sign = t[j]
                j += 1
            if sign:
                if num:
                    out.append(f"{number_to_words(num)} {'plus' if sign == '+' else 'minus'} charge")
                else:
                    out.append(f"{'plus' if sign == '+' else 'minus'} charge")
            else:
                out.append("charge")
            i = j
            continue

        # digits (coefficients or subscripts)
        if ch.isdigit():
            j = i
            num = ""
            while j < len(t) and t[j].isdigit():
                num += t[j]
                j += 1
            out.append(number_to_words(num))
            i = j
            continue

        # element symbol
        if ch.isalpha() and ch.isupper():
            sym = ch
            if i + 1 < len(t) and t[i + 1].isalpha() and t[i + 1].islower():
                sym += t[i + 1]
                i += 2
            else:
                i += 1
            out.append(_speak_element(sym, element_mode))
            continue

        # trailing ionic signs
        if ch in "+-":
            out.append("plus" if ch == "+" else "minus")
            if i == len(t) - 1:
                out.append("charge")
            i += 1
            continue

        out.append(ch)
        i += 1

    spoken = " ".join(out)
    spoken = re.sub(r"[ \t]+", " ", spoken).strip()
    spoken = re.sub(r"\b(plus|minus)\s*$", r"\1 charge", spoken)

    return spoken


def make_chem_speakable(
    text: str,
    element_mode: str = "Element names",
    include_single_element: bool = False,
) -> str:
    if not text:
        return ""

    s = _normalize_chem_unicode(text)

    # reaction arrows
    s = s.replace("⇌", " reversible reaction ")
    s = s.replace("→", " yields ")
    s = s.replace("⇒", " yields ")
    s = s.replace("<-", " reacts to form ")
    s = s.replace("->", " yields ")

    def repl(m: re.Match) -> str:
        tok = m.group(1)
        if not _is_likely_formula(tok, include_single_element=include_single_element):
            return tok
        spoken = _speak_formula(tok, element_mode=element_mode)
        return f" {spoken} "

    s = re.sub(_FORMULA_RE, repl, s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


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
              return [{{ text: normalizeText(t), pauseAfter: 0 }}];
            }}
          }}
        }}

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

        synth.cancel();
        speakItem(item.text);

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
# PDF helpers
# -----------------------------
def fetch_pdf_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content


def _clean_pdf_text(s: str) -> str:
    if not s:
        return ""
    # normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # de-hyphenate common PDF line breaks: "exam-\nple" -> "example"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # collapse "single line breaks" into spaces, keep paragraph breaks
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # many PDFs put line breaks every line; turn lone newlines into spaces
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def parse_pdf_to_pages(pdf_bytes: bytes) -> List[Dict]:
    """
    Returns:
      pages = [{"page": 1-based int, "text": str}, ...]
    """
    if PdfReader is None:
        raise RuntimeError("No PDF reader installed. Install 'pypdf' (recommended) or 'PyPDF2'.")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[Dict] = []
    for i, page in enumerate(getattr(reader, "pages", [])):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({"page": i + 1, "text": _clean_pdf_text(txt)})
    return pages


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="PDF Study Reader", layout="wide")
st.title("PDF Study Reader")
st.caption("Sidebar: Page navigation • Page: content + floating Controls (Listen + Next/Back)")

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
def load_pages_from_url(url: str) -> List[Dict]:
    return parse_pdf_to_pages(fetch_pdf_bytes(url))


@st.cache_data(show_spinner=True)
def load_pages_from_upload(file_bytes: bytes) -> List[Dict]:
    return parse_pdf_to_pages(file_bytes)


with st.sidebar:
    st.header("Document Source")
    uploaded = st.file_uploader("Upload PDF (optional)", type=["pdf"])
    url = st.text_input("PDF URL", value=DEFAULT_URL, disabled=uploaded is not None)

    st.caption("Navigation is by page number. Use Next/Back controls on the page to move sequentially.")

# Load PDF pages
try:
    if uploaded is not None:
        pages = load_pages_from_upload(uploaded.getvalue())
    else:
        pages = load_pages_from_url(url)
except Exception as e:
    st.error(f"Could not load PDF.\n\nError: {e}")
    st.stop()

if not pages:
    st.warning("No pages found in this PDF.")
    st.stop()

num_pages = len(pages)

# Session state for page index
if "page_index" not in st.session_state:
    st.session_state.page_index = 0
st.session_state.page_index = max(0, min(st.session_state.page_index, num_pages - 1))

# Sidebar navigation by page number
with st.sidebar:
    st.header("Navigate")
    page_options = [f"Page {i}" for i in range(1, num_pages + 1)]
    current_label = f"Page {st.session_state.page_index + 1}"
    default_idx = st.session_state.page_index

    selected_label = st.selectbox("Page", page_options, index=default_idx)
    selected_idx = page_options.index(selected_label)

    if st.button("Go", use_container_width=True):
        st.session_state.page_index = selected_idx
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

    st.divider()
    st.subheader("Chemistry reading")

    read_chem = st.toggle(
        "Speak chemical formulas clearly",
        value=True,
        help="Turns formulas like H2O / NaCl / (NH4)2SO4 / CuSO4·5H2O into clearer spoken text.",
    )
    chem_element_mode = st.selectbox(
        "Chemical element style",
        ["Element names", "Spell symbols"],
        index=0,
        disabled=not read_chem,
        help="Element names: 'sodium chloride'. Spell symbols: 'N A C L'.",
    )
    include_single_element = st.toggle(
        "Include single-element symbols (e.g., Fe)",
        value=False,
        disabled=not read_chem,
        help="Off by default to reduce false positives (e.g., acronyms).",
    )

    st.divider()
    st.subheader("Math reading")

    read_math = st.toggle(
        "Speak equations clearly",
        value=True,
        help="Converts math symbols to words (PDFs won't have Word equation objects).",
    )
    math_style = st.selectbox("Math style", ["Natural", "Literal"], index=0, disabled=not read_math)

    st.divider()
    show_tts_preview = st.toggle("Show TTS text preview", value=False)

# Current page content
cur_page_num = st.session_state.page_index + 1
cur_text = (pages[st.session_state.page_index].get("text") or "").strip()

tts_text = cur_text
if tts_text and read_chem:
    tts_text = make_chem_speakable(
        tts_text,
        element_mode=chem_element_mode,
        include_single_element=include_single_element,
    )
if tts_text and read_math:
    tts_text = make_math_speakable(tts_text, style=math_style)

col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    st.subheader(f"Page {cur_page_num} of {num_pages}")

    if cur_text:
        st.write(cur_text)
        if show_tts_preview and tts_text and tts_text != cur_text:
            with st.expander("TTS preview (what will be spoken)"):
                st.write(tts_text)
    else:
        st.info("No extractable text on this page (it may be an image-only/scanned page).")

with col_right:
    st.subheader("Controls")

    st.markdown("**Listen**")
    if tts_text:
        tts_component(
            tts_text,
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
        st.caption("Nothing to read for this page.")

    st.divider()
    st.markdown("**Navigate**")

    if st.button("⬅️ Back", disabled=(st.session_state.page_index == 0), use_container_width=True):
        st.session_state.page_index -= 1
        st.rerun()

    if st.button("Next ➡️", disabled=(st.session_state.page_index == num_pages - 1), use_container_width=True):
        st.session_state.page_index += 1
        st.rerun()

    st.divider()
    st.caption("Progress")
    st.write(f"Page {cur_page_num} of {num_pages}")
