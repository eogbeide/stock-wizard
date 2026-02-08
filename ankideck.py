# easymcat.py
# Streamlit PDF Study Reader
# - Sidebar navigation by PAGE NUMBER
# - Browser Text-to-Speech (installed voices dropdown)
# - Flow reading: sentence/paragraph pauses (optionally clause pauses)
# - Sticky floating Controls (Listen + Next/Back) on the page
# - Reads mathematical symbols more clearly (without turning prose dashes into “minus”)
# - Reads chemical formulas + simple chemical equations more clearly (H2O, NaCl, (NH4)2SO4, CuSO4·5H2O, Fe3+, etc.)
# - Better PDF text extraction + page text structuring into easy-to-narrate paragraphs & bullet “story mode”
# - NEW: Dedicated Resume button + robust multi-pause/multi-resume without restarting
# - NEW: Time progress + draggable scrubber (seek forward/back + pause by dragging)
#
# Run:
#   streamlit run easymcat.py
#
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/ankideck.pdf"

import ioe
import re
import uuid
from typing import List, Dict, Tuple

import requests
import streamlit as st

# PDF reader (supports pypdf or PyPDF2)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
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


def _smart_dash_vs_minus(s: str) -> str:
    """
    Protect prose dashes from being spoken as subtraction.

    Converts ' word - word ' into an em dash 'word — word' UNLESS it looks like math:
      - digit - digit
      - short token - short token (1-3 chars), e.g. x - y

    Also normalizes en-dash to em-dash for clearer pause.
    """
    if not s:
        return ""

    s = s.replace("–", "—")

    def repl(m: re.Match) -> str:
        left = m.group(1)
        right = m.group(2)

        if re.fullmatch(r"\d+(?:\.\d+)?", left) and re.fullmatch(r"\d+(?:\.\d+)?", right):
            return f"{left} - {right}"

        if re.fullmatch(r"[A-Za-z0-9]{1,3}", left) and re.fullmatch(r"[A-Za-z0-9]{1,3}", right):
            return f"{left} - {right}"

        return f"{left} — {right}"

    s = re.sub(r"\b([A-Za-z0-9]{2,})\s-\s([A-Za-z0-9]{2,})\b", repl, s)
    return s


def _replace_minus_in_math_context(s: str) -> str:
    """
    Convert spaced hyphen ' - ' to ' minus ' ONLY when it looks like subtraction:
      - digit - digit
      - short token - short token (1–3 chars) e.g., x - y, Na - Cl
    Python 3.13 safe: no lookbehind.
    """
    if not s:
        return ""

    s = re.sub(r"(\d)\s-\s(\d)", r"\1 minus \2", s)

    def _tok_minus(m: re.Match) -> str:
        a = m.group(1)
        b = m.group(2)
        return f"{a} minus {b}"

    s = re.sub(r"\b([A-Za-z0-9]{1,3})\s-\s([A-Za-z0-9]{1,3})\b", _tok_minus, s)
    return s


def make_math_speakable(text: str, style: str = "Natural") -> str:
    if not text:
        return ""

    s = _smart_dash_vs_minus(text)

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
        s = s.replace("*", " times ")
        s = _replace_minus_in_math_context(s)
    else:
        s = s.replace("=", " equals ")
        s = s.replace("+", " plus ")
        s = s.replace("*", " times ")
        s = _replace_minus_in_math_context(s)

        s = re.sub(r"(\b[A-Za-z][A-Za-z0-9]*)\s*\^\s*2\b", r"\1 squared", s)
        s = re.sub(r"(\b[A-Za-z][A-Za-z0-9]*)\s*\^\s*3\b", r"\1 cubed", s)
        s = re.sub(r"(\b[A-Za-z0-9][A-Za-z0-9]*)\s*\^\s*([A-Za-z0-9]+)\b", r"\1 to the power \2", s)

        s = re.sub(r"(\b[A-Za-z0-9]+)\s*_\s*([A-Za-z0-9]+)\b", r"\1 sub \2", s)
        s = re.sub(r"\(\s*([^)]+?)\s*\)\s*/\s*\(\s*([^)]+?)\s*\)", r" \1 over \2 ", s)
        s = s.replace(" / ", " over ")

    s = s.replace("—", " — ")
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
    for k, v in _SUPERSCRIPT_UNICODE.items():
        s = s.replace(k, v.replace("^", ""))
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

        if ch.isdigit():
            j = i
            num = ""
            while j < len(t) and t[j].isdigit():
                num += t[j]
                j += 1
            out.append(number_to_words(num))
            i = j
            continue

        if ch.isalpha() and ch.isupper():
            sym = ch
            if i + 1 < len(t) and t[i + 1].isalpha() and t[i + 1].islower():
                sym += t[i + 1]
                i += 2
            else:
                i += 1
            out.append(_speak_element(sym, element_mode))
            continue

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
# Page structuring helpers
# -----------------------------
_BULLET_RE = re.compile(r"^\s*(?:•|\u2022|-\s+|–\s+|\*\s+|·\s+)\s*(.+?)\s*$")
_NUMBERED_RE = re.compile(r"^\s*(\d+)[\)\.]\s+(.+?)\s*$")


def _split_sentences_py(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p and p.strip()]


def _chunk_sentences(sentences: List[str], max_sentences: int = 2) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for s in sentences:
        buf.append(s)
        if len(buf) >= max_sentences:
            out.append(" ".join(buf).strip())
            buf = []
    if buf:
        out.append(" ".join(buf).strip())
    return out


def structure_page_text(raw: str, fun_mode: bool = True) -> Tuple[str, str]:
    if not raw:
        return "", ""

    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    lines = [ln.strip() for ln in s.split("\n")]

    blocks: List[Dict] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal buf
        if buf:
            blocks.append({"type": "para", "text": " ".join(buf).strip()})
            buf = []

    for ln in lines:
        if not ln:
            flush_buf()
            continue

        bm = _BULLET_RE.match(ln)
        nm = _NUMBERED_RE.match(ln)

        if bm:
            flush_buf()
            blocks.append({"type": "bullet", "text": bm.group(1).strip()})
            continue

        if nm:
            flush_buf()
            blocks.append({"type": "number", "num": nm.group(1), "text": nm.group(2).strip()})
            continue

        if blocks and blocks[-1]["type"] in ("bullet", "number"):
            if (ln[:1].islower()) or (len(ln) <= 80 and not ln.isupper()):
                blocks[-1]["text"] = (blocks[-1]["text"] + " " + ln).strip()
                continue

        if len(ln) <= 60 and (ln.isupper() or ln.endswith(":")):
            flush_buf()
            blocks.append({"type": "heading", "text": ln.strip(":").strip()})
            continue

        buf.append(ln)

    flush_buf()

    display_parts: List[str] = []
    narr_parts: List[str] = []

    def narr_heading(h: str) -> str:
        if not fun_mode:
            return f"{h}."
        return f"Section: {h}. Ready? Let’s go."

    bullet_intro_added = False
    bullet_counter = 0
    bullet_words = ["First", "Next", "Then", "After that", "Finally"]

    for b in blocks:
        if b["type"] == "heading":
            h = b["text"].strip()
            if h:
                display_parts.append(f"### {h}")
                narr_parts.append(narr_heading(h))
                bullet_intro_added = False
                bullet_counter = 0
            continue

        if b["type"] in ("bullet", "number"):
            if not bullet_intro_added:
                if fun_mode:
                    narr_parts.append("Quick checklist time. Here are the key takeaways.")
                bullet_intro_added = True
                bullet_counter = 0
                display_parts.append("")

            bullet_counter += 1
            txt = b["text"].strip()

            if b["type"] == "number":
                display_parts.append(f"- **{b.get('num','')}**. {txt}")
            else:
                display_parts.append(f"- {txt}")

            if fun_mode:
                lead = bullet_words[min(bullet_counter - 1, len(bullet_words) - 1)]
                narr_parts.append(f"{lead}: {txt}.")
            else:
                narr_parts.append(f"{txt}.")
            continue

        p = (b.get("text") or "").strip()
        if not p:
            continue

        bullet_intro_added = False
        bullet_counter = 0

        sentences = _split_sentences_py(p)
        chunks = _chunk_sentences(sentences, max_sentences=2) if sentences else [p]

        for ci, chunk in enumerate(chunks):
            display_parts.append(chunk)
            if fun_mode and ci == 0 and len(chunks) > 1:
                narr_parts.append("Alright—here’s the idea.")
            narr_parts.append(chunk)

        display_parts.append("")
        narr_parts.append("")

    display_md = "\n\n".join([x for x in display_parts if x is not None]).strip()
    narration = "\n\n".join([x for x in narr_parts if x is not None]).strip()

    narration = _smart_dash_vs_minus(narration)
    narration = re.sub(r"[ \t]+", " ", narration)
    narration = re.sub(r"\n{3,}", "\n\n", narration).strip()

    return display_md, narration


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
        <button id="{component_id}_resume" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
          🔊 Resume
        </button>
        <button id="{component_id}_pause" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
          ⏸ Pause
        </button>
        <button id="{component_id}_stop" style="padding:8px 12px; border-radius:8px; border:1px solid #ddd; cursor:pointer;">
          ⏹ Stop
        </button>
        <span id="{component_id}_status" style="color:#666; font-size: 0.9rem;">Ready</span>
      </div>

      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <span id="{component_id}_time" style="color:#444; font-size:0.9rem; min-width:110px;">0:00 / 0:00</span>
        <input
          id="{component_id}_seek"
          type="range"
          min="0"
          max="0"
          value="0"
          step="100"
          style="flex: 1; min-width: 220px; accent-color: #555;"
          aria-label="Seek"
        />
        <span style="color:#888; font-size:0.85rem;">Drag to seek (auto-pauses)</span>
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
      const resumeBtn = document.getElementById(ROOT_ID + "_resume");
      const pauseBtn = document.getElementById(ROOT_ID + "_pause");
      const stopBtn = document.getElementById(ROOT_ID + "_stop");

      const seekEl = document.getElementById(ROOT_ID + "_seek");
      const timeEl = document.getElementById(ROOT_ID + "_time");

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

      // Scrub precision: smaller chunks => more accurate seeking
      const enableScrub = true;
      const microChunkWords = 12;

      const storageKey = "easymcat_tts_voice_uri";

      let queue = [];
      let idx = 0;

      // Between-items timer state (pauseAfter)
      let betweenTimer = null;
      let betweenDueAt = 0;
      let betweenRemaining = 0;
      let betweenPaused = false;

      // Track utterance time accounting (for progress)
      let utterStartAt = 0;
      let utterPausedAt = 0;
      let utterPausedAccum = 0;

      // Timeline estimation
      let timeline = [];     // [{speakMs, pauseMs, totalMs}]
      let starts = [];       // cumulative start per item
      let totalMs = 0;

      // Current "now" context
      let currentItemIndex = -1;
      let manualSeekPosMs = 0;
      let manualSeekPosValid = false;

      // Dragging state
      let userDragging = false;

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

      function clamp(n, lo, hi) {{
        return Math.max(lo, Math.min(hi, n));
      }}

      function msToClock(ms) {{
        const total = Math.max(0, Math.round(ms / 1000));
        const h = Math.floor(total / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
        const ss = String(s).padStart(2, "0");
        return h > 0 ? `${{h}}:${{mm}}:${{ss}}` : `${{mm}}:${{ss}}`;
      }}

      function setTimeUI(posMs) {{
        const p = clamp(posMs, 0, totalMs || 0);
        timeEl.textContent = `${{msToClock(p)}} / ${{msToClock(totalMs)}}`;
      }}

      function estimateSpeechMs(text) {{
        // Heuristic: estimate based on words; scales with speech rate.
        const words = ((text || "").trim().match(/\\S+/g) || []).length;
        const wpmBase = 170;
        const wpm = Math.max(80, wpmBase * Math.max(0.2, preferredRate));
        const minutes = words / wpm;
        const ms = minutes * 60 * 1000;
        return Math.max(280, ms);
      }}

      function splitIntoWordChunks(text, maxWords) {{
        const t = (text || "").trim();
        if (!t) return [];
        const words = t.split(/\\s+/g);
        if (words.length <= maxWords) return [t];
        const out = [];
        for (let i = 0; i < words.length; i += maxWords) {{
          out.push(words.slice(i, i + maxWords).join(" "));
        }}
        return out;
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
              // Single-item mode
              items.push({{ text: normalizeText(t), pauseAfter: 0 }});
              // break out
              pi = paragraphs.length;
              break;
            }}
          }}
        }}

        const base = items.length ? items : [{{ text: normalizeText(t), pauseAfter: 0 }}];

        if (!enableScrub) return base;

        // Make seeking finer: split each item into smaller word chunks.
        const refined = [];
        for (const it of base) {{
          const chunks = splitIntoWordChunks(it.text || "", microChunkWords);
          if (!chunks.length) continue;

          if (chunks.length === 1) {{
            refined.push({{ text: chunks[0], pauseAfter: Math.max(0, it.pauseAfter || 0) }});
          }} else {{
            for (let k = 0; k < chunks.length; k++) {{
              refined.push({{ text: chunks[k], pauseAfter: 0 }});
            }}
            refined[refined.length - 1].pauseAfter = Math.max(0, it.pauseAfter || 0);
          }}
        }}
        return refined.length ? refined : base;
      }}

      function clearBetweenTimer() {{
        if (betweenTimer) {{
          clearTimeout(betweenTimer);
          betweenTimer = null;
        }}
      }}

      function rebuildTimeline() {{
        timeline = [];
        starts = [];
        totalMs = 0;

        let cum = 0;
        for (let i = 0; i < queue.length; i++) {{
          const speakMs = estimateSpeechMs(queue[i].text || "");
          const pauseMs = Math.max(0, queue[i].pauseAfter || 0);
          const total = speakMs + pauseMs;
          timeline.push({{ speakMs, pauseMs, totalMs: total }});
          starts.push(cum);
          cum += total;
        }}
        totalMs = cum;

        seekEl.max = String(Math.max(0, totalMs));
        seekEl.value = "0";
        setTimeUI(0);
      }}

      function cancelPlaybackKeepQueue() {{
        if (!ensureSpeechSupport()) return;

        clearBetweenTimer();
        betweenPaused = false;
        betweenRemaining = 0;

        utterStartAt = 0;
        utterPausedAt = 0;
        utterPausedAccum = 0;

        window.speechSynthesis.cancel();
      }}

      function stopAll() {{
        if (!ensureSpeechSupport()) return;

        clearBetweenTimer();
        betweenPaused = false;
        betweenRemaining = 0;

        utterStartAt = 0;
        utterPausedAt = 0;
        utterPausedAccum = 0;

        manualSeekPosValid = false;
        manualSeekPosMs = 0;

        window.speechSynthesis.cancel();
        queue = [];
        idx = 0;
        currentItemIndex = -1;

        rebuildTimeline(); // resets seek UI
        setStatus("Stopped.");
      }}

      function pauseAll() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        // If we are between items (timer running), pause the timer
        if (betweenTimer) {{
          const now = Date.now();
          betweenRemaining = Math.max(0, betweenDueAt - now);
          clearBetweenTimer();
          betweenPaused = true;
          setStatus("Paused (between items).");
          return;
        }}

        // If currently speaking, pause speech
        if (synth.speaking && !synth.paused) {{
          utterPausedAt = Date.now();
          synth.pause();
          setStatus("Paused.");
          return;
        }}

        // If already paused, do nothing
        if (synth.paused || betweenPaused) {{
          setStatus("Paused.");
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
          utterStartAt = Date.now();
          utterPausedAccum = 0;
          utterPausedAt = 0;
          setStatus("Speaking…");
        }};

        utter.onend = () => {{
          // no-op; progress uses watcher
        }};

        utter.onerror = () => {{
          setStatus("TTS error.");
        }};

        window.speechSynthesis.speak(utter);
      }}

      function speakNext() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        if (!queue.length) {{
          setStatus("Nothing to read.");
          return;
        }}

        if (idx >= queue.length) {{
          setStatus("Done.");
          manualSeekPosValid = false;
          manualSeekPosMs = totalMs;
          return;
        }}

        // Clear any manual seek "cursor" once we begin speaking from it
        manualSeekPosValid = false;

        const item = queue[idx];
        currentItemIndex = idx;
        idx += 1;

        // Important: don't cancel while paused; only cancel when moving to a new item
        if (!synth.paused) {{
          synth.cancel();
        }}

        speakItem(item.text);

        const pauseAfter = Math.max(0, item.pauseAfter || 0);

        const watcher = setInterval(() => {{
          if (!ensureSpeechSupport()) {{
            clearInterval(watcher);
            return;
          }}
          const s = window.speechSynthesis;

          // If user paused, don't schedule next
          if (s.paused) {{
            clearInterval(watcher);
            return;
          }}

          // If not speaking (and not paused), schedule the next
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

        utterStartAt = 0;
        utterPausedAt = 0;
        utterPausedAccum = 0;

        manualSeekPosValid = false;
        manualSeekPosMs = 0;

        window.speechSynthesis.cancel();

        queue = buildQueueFromText(TEXT);
        idx = 0;
        currentItemIndex = -1;

        rebuildTimeline();
        seekEl.value = "0";
        setTimeUI(0);

        speakNext();
      }}

      function resumeAll() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        // Resume between-items pause
        if (betweenPaused) {{
          betweenPaused = false;
          const delay = Math.max(0, betweenRemaining);
          betweenRemaining = 0;
          setStatus("Speaking…");
          scheduleNext(delay);
          return;
        }}

        // Resume a paused utterance
        if (synth.paused) {{
          if (utterPausedAt) {{
            utterPausedAccum += (Date.now() - utterPausedAt);
            utterPausedAt = 0;
          }}
          synth.resume();
          setStatus("Speaking…");
          return;
        }}

        // If not paused but we have a queue and we're not speaking, continue
        if (queue.length && idx < queue.length && !synth.speaking) {{
          setStatus("Speaking…");
          speakNext();
          return;
        }}

        // If nothing has started yet, start fresh
        if (!queue.length) {{
          startFresh();
        }}
      }}

      function togglePlayPause() {{
        if (!ensureSpeechSupport()) return;

        const synth = window.speechSynthesis;

        // If paused, resume
        if (betweenPaused || synth.paused) {{
          resumeAll();
          return;
        }}

        // If currently speaking or between items, pause
        if (betweenTimer || (synth.speaking && !synth.paused)) {{
          pauseAll();
          return;
        }}

        // If we have a queue and are mid-way, continue
        if (queue.length && idx < queue.length) {{
          setStatus("Speaking…");
          speakNext();
          return;
        }}

        // Otherwise start new
        startFresh();
      }}

      function findIndexForMs(ms) {{
        const t = clamp(ms, 0, totalMs);
        if (!starts.length) return 0;
        // Linear scan is fine for typical page sizes; binary search if you want later.
        for (let i = starts.length - 1; i >= 0; i--) {{
          if (t >= starts[i]) return i;
        }}
        return 0;
      }}

      function seekToMs(ms) {{
        if (!ensureSpeechSupport()) return;

        // If not started, build queue/timeline first (but do not autoplay).
        if (!queue.length) {{
          queue = buildQueueFromText(TEXT);
          idx = 0;
          currentItemIndex = -1;
          rebuildTimeline();
        }}

        const t = clamp(ms, 0, totalMs);
        const targetIndex = findIndexForMs(t);

        cancelPlaybackKeepQueue();

        idx = targetIndex;
        currentItemIndex = -1;

        manualSeekPosValid = true;
        manualSeekPosMs = t;

        seekEl.value = String(Math.round(t));
        setTimeUI(t);

        setStatus("Seeked. Press Resume to continue.");
      }}

      function computeProgressMs() {{
        if (!queue.length || !timeline.length) {{
          return 0;
        }}

        // If user has seeked and we haven't started speaking yet, show that exact cursor
        if (manualSeekPosValid) {{
          return clamp(manualSeekPosMs, 0, totalMs);
        }}

        const synth = window.speechSynthesis;

        // Between-items countdown (silent pauseAfter)
        if (betweenTimer || betweenPaused) {{
          const i = currentItemIndex;
          if (i >= 0 && i < timeline.length) {{
            const base = starts[i] + timeline[i].speakMs;
            const remaining = betweenTimer ? Math.max(0, betweenDueAt - Date.now()) : Math.max(0, betweenRemaining);
            const elapsedPause = clamp(timeline[i].pauseMs - remaining, 0, timeline[i].pauseMs);
            return clamp(base + elapsedPause, 0, totalMs);
          }}
        }}

        // Speaking or paused mid-utterance
        if (synth.speaking || synth.paused) {{
          const i = currentItemIndex;
          if (i >= 0 && i < timeline.length) {{
            const elapsed = clamp(Date.now() - utterStartAt - utterPausedAccum, 0, timeline[i].speakMs);
            return clamp(starts[i] + elapsed, 0, totalMs);
          }}
        }}

        // Not speaking: use idx cursor as progress
        if (idx >= timeline.length) return totalMs;
        return clamp(starts[idx] || 0, 0, totalMs);
      }}

      function updateProgressUI() {{
        if (userDragging) return;
        const pos = computeProgressMs();
        if (seekEl.max !== String(Math.max(0, totalMs))) {{
          seekEl.max = String(Math.max(0, totalMs));
        }}
        seekEl.value = String(Math.round(pos));
        setTimeUI(pos);
      }}

      // Update progress periodically
      setInterval(() => {{
        try {{
          updateProgressUI();
        }} catch (e) {{}}
      }}, 200);

      // Scrubber interactions (drag => auto-pause + seek; stays paused after drag)
      function beginDrag() {{
        userDragging = true;
        pauseAll();
        setStatus("Paused (seeking)…");
      }}

      function endDrag() {{
        userDragging = false;
        const target = Number(seekEl.value || "0");
        seekToMs(target);
      }}

      seekEl.addEventListener("mousedown", beginDrag);
      seekEl.addEventListener("touchstart", beginDrag, {{ passive: true }});

      seekEl.addEventListener("input", () => {{
        // While dragging, show the "target time" live
        const target = Number(seekEl.value || "0");
        setTimeUI(target);
      }});

      seekEl.addEventListener("mouseup", endDrag);
      seekEl.addEventListener("touchend", endDrag);

      // Also handle keyboard / click seek changes
      seekEl.addEventListener("change", () => {{
        if (userDragging) return;
        const target = Number(seekEl.value || "0");
        seekToMs(target);
      }});

      // UI wiring
      playPauseBtn.addEventListener("click", togglePlayPause);
      pauseBtn.addEventListener("click", pauseAll);
      resumeBtn.addEventListener("click", resumeAll);
      stopBtn.addEventListener("click", stopAll);

      voiceSelect.addEventListener("change", () => {{
        try {{ localStorage.setItem(storageKey, voiceSelect.value || ""); }} catch (e) {{}}
        // Voice changes mid-speech can be flaky; stop to avoid strange state.
        if (ensureSpeechSupport() && (window.speechSynthesis.speaking || window.speechSynthesis.paused || betweenTimer || betweenPaused)) {{
          stopAll();
        }} else {{
          setStatus("Voice selected.");
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
        if (ensureSpeechSupport() && (window.speechSynthesis.speaking || window.speechSynthesis.paused || betweenTimer || betweenPaused)) stopAll();
      }});

      initVoices();

      // Initialize seek UI for first render (no queue yet)
      timeEl.textContent = "0:00 / 0:00";
      seekEl.max = "0";
      seekEl.value = "0";
    </script>
    """
    st.components.v1.html(html, height=210)


# -----------------------------
# PDF helpers
# -----------------------------
def fetch_pdf_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content


def parse_pdf_to_pages(pdf_bytes: bytes) -> List[Dict]:
    """
    Returns:
      pages = [{"page": 1-based int, "raw": str}, ...]
    Tries multiple extraction strategies for reliability.
    """
    if PdfReader is None:
        raise RuntimeError("No PDF reader installed. Install 'pypdf' (recommended) or 'PyPDF2'.")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[Dict] = []

    for i, page in enumerate(getattr(reader, "pages", [])):
        raw = ""

        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""

        if not raw.strip():
            try:
                raw = page.extract_text(extraction_mode="layout") or ""
            except Exception:
                pass

        if not raw.strip():
            try:
                raw = (page.extract_text() or "").strip()
            except Exception:
                raw = ""

        pages.append({"page": i + 1, "raw": raw})

    return pages


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="PDF Study Reader", layout="wide")
st.title("PDF Study Reader")
st.caption("Sidebar: Page navigation • Page: structured content + floating Controls (Listen + Next/Back)")

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

if "page_index" not in st.session_state:
    st.session_state.page_index = 0
st.session_state.page_index = max(0, min(st.session_state.page_index, num_pages - 1))

with st.sidebar:
    st.header("Navigate")
    page_options = [f"Page {i}" for i in range(1, num_pages + 1)]
    default_idx = st.session_state.page_index

    selected_label = st.selectbox("Page", page_options, index=default_idx)
    selected_idx = page_options.index(selected_label)

    if st.button("Go", use_container_width=True):
        st.session_state.page_index = selected_idx
        st.rerun()

    st.divider()
    st.subheader("Page formatting")

    fun_mode = st.toggle(
        "Make it narration-friendly (story mode)",
        value=True,
        help="Structures extracted text into shorter paragraphs, headings, and a friendly bullet narration style.",
    )
    show_raw = st.toggle(
        "Show raw extracted text (debug)",
        value=False,
        help="Helpful if the PDF is hard to extract from (scanned or complex layout).",
    )

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


cur_page_num = st.session_state.page_index + 1
raw_text = (pages[st.session_state.page_index].get("raw") or "").strip()

display_md, base_tts_text = structure_page_text(raw_text, fun_mode=fun_mode)

if not raw_text.strip():
    display_md = ""
    base_tts_text = ""

tts_text = base_tts_text

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

    if show_raw and raw_text:
        with st.expander("Raw extracted text (from PDF extractor)"):
            st.write(raw_text)

    if raw_text:
        if display_md:
            st.markdown(display_md)
        else:
            st.write(raw_text)

        if show_tts_preview and tts_text:
            with st.expander("TTS preview (what will be spoken)"):
                st.write(tts_text)
    else:
        st.info(
            "No extractable text on this page.\n\n"
            "This usually means the page is image-only/scanned (no embedded text layer). "
            "If you need those pages narrated, add OCR support."
        )

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
