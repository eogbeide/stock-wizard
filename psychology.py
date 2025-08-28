# psychology.py — MCQs (no passages) + TTS autoplay + AUTO-DOWNLOAD MP3 + auto-next (10 MCQs/page)
import re
import json
import base64
from io import BytesIO

import requests
import streamlit as st
from gtts import gTTS
from streamlit.components.v1 import html

# .docx extraction
try:
    from docx import Document
    from docx.enum.text import WD_BREAK
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# ---------- App config ----------
st.set_page_config(page_title="📖 MCQs Reader (Auto TTS/MP3 Download/Next)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.docx"
MCQS_PER_PAGE = 10  # up to 10 MCQs per page

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def best_effort_bytes_to_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return "".join(chr(b) if 32 <= b <= 126 or b in (9, 10, 13) else " " for b in data)

def extract_docx_text(data: bytes) -> str:
    if not DOCX_OK:
        raise RuntimeError("python-docx not available. Add 'python-docx' to requirements.txt.")
    doc = Document(BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
        for run in para.runs:
            if getattr(run, "break_type", None) == WD_BREAK.PAGE:
                parts.append("\f")
    text = "\n".join(parts).replace("\f\n", "\f")
    return normalize_text(text)

# -------- Patterns --------
MCQ_START_PAT = re.compile(
    r"""^\s*(?:
            (?:Q(?:uestion)?\s*\d*)[.)\s:-]* |
            \d+\s*[.)-]\s+
        )""",
    re.IGNORECASE | re.VERBOSE,
)
OPTION_PAT = re.compile(r"""^\s*([A-Ha-h])\s*[\).:,-]\s+""", re.VERBOSE)
ANSWER_PAT = re.compile(r"""^\s*(?:answer|answers?|ans|correct\s*answer|key|solution)\s*[:\-]?\s*(.*)""", re.IGNORECASE)
EXPL_PAT   = re.compile(r"""^\s*(?:explanation|rationale|why|reason(?:ing)?)\s*[:\-]?\s*(.*)""", re.IGNORECASE)
PASSAGE_HEADER_PAT = re.compile(r"^\s*passage(\s*[ivx]+|\s*\d+|\s*[a-z])?\b", re.IGNORECASE)
QUESTIONS_HEADER_PAT = re.compile(r"^\s*questions?\b", re.IGNORECASE)

def looks_like_question(line: str) -> bool:
    return bool(MCQ_START_PAT.match(line))

def looks_like_option(line: str) -> bool:
    return bool(OPTION_PAT.match(line))

def parse_answer(line: str):
    m = ANSWER_PAT.match(line);  return m.group(1).strip() if m else None

def parse_expl(line: str):
    m = EXPL_PAT.match(line);    return m.group(1).strip() if m else None

def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()

def remove_passages(full_text: str) -> list[str]:
    """Drop passage blocks entirely."""
    lines = [l.rstrip() for l in full_text.split("\n")]
    filtered, in_passage = [], False
    for raw in lines:
        line = raw.strip()
        if PASSAGE_HEADER_PAT.match(line):
            in_passage = True
            continue
        if in_passage:
            if QUESTIONS_HEADER_PAT.match(line) or looks_like_question(line):
                in_passage = False
            else:
                continue
        if not in_passage:
            filtered.append(raw)
    return filtered

def extract_mcqs_with_answers(full_text: str):
    """Return MCQ blocks: stem + options + Answer/Explanation (passages removed)."""
    lines = remove_passages(full_text)
    mcqs = []
    cur_stem, cur_opts = [], []
    cur_answer, cur_expl = None, None
    in_mcq, options_started = False, False

    def flush():
        nonlocal cur_stem, cur_opts, cur_answer, cur_expl, in_mcq, options_started
        stem_text = " ".join([clean_line(s) for s in cur_stem if s is not None]).strip()
        if stem_text and len(cur_opts) >= 2:
            block = []
            block.append(stem_text)
            block.extend(cur_opts)
            if cur_answer:
                block.append(f"Answer: {cur_answer}")
            if cur_expl:
                block.append(f"Explanation: {cur_expl}")
            mcqs.append("\n".join(block).strip())
        cur_stem, cur_opts = [], []
        cur_answer, cur_expl = None, None
        in_mcq, options_started = False, False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_mcq and cur_stem and not options_started:
                cur_stem.append("")
            continue

        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq, options_started = True, False
            cur_stem = [clean_line(line)]
            cur_opts = []
            cur_answer, cur_expl = None, None
            continue

        if not in_mcq:
            continue

        if looks_like_option(line):
            options_started = True
            cur_opts.append(clean_line(line))
            continue

        ans = parse_answer(line)
        if ans is not None:
            cur_answer = ans or cur_answer or ""
            continue

        expl = parse_expl(line)
        if expl is not None:
            cur_expl = (cur_expl + " " + expl).strip() if cur_expl else expl
            continue

        if not options_started:
            cur_stem.append(clean_line(line))
        else:
            if cur_answer is not None or cur_expl is not None:
                cur_expl = (cur_expl + " " + clean_line(line)).strip() if cur_expl else clean_line(line)

    if in_mcq:
        flush()

    return mcqs

def mcqs_to_pages(mcqs, per_page: int):
    pages = []
    for i in range(0, len(mcqs), per_page):
        pages.append("\n\n".join(mcqs[i:i+per_page]))
    return pages or ["No MCQs (with options) found."]

def tts_mp3(text: str) -> BytesIO:
    step = 4500
    combined = BytesIO()
    for i in range(0, len(text), step):
        chunk = text[i:i+step]
        buf = BytesIO()
        gTTS(chunk, lang="en").write_to_fp(buf)
        buf.seek(0)
        combined.write(buf.read())
    combined.seek(0)
    return combined

def audio_with_autonext(audio_bytes: BytesIO, autoplay: bool, next_idx: int | None):
    """Audio element that autoplays and jumps to ?p=<next_idx> on 'ended' when autoplay is True."""
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto = "autoplay" if autoplay else ""
    js = f"""
    <script>
      (function(){{
        const AUTO = {str(autoplay).lower()};
        const NEXT = {('null' if next_idx is None else int(next_idx))};
        const audio = document.getElementById('tts-audio');
        if (!audio) return;
        if (AUTO) {{
          audio.addEventListener('ended', function() {{
            if (NEXT !== null) {{
              const u = new URL(window.location);
              u.searchParams.set('p', NEXT);
              window.location.href = u.toString();
            }}
          }});
        }}
      }})();
    </script>
    """
    html(
        f"""
        <audio id="tts-audio" controls {auto} style="width:100%;">
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          Your browser does not support the audio element.
        </audio>
        {js}
        """,
        height=90,
    )

def auto_download_audio(audio_bytes: BytesIO, filename: str, token: str, enabled: bool):
    """
    Auto-download the MP3 once per page when `enabled` (autoplay ON).
    Uses localStorage<token> to avoid repeat downloads during reruns.
    """
    if not enabled:
        return
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    key = f"mcqmp3-{token}"
    html(
        f"""
        <script>
          (function(){{
            const KEY = {json.dumps(key)};
            if (!window.localStorage.getItem(KEY)) {{
              const a = document.createElement('a');
              a.href = "data:audio/mp3;base64,{b64}";
              a.download = {json.dumps(filename)};
              document.body.appendChild(a);
              a.click();
              setTimeout(()=>{{ a.remove(); }}, 500);
              window.localStorage.setItem(KEY, "1");
            }}
          }})();
        </script>
        """,
        height=0,
    )

# ---------- UI ----------
st.title("📖 MCQs Reader — Auto TTS + Auto MP3 Download + Auto Next")
st.caption("Each page (max 10 MCQs) auto-plays, auto-downloads its MP3, and moves to the next unless paused.")

with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    loop_end = st.toggle("Loop to first page after the last", value=False)

    # playback controls
    if "auto" not in st.session_state:
        st.session_state.auto = False
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start / Resume"):
            st.session_state.auto = True
    with c2:
        if st.button("⏸️ Pause"):
            st.session_state.auto = False

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# Sync index from URL (?p=)
try:
    q = st.experimental_get_query_params()
    if "p" in q:
        p = int(q["p"][0])
        if p >= 0:
            st.session_state.page_idx = p
except Exception:
    pass

# ---------- Load → decode → extract ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching & parsing MCQs..."):
            data = fetch_bytes(url)
            if url.lower().endswith(".docx"):
                full = extract_docx_text(data)
            else:
                full = normalize_text(best_effort_bytes_to_text(data))
            mcqs = extract_mcqs_with_answers(full)
            pages = mcqs_to_pages(mcqs, MCQS_PER_PAGE)
            st.session_state.pages = pages
            st.session_state.page_idx = 0
            st.session_state.loaded_url = url
            if pages and pages[0].startswith("No MCQs"):
                st.warning("No MCQs with options detected. Ensure questions begin like 'Q...' or '1.' and options like 'A) ...'.")
    except Exception as e:
        st.error(f"Could not load/parse: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# clamp index and mirror to URL
st.session_state.page_idx = max(0, min(st.session_state.page_idx, len(st.session_state.pages) - 1))
st.experimental_set_query_params(p=st.session_state.page_idx)

# ---------- Navigation ----------
left, mid, right = st.columns([1, 3, 1])
with left:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.page_idx == 0):
        st.session_state.page_idx = max(0, st.session_state.page_idx - 1)
with mid:
    st.markdown(
        f"<div style='text-align:center;font-weight:600;'>Page {st.session_state.page_idx + 1} / {len(st.session_state.pages)}</div>",
        unsafe_allow_html=True,
    )
with right:
    if st.button("Next ➡️", use_container_width=True, disabled=st.session_state.page_idx >= len(st.session_state.pages) - 1):
        st.session_state.page_idx = min(len(st.session_state.pages) - 1, st.session_state.page_idx + 1)

# ---------- Current page ----------
page_text = st.session_state.pages[st.session_state.page_idx]
st.text_area("MCQs on this page (max 10)", page_text, height=520)

# ---------- Generate audio once per render ----------
audio_bytes = tts_mp3(page_text)

# ---------- Auto-download MP3 (one-time per page when autoplay ON) ----------
mp3_filename = f"mcqs_page_{st.session_state.page_idx + 1}.mp3"
download_token = f"{st.session_state.loaded_url}|{st.session_state.page_idx}|{len(page_text)}"
auto_download_audio(audio_bytes, mp3_filename, token=download_token, enabled=st.session_state.auto)

# ---------- TTS + Auto-next (when audio ends) ----------
last_idx = len(st.session_state.pages) - 1
next_idx = (st.session_state.page_idx + 1) if st.session_state.page_idx < last_idx else (0 if loop_end else None)
audio_with_autonext(audio_bytes, autoplay=st.session_state.auto, next_idx=next_idx)

# ---------- Manual downloads (optional) ----------
st.download_button("⬇️ Download this page (txt)", data=page_text.encode("utf-8"),
                   file_name=f"mcqs_page_{st.session_state.page_idx+1}.txt", mime="text/plain")
st.download_button("⬇️ Download this page (MP3)", data=audio_bytes.getvalue(),
                   file_name=mp3_filename, mime="audio/mpeg")

# ---------- Jump ----------
with st.expander("Jump to page"):
    total = max(1, len(st.session_state.pages))
    idx = st.number_input("Go to page #", min_value=1, max_value=total,
                          value=min(st.session_state.page_idx + 1, total), step=1)
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_set_query_params(p=st.session_state.page_idx)
        st.experimental_rerun()
