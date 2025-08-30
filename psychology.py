# psychology.py — MCQs + Options + Answers/Explanations only (Passages removed) → TTS + Video
import redf
import textwrap
import tempfile
from io import BytesIO

import requests
import streamlit as st
from gtts import gTTS

# NEW: video/image deps (no ImageMagick required)
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip

# .docx extraction
try:
    from docx import Document
    from docx.enum.text import WD_BREAK
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# ---------- App config ----------
st.set_page_config(page_title="📖 MCQs + Answers Reader (GitHub → TTS → Video)", page_icon="🎧", layout="wide")
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/psychbooks.docx"

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

PASSAGE_HEADER_PAT   = re.compile(r"^\s*passage(\s*[ivx]+|\s*\d+|\s*[a-z])?\b", re.IGNORECASE)
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
    """
    Remove passage blocks. If a line starts with 'Passage...' we skip lines
    until we hit a question line or a 'Questions' header.
    """
    lines = [l.rstrip() for l in full_text.split("\n")]
    filtered = []
    in_passage = False
    for raw in lines:
        line = raw.strip()

        if PASSAGE_HEADER_PAT.match(line):
            in_passage = True
            continue
        if in_passage:
            if not line:
                continue
            if QUESTIONS_HEADER_PAT.match(line) or looks_like_question(line):
                in_passage = False
            else:
                continue

        filtered.append(raw)
    return filtered

def extract_mcqs_with_answers(full_text: str):
    """
    Return only MCQs (stem + options) and Answer/Explanation lines. Passages removed.
    """
    lines = remove_passages(full_text)
    mcqs = []
    cur_stem, cur_opts = [], []
    cur_answer, cur_expl = None, None
    in_mcq = False
    options_started = False

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
        # reset
        cur_stem, cur_opts = [], []
        cur_answer, cur_expl = None, None
        in_mcq = False
        options_started = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_mcq and cur_stem and not options_started:
                cur_stem.append("")
            continue

        if looks_like_question(line):
            if in_mcq:
                flush()
            in_mcq = True
            options_started = False
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

# ---------- TTS ----------
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

# ---------- VIDEO (PIL + MoviePy, no ImageMagick) ----------
def _load_font(size: int = 40):
    # Try DejaVu (usually available); fall back to default PIL font
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def render_text_slide(text: str, size=(1280, 720), margin=64, font_size=40, line_spacing=1.4) -> Image.Image:
    """Render wrapped MCQ text to a single PNG frame."""
    W, H = size
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(bg)
    title_font = _load_font(font_size + 6)
    font = _load_font(font_size)

    # Wrap to fit width using rough char estimate
    chars_per_line = max(20, int((W - 2 * margin) / (font_size * 0.55)))
    wrapped = []
    for block in text.split("\n\n"):
        wrapped.extend(textwrap.wrap(block, width=chars_per_line, break_long_words=False, replace_whitespace=False) or [""])
        wrapped.append("")  # paragraph gap
    if wrapped and wrapped[-1] == "":
        wrapped.pop()

    # Header
    header = "MCQs + Options + Answers"
    y = margin // 2
    draw.text((margin, y), header, fill=(0, 0, 0), font=title_font)
    y += int((font_size + 10) * 1.8)

    # Body
    for line in wrapped:
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
        y += int(font_size * line_spacing)
        if y > H - margin:
            break  # stop drawing if overflow; text is still narrated

    # Footer
    footer = "Auto-narrated"
    fw, fh = draw.textbbox((0, 0), footer, font=font)[2:]
    draw.text((W - margin - fw, H - margin - fh), footer, fill=(80, 80, 80), font=font)

    return bg

def video_from_audio_and_text(audio_mp3_bytes: BytesIO, text: str, size=(1280, 720), fps=24) -> BytesIO:
    """
    Build an MP4: single white slide with wrapped text + the audio as soundtrack.
    No ImageMagick dependency.
    """
    # Save audio to a temp file so MoviePy can read duration
    audio_mp3_bytes.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_audio:
        f_audio.write(audio_mp3_bytes.read())
        audio_path = f_audio.name

    # Render slide image
    frame = render_text_slide(text, size=size)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_img:
        frame.save(f_img.name, format="PNG")
        img_path = f_img.name

    # Compose video
    audio_clip = AudioFileClip(audio_path)
    duration = max(2.0, audio_clip.duration)  # at least 2s
    img_clip = ImageClip(img_path).set_duration(duration)
    vid_clip = img_clip.set_audio(audio_clip)

    # Export MP4
    out_bytes = BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
        out_path = f_out.name

    vid_clip.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        bitrate="1800k",
        audio_bitrate="128k",
        preset="ultrafast",
        verbose=False,
        logger=None,
    )
    audio_clip.close()
    vid_clip.close()

    with open(out_path, "rb") as rf:
        out_bytes.write(rf.read())
    out_bytes.seek(0)
    return out_bytes

# ---------- Sidebar ----------
st.title("📖 MCQs + Answers Reader (Passages removed) → TTS + Video")
st.caption("Reads MCQs + options + answers/explanations (no passages), and can generate a narrated video for each page.")
with st.sidebar:
    url = st.text_input("GitHub RAW file URL", value=DEFAULT_URL)
    mcqs_per_page = st.slider("MCQs per page", 1, 10, 3, 1)
    make_video = st.toggle("Also make explainer video", value=True)
    st.markdown("- Use a **raw** URL (`https://raw.githubusercontent.com/...`). Best with **.docx** or **.txt**.")

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0

# ---------- Load → decode → extract ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching file..."):
            data = fetch_bytes(url)
        lower = url.lower()

        if lower.endswith(".docx"):
            full_text = extract_docx_text(data)
        elif lower.endswith(".txt") or lower.endswith(".md"):
            full_text = normalize_text(best_effort_bytes_to_text(data))
        elif lower.endswith(".doc"):
            st.warning("Legacy **.doc** detected. Convert to **.docx**/**.txt** for clean parsing.")
            use_fallback = st.toggle("Try fallback decode (may be messy)", value=False)
            if not use_fallback:
                st.stop()
            full_text = normalize_text(best_effort_bytes_to_text(data))
        else:
            full_text = normalize_text(best_effort_bytes_to_text(data))

        mcqs = extract_mcqs_with_answers(full_text)
        pages = mcqs_to_pages(mcqs, mcqs_per_page)

        st.session_state.pages = pages
        st.session_state.page_idx = 0
        st.session_state.loaded_url = url

        if pages and pages[0].startswith("No MCQs"):
            st.warning("No MCQs with options detected. Ensure questions start with 'Q...' or '1.' and options like 'A) text'.")
    except Exception as e:
        st.error(f"Could not load/parse: {e}")
        st.stop()

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
st.text_area("MCQs + Answers (no passage)", page_text, height=480)

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.button("🔊 Generate TTS for this page"):
        try:
            with st.spinner("Generating audio..."):
                audio_bytes = tts_mp3(page_text)
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download narration (MP3)",
                data=audio_bytes.getvalue(),
                file_name=f"mcqs_page_{st.session_state.page_idx+1}.mp3",
                mime="audio/mpeg",
            )
        except Exception as e:
            st.error(f"TTS failed: {e}")

with col2:
    if st.button("🎬 Generate Explainer Video (audio + slide)"):
        try:
            with st.spinner("Generating audio..."):
                audio_bytes = tts_mp3(page_text)
            with st.spinner("Rendering video..."):
                video_bytes = video_from_audio_and_text(audio_bytes, page_text, size=(1280, 720), fps=24)
            st.video(video_bytes)
            st.download_button(
                "⬇️ Download video (MP4)",
                data=video_bytes.getvalue(),
                file_name=f"mcqs_page_{st.session_state.page_idx+1}.mp4",
                mime="video/mp4",
            )
        except Exception as e:
            st.error(
                "Video render failed. Make sure your environment has `moviepy` and `imageio-ffmpeg` installed. "
                f"Details: {e}"
            )

with col3:
    st.download_button(
        "⬇️ Download this page (TXT)",
        data=page_text.encode("utf-8"),
        file_name=f"mcqs_page_{st.session_state.page_idx+1}.txt",
        mime="text/plain",
    )

# ---------- Jump ----------
with st.expander("Jump to page"):
    idx = st.number_input("Go to page #", min_value=1, max_value=len(st.session_state.pages), value=st.session_state.page_idx + 1, step=1)
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
