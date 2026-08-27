# med_interview.py — High-yield Medical School Interview + MMI Prep
# Reads med.pdf from GitHub, keeps the interview-prep content that matters,
# adds category filtering, practice/reveal mode, pagination, and TTS.

import re
import base64
from io import BytesIO

import requests
import streamlit as st
from gtts import gTTS
from pypdf import PdfReader


# ---------- App config ----------
st.set_page_config(
    page_title="🎤 Medical School Interview Prep",
    page_icon="🎤",
    layout="wide",
)

# Save the attached courseware in the same GitHub repo as: med.pdf
DEFAULT_URL = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/med.pdf"


# ---------- Download / PDF extraction ----------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=False)
def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return clean_pdf_text("\n".join(pages))


def clean_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    text = text.replace("￾", "-")

    # Remove repeated courseware footer/header noise.
    text = re.sub(
        r"(?im)^\s*Medical School Interview\s*&\s*MMI Courseware\s*\|\s*$",
        "",
        text,
    )

    # Normalize bullets and spacing without destroying useful line breaks.
    text = text.replace("", "•")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------- Text helpers ----------
def slice_between(text: str, start: str, end: str | None = None) -> str:
    start_pos = text.find(start)
    if start_pos == -1:
        return ""

    start_pos += len(start)

    if end:
        end_pos = text.find(end, start_pos)
        if end_pos == -1:
            end_pos = len(text)
    else:
        end_pos = len(text)

    return text[start_pos:end_pos].strip()


def tidy_block(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    out = []

    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue

        # Skip reference/source-only lines and repeated artifact labels.
        if line.startswith(("Best-Answer Benchmarking Sources", "Research Sources & Official Interview Guidance")):
            break

        out.append(line)

    result = "\n".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def normalize_for_display(text: str) -> str:
    # Make labels easier to scan.
    replacements = {
        "MODEL ANSWER |": "**MODEL ANSWER**  \n",
        "MEMORY HOOK |": "**MEMORY HOOK:** ",
        "FRAMEWORK |": "**FRAMEWORK:** ",
        "What they are assessing:": "**What they are assessing:**",
        "What they are testing:": "**What they are testing:**",
        "Scenario:": "**Scenario:**",
        "Best structure:": "**Best structure:**",
        "Coach note:": "**Coach note:**",
        "Adapt it:": "**Adapt it:**",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Explicit PREP labels should remain highly visible.
    text = re.sub(r"(?m)^Point:\s*", "**Point:** ", text)
    text = re.sub(r"(?m)^Reason:\s*", "**Reason:** ", text)
    text = re.sub(r"(?m)^Example:\s*", "**Example:** ", text)
    text = re.sub(r"(?m)^Point Summarized:\s*", "**Point Summarized:** ", text)

    return text.strip()


# ---------- Parsers ----------
def split_numbered_questions(region: str, category: str):
    """
    Parses blocks such as:
      1. Tell me about yourself.
      ...
      30. What do you do outside work and school?
    """
    if not region:
        return []

    pattern = re.compile(r"(?m)^(?P<num>\d{1,2})\.\s+(?P<title>[^\n]+)")
    matches = list(pattern.finditer(region))
    items = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(region)
        raw = tidy_block(region[start:end])

        if not raw:
            continue

        # Avoid accidentally capturing numbered instructional lists.
        if "FRAMEWORK" not in raw and "What they are" not in raw:
            continue

        title = f"{match.group('num')}. {match.group('title').strip()}"
        items.append(
            {
                "category": category,
                "title": title,
                "text": raw,
            }
        )

    return items


def split_mmi_questions(region: str):
    if not region:
        return []

    pattern = re.compile(r"(?m)^MMI\s+(?P<num>\d+)\.\s+(?P<title>[^\n]+)")
    matches = list(pattern.finditer(region))
    items = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(region)
        raw = tidy_block(region[start:end])

        if not raw:
            continue

        items.append(
            {
                "category": "MMI / Ethics",
                "title": f"MMI {match.group('num')}. {match.group('title').strip()}",
                "text": raw,
            }
        )

    return items


def make_section_item(category: str, title: str, text: str):
    text = tidy_block(text)
    if not text:
        return None
    return {"category": category, "title": title, "text": text}


def extract_high_yield_items(full_text: str):
    items = []

    # 1) Frameworks — keep the compact framework section, not the resource table.
    frameworks = slice_between(
        full_text,
        "High-Yield Answer Frameworks",
        "FRAMEWORK EDITION:",
    )
    item = make_section_item(
        "Frameworks",
        "High-Yield Answer Frameworks",
        frameworks,
    )
    if item:
        items.append(item)

    # 2) Personalized story bank.
    story_bank = slice_between(
        full_text,
        "Your Interview Story Bank",
        "Common Traditional Interview Questions",
    )
    item = make_section_item(
        "Frameworks",
        "Your Interview Story Bank",
        story_bank,
    )
    if item:
        items.append(item)

    # 3) Traditional question bank.
    traditional = slice_between(
        full_text,
        "1. Tell me about yourself.",
        "Questions You Can Ask Interviewers",
    )
    items.extend(split_numbered_questions(traditional, "Traditional"))

    # 4) Strong questions for interviewers.
    ask_interviewers = slice_between(
        full_text,
        "Questions You Can Ask Interviewers",
        "MMI Model Responses",
    )
    item = make_section_item(
        "Interview Day",
        "Questions You Can Ask Interviewers",
        ask_interviewers,
    )
    if item:
        items.append(item)

    # 5) MMI / ethics model responses.
    mmi = slice_between(
        full_text,
        "MMI Model Responses",
        "Handling Inappropriate Questions",
    )
    items.extend(split_mmi_questions(mmi))

    # 6) Handling inappropriate questions + scoring rubric.
    inappropriate = slice_between(
        full_text,
        "Handling Inappropriate Questions",
        "Research Sources & Official Interview Guidance",
    )
    item = make_section_item(
        "Interview Day",
        "Handling Inappropriate Questions + Self-Scoring Rubric",
        inappropriate,
    )
    if item:
        items.append(item)

    # 7) UWSOM school-specific module.
    uwsom = slice_between(
        full_text,
        "UWSOM Interview Module",
        "UWSOM Rapid-Fire Practice Prompts",
    )
    items.extend(split_numbered_questions(uwsom, "UWSOM"))

    uwsom_rapid = slice_between(
        full_text,
        "UWSOM Rapid-Fire Practice Prompts",
        "PNWU-COM Interview Module",
    )
    item = make_section_item(
        "UWSOM",
        "UWSOM Rapid-Fire Practice Prompts",
        uwsom_rapid,
    )
    if item:
        items.append(item)

    # 8) PNWU school-specific module.
    pnwu = slice_between(
        full_text,
        "PNWU-COM Interview Module",
        "PNWU Group Interview Strategy",
    )
    items.extend(split_numbered_questions(pnwu, "PNWU-COM"))

    pnwu_strategy = slice_between(
        full_text,
        "PNWU Group Interview Strategy",
        "Questions to Ask UWSOM and PNWU",
    )
    item = make_section_item(
        "PNWU-COM",
        "PNWU Group Interview Strategy + Rapid-Fire Prompts",
        pnwu_strategy,
    )
    if item:
        items.append(item)

    # 9) School-specific questions to ask.
    school_questions = slice_between(
        full_text,
        "Questions to Ask UWSOM and PNWU",
        "How to Convert a Model Answer into Your Answer",
    )
    item = make_section_item(
        "Interview Day",
        "Questions to Ask UWSOM and PNWU",
        school_questions,
    )
    if item:
        items.append(item)

    # 10) Final compact framework cheat sheet.
    cheat_sheet = slice_between(
        full_text,
        "One-Page Framework Cheat Sheet",
        None,
    )
    # Stop before any accidental source footer if present.
    cheat_sheet = re.split(
        r"(?m)^Best-Answer Benchmarking Sources|^Important:",
        cheat_sheet,
        maxsplit=1,
    )[0].strip()

    item = make_section_item(
        "Frameworks",
        "One-Page Framework Cheat Sheet",
        cheat_sheet,
    )
    if item:
        items.append(item)

    return items


# ---------- TTS ----------
def plain_text_for_tts(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("->", " then ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tts_mp3(text: str) -> BytesIO:
    # gTTS is more reliable with moderate chunks.
    step = 4200
    combined = BytesIO()

    for i in range(0, len(text), step):
        chunk = text[i:i + step]
        buf = BytesIO()
        gTTS(chunk, lang="en").write_to_fp(buf)
        buf.seek(0)
        combined.write(buf.read())

    combined.seek(0)
    return combined


def render_speedy_audio(audio_bytes: BytesIO, rate: float = 1.5, autoplay: bool = True):
    audio_bytes.seek(0)
    b64 = base64.b64encode(audio_bytes.read()).decode("ascii")
    auto = "autoplay" if autoplay else ""
    elem_id = "tts_player"

    st.components.v1.html(
        f"""
        <div>
          <audio id="{elem_id}" controls {auto} style="width:100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
          </audio>
          <script>
            const p = document.getElementById("{elem_id}");
            if (p) {{
              p.playbackRate = {rate};
              const tryPlay = () => p.play().catch(() => {{}});
              tryPlay();
            }}
          </script>
        </div>
        """,
        height=90,
    )


# ---------- Session state ----------
defaults = {
    "loaded_url": "",
    "items": [],
    "page_idx": 0,
    "playback_rate": 1.5,
    "revealed": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Header ----------
st.title("🎤 Medical School Interview + MMI Prep")
st.caption(
    "High-yield review from **med.pdf**: frameworks, personalized answers, "
    "MMIs, UWSOM, PNWU-COM, memory hooks, and interview-day practice."
)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Study controls")

    url = st.text_input(
        "GitHub RAW PDF URL",
        value=DEFAULT_URL,
        help="Save the PDF as med.pdf in the repo, then use the raw GitHub URL.",
    )

    selected_categories = st.multiselect(
        "Content",
        [
            "Frameworks",
            "Traditional",
            "MMI / Ethics",
            "UWSOM",
            "PNWU-COM",
            "Interview Day",
        ],
        default=[
            "Traditional",
            "MMI / Ethics",
            "UWSOM",
            "PNWU-COM",
        ],
    )

    practice_mode = st.toggle(
        "Practice mode — hide answer first",
        value=True,
    )

    show_assessment = st.toggle(
        "Show what they are assessing",
        value=True,
    )

    st.markdown(
        "**Included:** interview questions, frameworks, model answers, "
        "memory hooks, coach/adaptation notes, MMIs, and school-specific prep."
    )
    st.markdown(
        "**Excluded:** course resource tables, URLs, citations/source lists, "
        "and other low-yield reference material."
    )


# ---------- Load PDF ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching and extracting med.pdf..."):
            data = fetch_bytes(url)
            full_text = extract_pdf_text(data)
            items = extract_high_yield_items(full_text)

        if not items:
            st.error(
                "The PDF loaded, but no interview sections were detected. "
                "Confirm that med.pdf is the interview courseware PDF."
            )
            st.stop()

        st.session_state.items = items
        st.session_state.page_idx = 0
        st.session_state.loaded_url = url
        st.session_state.revealed = False

    except Exception as e:
        st.error(f"Could not load/parse the PDF: {e}")
        st.stop()


# ---------- Filter ----------
filtered_items = [
    item
    for item in st.session_state.items
    if item["category"] in selected_categories
]

if not filtered_items:
    st.warning("Choose at least one content category in the sidebar.")
    st.stop()

# Keep page index valid after filters change.
st.session_state.page_idx = min(
    st.session_state.page_idx,
    len(filtered_items) - 1,
)

current = filtered_items[st.session_state.page_idx]


# ---------- Sidebar item selector ----------
with st.sidebar:
    labels = [
        f"{i + 1}. [{item['category']}] {item['title']}"
        for i, item in enumerate(filtered_items)
    ]

    selected_label = st.selectbox(
        "Jump to",
        labels,
        index=st.session_state.page_idx,
    )

    new_idx = labels.index(selected_label)
    if new_idx != st.session_state.page_idx:
        st.session_state.page_idx = new_idx
        st.session_state.revealed = False
        st.rerun()


# ---------- Speed controls ----------
st.subheader("Playback speed")
c1, c2, c3, c4, c5, c6 = st.columns(6)

if c1.button("0.75×"):
    st.session_state.playback_rate = 0.75
if c2.button("1.0×"):
    st.session_state.playback_rate = 1.0
if c3.button("1.5×"):
    st.session_state.playback_rate = 1.5
if c4.button("2.0×"):
    st.session_state.playback_rate = 2.0
if c5.button("2.5×"):
    st.session_state.playback_rate = 2.5
if c6.button("3.0×"):
    st.session_state.playback_rate = 3.0

st.caption(f"Current speed: **{st.session_state.playback_rate}×**")


# ---------- Navigation ----------
left, mid, right = st.columns([1, 3, 1])

with left:
    if st.button(
        "⬅️ Previous",
        use_container_width=True,
        disabled=st.session_state.page_idx == 0,
    ):
        st.session_state.page_idx -= 1
        st.session_state.revealed = False
        st.rerun()

with mid:
    st.markdown(
        f"<div style='text-align:center;font-weight:700;'>"
        f"{st.session_state.page_idx + 1} / {len(filtered_items)}"
        f"</div>",
        unsafe_allow_html=True,
    )

with right:
    if st.button(
        "Next ➡️",
        use_container_width=True,
        disabled=st.session_state.page_idx >= len(filtered_items) - 1,
    ):
        st.session_state.page_idx += 1
        st.session_state.revealed = False
        st.rerun()


st.markdown("---")


# ---------- Current study item ----------
current = filtered_items[st.session_state.page_idx]
display_text = normalize_for_display(current["text"])

st.markdown(f"### {current['title']}")
st.caption(current["category"])

# Split the first line (question/title) from the coaching content.
body_lines = display_text.splitlines()
if body_lines and re.match(r"^(?:\d+\.|MMI\s+\d+\.)", body_lines[0].strip()):
    body_lines = body_lines[1:]

body = "\n".join(body_lines).strip()

if not show_assessment:
    body = re.sub(
        r"\*\*What they are (?:assessing|testing):\*\*.*?(?=\n\*\*FRAMEWORK:|\Z)",
        "",
        body,
        flags=re.S,
    ).strip()

if practice_mode:
    st.info("Answer this aloud before revealing the model/coaching content.")

    if st.button(
        "👁️ Reveal answer + coaching",
        use_container_width=True,
    ):
        st.session_state.revealed = True

    if st.session_state.revealed:
        st.markdown(body)
else:
    st.markdown(body)


# ---------- TTS ----------
st.markdown("---")

tts_text = current["title"]
if not practice_mode or st.session_state.revealed:
    tts_text += "\n" + body

if st.button(
    "🔊 Read current item aloud",
    use_container_width=True,
):
    try:
        with st.spinner("Generating audio..."):
            audio_buf = tts_mp3(plain_text_for_tts(tts_text))

        render_speedy_audio(
            audio_buf,
            rate=st.session_state.playback_rate,
            autoplay=True,
        )

    except Exception as e:
        st.error(f"TTS failed: {e}")


# ---------- Quick memory-hook view ----------
memory = re.search(
    r"\*\*MEMORY HOOK:\*\*\s*(.*?)(?=\n\*\*Coach note:|\n\*\*Adapt it:|\Z)",
    body,
    flags=re.S,
)

if memory:
    with st.expander("🧠 Memory hook only"):
        st.write(memory.group(1).strip())


# ---------- Download current item ----------
download_text = f"{current['title']}\n\n{plain_text_for_tts(body)}"

st.download_button(
    "⬇️ Download current item",
    data=download_text.encode("utf-8"),
    file_name=f"interview_item_{st.session_state.page_idx + 1}.txt",
    mime="text/plain",
    use_container_width=True,
)
