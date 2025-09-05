# lab.py — Read ALL pages & passages (no exclusions) + TTS
import re
from io import BytesIO
from base64 import b64encode

import requests
import streamlit as st
import streamlit.components.v1 as components
def render_audio_player(audio_bytes: BytesIO, rate: float = 1.5):
    """
    Render an HTML5 audio player with a specified playbackRate.
    """
    b64 = b64encode(audio_bytes.getvalue()).decode("utf-8")
    html = f"""
    <audio id="tts_player" controls style="width:100%;">
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    <script>
      const audio = document.getElementById('tts_player');
      audio.addEventListener('loadedmetadata', () => {{
          audio.playbackRate = {rate};
      }});
      audio.playbackRate = {rate};
    </script>
    """
    components.html(html, height=80, scrolling=False)

# ---------- UI ----------
st.title("📖 Lab Reader (All Content) → Text-to-Speech")
st.caption("Reads **everything** from the document, including all pages and passages. Nothing is excluded.")

# ---------- Session state ----------
if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = ""
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "playback_rate" not in st.session_state:
    st.session_state.playback_rate = 1.5  # default 1.5× speed

# ---------- Sidebar: inputs + navigation + speed ----------
with st.sidebar:
    url = st.text_input("GitHub RAW .docx URL", value=DEFAULT_URL)
    target_chars = st.slider("Target characters per page (fallback when no page breaks)", 800, 3200, DEFAULT_PAGE_CHARS, 100)
    st.markdown(
        "- Use a **raw** URL (https://raw.githubusercontent.com/...).\n"
        "- If your file isn’t .docx, convert it to .docx for best results."
    )

# ---------- Load & paginate ----------
if url != st.session_state.loaded_url:
    try:
        with st.spinner("Fetching and preparing document..."):
            data = fetch_bytes(url)
            if url.lower().endswith(".docx"):
                full_text = extract_docx_text_with_breaks(data)
            else:
                full_text = normalize_text(best_effort_bytes_to_text(data))

            pages = paginate_full_text(full_text, target_chars)
            st.session_state.pages = pages
            st.session_state.page_idx = 0
            st.session_state.loaded_url = url
            st.session_state.last_audio = None  # clear previous audio on new load
    except Exception as e:
        st.error(f"Could not load the document: {e}")
        st.stop()

if not st.session_state.pages:
    st.warning("No content found.")
    st.stop()

# ---------- Sidebar: page navigation & playback speed ----------
with st.sidebar:
    total_pages = len(st.session_state.pages)
    # Page dropdown (1-based)
    current_display_idx = st.session_state.page_idx + 1
    page_choice = st.selectbox(
        "Jump to page",
        options=list(range(1, total_pages + 1)),
        index=current_display_idx - 1,
        help="Navigate directly to a page.",
        key="page_select_sidebar"
    )
    if page_choice != current_display_idx:
        st.session_state.page_idx = page_choice - 1

    # Playback speed selector (default 1.5×)
    speed_options = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    default_index = speed_options.index(1.5)
    playback_rate = st.selectbox(
        "Playback speed",
        options=speed_options,
        index=default_index if st.session_state.playback_rate not in speed_options
              else speed_options.index(st.session_state.playback_rate),
        format_func=lambda x: f"{x}×",
        help="Controls audio playback speed."
    )
    st.session_state.playback_rate = playback_rate

# ---------- Top navigation buttons (optional) ----------
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
st.text_area("Page Content (Full, unfiltered)", page_text, height=520)

col_play, col_dl = st.columns([2, 1])
with col_play:
    if st.button("🔊 Read this page aloud"):
        try:
            with st.spinner("Generating audio..."):
                audio_buf = tts_mp3(page_text)
                st.session_state.last_audio = audio_buf
        except Exception as e:
            st.error(f"TTS failed: {e}")

    # If we have audio, render player at selected speed (updates when speed changes)
    if st.session_state.last_audio is not None:
        render_audio_player(st.session_state.last_audio, rate=st.session_state.playback_rate)

with col_dl:
    st.download_button(
        "⬇️ Download this page (txt)",
        data=page_text.encode("utf-8"),
        file_name=f"page_{st.session_state.page_idx+1}.txt",
        mime="text/plain",
    )

# ---------- Jump (numeric input, optional) ----------
with st.expander("Jump to page (numeric)"):
    total = max(1, len(st.session_state.pages))
    idx = st.number_input("Go to page #", min_value=1, max_value=total,
                          value=min(st.session_state.page_idx + 1, total), step=1)
    if st.button("Go"):
        st.session_state.page_idx = int(idx) - 1
        st.experimental_rerun()
