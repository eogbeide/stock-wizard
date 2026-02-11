import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json

# ---------------------------
# Load data from Excel on GitHub
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/Product.xlsx"
    try:
        df = pd.read_excel(url)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={"Interviewer": "QuestionType", "Interviewee": "Interview"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

product_data = load_data()
if product_data.empty:
    st.warning("No data available.")
    st.stop()

# Basic column safety (optional but helps avoid silent issues)
required_cols = {"Category", "Subcategory", "QuestionType", "Interview"}
missing = required_cols - set(product_data.columns)
if missing:
    st.error(f"Missing required columns in Product.xlsx: {', '.join(sorted(missing))}")
    st.stop()

st.sidebar.title("Interview Navigation")

# 1) Category selector
categories = product_data["Category"].dropna().unique()
selected_category = st.sidebar.selectbox("Select Category", categories)

# 2) Subcategory selector
sub_df = product_data[product_data["Category"] == selected_category]
subcategories = sub_df["Subcategory"].dropna().unique()
selected_subcategory = st.sidebar.selectbox("Select Subcategory", subcategories)

# 3) QuestionType selector
qt_df = sub_df[sub_df["Subcategory"] == selected_subcategory]
question_types = qt_df["QuestionType"].dropna().unique()
selected_qtype = st.sidebar.selectbox("Select QuestionType", question_types)

# Apply all three filters
filtered_data = product_data[
    (product_data["Category"] == selected_category)
    & (product_data["Subcategory"] == selected_subcategory)
    & (product_data["QuestionType"] == selected_qtype)
]

# Clamp index
max_idx = len(filtered_data) - 1
if max_idx < 0:
    st.warning("No entries for that combination.")
    st.stop()

if "question_index" not in st.session_state:
    st.session_state.question_index = 0
st.session_state.question_index = min(st.session_state.question_index, max_idx)

# ---------------------------
# Conversational narration builder
# ---------------------------
def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())  # normalize whitespace
    return s

def build_conversational_text(question_type: str, interview: str) -> str:
    qt = _clean_text(question_type)
    iv = _clean_text(interview)

    parts = []
    parts.append("Alright — let’s go through this together.")
    if qt:
        parts.append(f"First up, the question type is: {qt}.")
    if iv:
        parts.append("Here’s the interview response:")
        parts.append(iv)
    parts.append("When you’re ready, you can hit Next to continue.")
    return " ".join(parts).strip()

# ---------------------------
# TTS controls component (Web Speech API)
# - Google UK English Male by default (if available)
# - Play / Pause / Resume / Stop
# - ±10s and seek slider (estimated duration)
# ---------------------------
def tts_controls(text: str):
    safe_text = json.dumps(text)  # safely embed into JS string

    html = f"""
    <div class="tts-wrap">
      <div class="title">Controls</div>
      <div class="section">Listen</div>

      <div class="row">
        <button id="btnPlay" class="btn primary">▶️ Play</button>
        <button id="btnPause" class="btn" disabled>⏸ Pause</button>
        <button id="btnResume" class="btn" disabled>🔊 Resume</button>
      </div>

      <div class="row">
        <button id="btnStop" class="btn" disabled>■ Stop</button>
      </div>

      <div id="voiceStatus" class="status">Loading voices…</div>

      <div class="row">
        <button id="btnBack" class="btn small">⏪ 10s</button>
        <button id="btnFwd" class="btn small">10s ⏩</button>
      </div>

      <input id="progress" type="range" min="0" max="1000" value="0" />
      <div class="timeRow">
        <span id="tCur">0:00</span>
        <span id="tTot">0:00</span>
      </div>

      <div class="voiceRow">
        <div class="label">Voice</div>
        <select id="voiceSelect"></select>
      </div>
    </div>

    <style>
      :root {{
        --bg1: #0b0f17;
        --bg2: #0a0d14;
        --text: rgba(255,255,255,0.95);
        --muted: rgba(255,255,255,0.55);
        --btn: rgba(255,255,255,0.08);
        --btnBorder: rgba(255,255,255,0.28);
        --btnDisabled: rgba(255,255,255,0.04);
      }}
      html, body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      }}
      .tts-wrap {{
        border-radius: 18px;
        padding: 22px 22px 18px 22px;
        background: radial-gradient(1200px 600px at 30% 10%, #131b2b 0%, var(--bg1) 35%, var(--bg2) 100%);
        color: var(--text);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      }}
      .title {{
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 12px 0;
      }}
      .section {{
        font-size: 24px;
        font-weight: 750;
        margin: 6px 0 14px 0;
      }}
      .row {{
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        align-items: center;
        margin: 12px 0;
      }}
      .btn {{
        appearance: none;
        border: 1px solid var(--btnBorder);
        background: var(--btn);
        color: var(--text);
        padding: 14px 18px;
        border-radius: 16px;
        font-size: 20px;
        font-weight: 650;
        cursor: pointer;
        transition: transform 0.06s ease, background 0.2s ease, border-color 0.2s ease, opacity 0.2s ease;
        user-select: none;
      }}
      .btn:hover:enabled {{
        background: rgba(255,255,255,0.12);
        border-color: rgba(255,255,255,0.38);
      }}
      .btn:active:enabled {{
        transform: translateY(1px);
      }}
      .btn.primary {{
        background: rgba(255,255,255,0.16);
        border-color: rgba(255,255,255,0.45);
      }}
      .btn.small {{
        padding: 12px 16px;
        font-size: 18px;
        border-radius: 14px;
      }}
      .btn:disabled {{
        cursor: not-allowed;
        background: var(--btnDisabled);
        border-color: rgba(255,255,255,0.18);
        opacity: 0.55;
      }}
      .status {{
        margin-top: 6px;
        color: var(--muted);
        font-size: 18px;
      }}
      #progress {{
        width: 100%;
        margin: 16px 0 6px 0;
      }}
      .timeRow {{
        display: flex;
        justify-content: space-between;
        color: var(--muted);
        font-size: 18px;
        margin-bottom: 10px;
      }}
      .label {{
        color: var(--muted);
        font-size: 18px;
        margin-bottom: 8px;
      }}
      select {{
        width: 100%;
        padding: 12px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.06);
        color: var(--text);
        font-size: 18px;
        outline: none;
      }}
      option {{
        background: #0c1120;
        color: var(--text);
      }}
    </style>

    <script>
      const TEXT = {safe_text};

      const btnPlay = document.getElementById("btnPlay");
      const btnPause = document.getElementById("btnPause");
      const btnResume = document.getElementById("btnResume");
      const btnStop = document.getElementById("btnStop");
      const btnBack = document.getElementById("btnBack");
      const btnFwd = document.getElementById("btnFwd");
      const progress = document.getElementById("progress");
      const tCur = document.getElementById("tCur");
      const tTot = document.getElementById("tTot");
      const voiceSelect = document.getElementById("voiceSelect");
      const voiceStatus = document.getElementById("voiceStatus");

      let voices = [];
      let selectedVoice = null;

      let utter = null;
      let speaking = false;
      let paused = false;

      let baseIndex = 0;
      let baseSecondsOffset = 0;
      let totalSeconds = 0;

      let startPerf = 0;
      let elapsedBefore = 0;
      let tickTimer = null;

      function wordsCount(s) {{
        const t = (s || "").trim();
        if (!t) return 0;
        return t.split(/\\s+/).length;
      }}

      function estimateDurationSeconds(text) {{
        const w = wordsCount(text);
        const wpm = 170;
        return Math.max(1, (w / wpm) * 60);
      }}

      function fmtTime(sec) {{
        sec = Math.max(0, Math.floor(sec));
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return `${{m}}:${{String(s).padStart(2, "0")}}`;
      }}

      function setButtons(state) {{
        if (state === "idle") {{
          btnPlay.disabled = false;
          btnPause.disabled = true;
          btnResume.disabled = true;
          btnStop.disabled = true;
        }} else if (state === "playing") {{
          btnPlay.disabled = true;
          btnPause.disabled = false;
          btnResume.disabled = true;
          btnStop.disabled = false;
        }} else if (state === "paused") {{
          btnPlay.disabled = true;
          btnPause.disabled = true;
          btnResume.disabled = false;
          btnStop.disabled = false;
        }}
      }}

      function setProgressSeconds(sec) {{
        sec = Math.max(0, Math.min(totalSeconds, sec));
        const v = totalSeconds > 0 ? Math.round((sec / totalSeconds) * 1000) : 0;
        progress.value = String(v);
        tCur.textContent = fmtTime(sec);
      }}

      function getProgressSeconds() {{
        const v = Number(progress.value || 0);
        return totalSeconds > 0 ? (v / 1000) * totalSeconds : 0;
      }}

      function timeToCharIndex(targetSec) {{
        const frac = totalSeconds > 0 ? Math.max(0, Math.min(1, targetSec / totalSeconds)) : 0;
        let idx = Math.floor(TEXT.length * frac);

        if (idx > 0 && idx < TEXT.length - 1) {{
          let left = idx;
          while (left > 0 && TEXT[left] !== " " && TEXT[left] !== "\\n" && TEXT[left] !== "\\t") left--;
          let right = idx;
          while (right < TEXT.length && TEXT[right] !== " " && TEXT[right] !== "\\n" && TEXT[right] !== "\\t") right++;
          idx = (idx - left <= right - idx) ? left : right;
        }}
        return Math.max(0, Math.min(TEXT.length, idx));
      }}

      function stopTicker() {{
        if (tickTimer) {{
          clearInterval(tickTimer);
          tickTimer = null;
        }}
      }}

      function startTicker() {{
        stopTicker();
        tickTimer = setInterval(() => {{
          if (!speaking) return;

          if (window.speechSynthesis && !window.speechSynthesis.paused) {{
            const now = performance.now();
            const run = elapsedBefore + (startPerf ? (now - startPerf) / 1000 : 0);
            const cur = Math.min(totalSeconds, baseSecondsOffset + run);
            setProgressSeconds(cur);
          }}
        }}, 250);
      }}

      function choosePreferredVoice(vs) {{
        const exact = vs.find(v => (v.name || "").toLowerCase() === "google uk english male");
        if (exact) return exact;

        const googleUk = vs.find(v =>
          (v.name || "").toLowerCase().includes("google") &&
          (v.lang || "").toLowerCase().startsWith("en-gb")
        );
        if (googleUk) return googleUk;

        const googleEn = vs.find(v =>
          (v.name || "").toLowerCase().includes("google") &&
          (v.lang || "").toLowerCase().startsWith("en")
        );
        if (googleEn) return googleEn;

        const anyEn = vs.find(v => (v.lang || "").toLowerCase().startsWith("en"));
        return anyEn || (vs.length ? vs[0] : null);
      }}

      function populateVoiceSelect() {{
        voiceSelect.innerHTML = "";
        const enVoices = voices.filter(v => (v.lang || "").toLowerCase().startsWith("en"));
        const list = (enVoices.length ? enVoices : voices);

        list.forEach((v, i) => {{
          const opt = document.createElement("option");
          opt.value = String(i);
          opt.textContent = `${{v.name}} — ${{v.lang}}`;
          voiceSelect.appendChild(opt);
        }});

        selectedVoice = choosePreferredVoice(list);

        const idx = list.findIndex(v => v === selectedVoice);
        if (idx >= 0) voiceSelect.value = String(idx);

        const defaultMsg = selectedVoice && (selectedVoice.name || "").toLowerCase() === "google uk english male"
          ? "Voice loaded (default: Google UK English Male)."
          : "Voice loaded (default: Google UK Male if available).";
        voiceStatus.textContent = defaultMsg;
      }}

      function loadVoicesWithRetry(tries = 0) {{
        if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {{
          voiceStatus.textContent = "Speech synthesis isn’t available in this browser.";
          return;
        }}

        voices = window.speechSynthesis.getVoices() || [];
        if (voices.length) {{
          populateVoiceSelect();
          return;
        }}

        if (tries < 25) {{
          setTimeout(() => loadVoicesWithRetry(tries + 1), 150);
        }} else {{
          voiceStatus.textContent = "Couldn’t load voices. Try reloading the page.";
        }}
      }}

      function speakFromCharIndex(charIndex) {{
        if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) return;

        window.speechSynthesis.cancel();
        utter = null;

        baseIndex = Math.max(0, Math.min(TEXT.length, charIndex));
        baseSecondsOffset = TEXT.length > 0 ? (baseIndex / TEXT.length) * totalSeconds : 0;

        elapsedBefore = 0;
        startPerf = 0;

        const slice = TEXT.slice(baseIndex);
        if (!slice.trim()) {{
          setButtons("idle");
          setProgressSeconds(totalSeconds);
          speaking = false;
          paused = false;
          return;
        }}

        utter = new SpeechSynthesisUtterance(slice);
        if (selectedVoice) utter.voice = selectedVoice;
        utter.rate = 1.0;
        utter.pitch = 1.0;

        utter.onstart = () => {{
          speaking = true;
          paused = false;
          setButtons("playing");
          startPerf = performance.now();
          startTicker();
        }};

        utter.onend = () => {{
          speaking = false;
          paused = false;
          elapsedBefore = 0;
          startPerf = 0;
          setButtons("idle");
          stopTicker();
          setProgressSeconds(totalSeconds);
        }};

        utter.onerror = () => {{
          speaking = false;
          paused = false;
          elapsedBefore = 0;
          startPerf = 0;
          setButtons("idle");
          stopTicker();
        }};

        utter.onboundary = (e) => {{
          if (!e) return;
          if (typeof e.charIndex === "number") {{
            const globalIdx = baseIndex + e.charIndex;
            const frac = TEXT.length > 0 ? Math.max(0, Math.min(1, globalIdx / TEXT.length)) : 0;
            const cur = frac * totalSeconds;
            setProgressSeconds(cur);
          }}
        }};

        window.speechSynthesis.speak(utter);
      }}

      function playFromCurrentSlider() {{
        const sec = getProgressSeconds();
        const idx = timeToCharIndex(sec);
        speakFromCharIndex(idx);
      }}

      function stopAll(resetToStart = true) {{
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        speaking = false;
        paused = false;
        utter = null;
        elapsedBefore = 0;
        startPerf = 0;
        stopTicker();
        setButtons("idle");
        if (resetToStart) setProgressSeconds(0);
      }}

      function pauseSpeaking() {{
        if (!("speechSynthesis" in window)) return;
        if (!speaking) return;

        if (startPerf) {{
          elapsedBefore += (performance.now() - startPerf) / 1000;
          startPerf = 0;
        }}
        window.speechSynthesis.pause();
        paused = true;
        setButtons("paused");
      }}

      function resumeSpeaking() {{
        if (!("speechSynthesis" in window)) return;
        if (!speaking) return;

        window.speechSynthesis.resume();
        paused = false;
        setButtons("playing");
        startPerf = performance.now();
      }}

      function seekToSeconds(sec) {{
        sec = Math.max(0, Math.min(totalSeconds, sec));
        setProgressSeconds(sec);

        if (speaking || paused) {{
          const idx = timeToCharIndex(sec);
          speakFromCharIndex(idx);
        }}
      }}

      // Init
      totalSeconds = estimateDurationSeconds(TEXT);
      tTot.textContent = fmtTime(totalSeconds);
      tCur.textContent = "0:00";
      setButtons("idle");
      loadVoicesWithRetry();

      // Voice changes
      voiceSelect.addEventListener("change", () => {{
        const enVoices = voices.filter(v => (v.lang || "").toLowerCase().startsWith("en"));
        const list = (enVoices.length ? enVoices : voices);
        const i = Number(voiceSelect.value || 0);
        selectedVoice = list[i] || selectedVoice;

        if (speaking || paused) {{
          const sec = getProgressSeconds();
          const idx = timeToCharIndex(sec);
          speakFromCharIndex(idx);
        }}
      }});

      // Buttons
      btnPlay.addEventListener("click", () => playFromCurrentSlider());
      btnPause.addEventListener("click", () => pauseSpeaking());
      btnResume.addEventListener("click", () => resumeSpeaking());
      btnStop.addEventListener("click", () => stopAll(true));

      btnBack.addEventListener("click", () => {{
        const cur = getProgressSeconds();
        seekToSeconds(cur - 10);
      }});

      btnFwd.addEventListener("click", () => {{
        const cur = getProgressSeconds();
        seekToSeconds(cur + 10);
      }});

      // Slider interactions
      progress.addEventListener("input", () => {{
        const sec = getProgressSeconds();
        tCur.textContent = fmtTime(sec);
      }});

      progress.addEventListener("change", () => {{
        const sec = getProgressSeconds();
        seekToSeconds(sec);
      }});

      // Sync state
      setInterval(() => {{
        if (!("speechSynthesis" in window)) return;

        const synth = window.speechSynthesis;
        if (!synth.speaking && speaking) {{
          speaking = false;
          paused = false;
          setButtons("idle");
          stopTicker();
        }} else if (synth.speaking && synth.paused && speaking && !paused) {{
          paused = true;
          setButtons("paused");
        }} else if (synth.speaking && !synth.paused && speaking && paused) {{
          paused = false;
          setButtons("playing");
        }}

        btnStop.disabled = !(speaking || paused);
      }}, 300);
    </script>
    """

    # ✅ FIX: Streamlit's components.html() doesn't accept key= in many versions
    components.html(html, height=470, scrolling=False)

# ---------------------------
# UI render
# ---------------------------
def display_entry(idx: int):
    if idx < 0 or idx > max_idx:
        st.write("End of entries.")
        return False

    row = filtered_data.iloc[idx]

    st.subheader("Question Type")
    qt_text = str(row["QuestionType"]).strip().replace("\n\n", "  \n\n")
    st.markdown(qt_text)

    st.subheader("Interview")
    int_text = str(row["Interview"]).strip().replace("\n\n", "  \n\n")
    st.markdown(int_text)

    st.markdown("---")

    narration = build_conversational_text(row["QuestionType"], row["Interview"])
    tts_controls(narration)

    return True

# Show the current entry
if display_entry(st.session_state.question_index):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back"):
            if st.session_state.question_index > 0:
                st.session_state.question_index -= 1
    with col2:
        if st.button("Next ▶"):
            if st.session_state.question_index < max_idx:
                st.session_state.question_index += 1
            else:
                st.success("End of Interview. Thank you!")
