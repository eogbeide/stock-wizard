import streamlit as st
import pandas as pd
import requests
from gtts import gTTS
import tempfile
import os

# Helper to play any text via gTTS
def play_text(text: str):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
    os.remove(fp.name)

# Function to load data from GitHub
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/scenarios.xls'
    r = requests.get(url)
    if r.status_code == 200:
        with open('scenarios.xls', 'wb') as f:
            f.write(r.content)
        return pd.read_excel('scenarios.xls')
    st.error("Failed to load data from GitHub.")
    return None

data = load_data()
if data is not None:
    # Init session state
    if 'current_scenario_index' not in st.session_state:
        st.session_state.current_scenario_index = 0

    # Sidebar: Scenario selector
    st.sidebar.markdown(
        "<h3 style='color:#4CAF50;'>Select Scenario Number:</h3>", unsafe_allow_html=True
    )
    scenario_options = data['scenario#'].unique()
    selected = st.sidebar.selectbox("", scenario_options)
    st.session_state.current_scenario_index = int(selected) - 1

    # Welcome banner
    st.markdown(
        "<div style='padding:10px; border:1px solid #4CAF50; border-radius:5px; "
        "background-color:black; color:white;'>"
        "<h2 style='color:#4CAF50; margin:0;'>Welcome to the ABA ORAL EXAM PRACTICE</h2>"
        "<p style='margin:0;'>Select a scenario, topic, and question to explore.</p>"
        "</div>",
        unsafe_allow_html=True
    )

    # Scenario overview
    scenario_text = data.loc[data['scenario#'] == selected, 'scenario'].iloc[0]
    with st.expander("Scenario Overview", expanded=False):
        st.markdown(
            f"<div style='padding:10px; border:1px solid #4CAF50; border-radius:5px; "
            f"background-color:black; color:white;'>"
            f"<strong style='color:#4CAF50;'>Scenario {selected}:</strong> {scenario_text}"
            "</div>",
            unsafe_allow_html=True
        )
        if st.button("🔊 Read Scenario Aloud"):
            play_text(scenario_text)

    # Topic selector
    st.markdown("<strong style='color:#4CAF50;'>Select a Topic:</strong>", unsafe_allow_html=True)
    category = st.selectbox("", data['category'].unique())

    # Section selector
    filtered_sections = data[data['category'] == category]['section'].unique()
    st.markdown("<strong style='color:#4CAF50;'>Select a Topic Question:</strong>", unsafe_allow_html=True)
    section = st.selectbox("", filtered_sections)

    # Display questions
    st.markdown("<h4>Questions</h4><hr>", unsafe_allow_html=True)
    filtered = data[
        (data['scenario#'] == selected) &
        (data['category'] == category) &
        (data['section'] == section)
    ]

    for idx, row in filtered.iterrows():
        q_text = row['question']
        st.markdown(f"**Question {idx+1}:** {q_text}")
        if st.button(f"🔊 Read Question {idx+1} Aloud"):
            play_text(q_text)

        # Show solution on demand
        if st.button(f"Show Solution for Question {idx+1}"):
            sol = str(row['solution'])
            st.write(f"**Solution:** {sol}")
            if st.button(f"🔊 Read Solution {idx+1} Aloud"):
                play_text(sol)

            source = row.get('source', '')
            if pd.notna(source) and source:
                google_link = f"https://www.google.com/search?q={source.replace(' ', '+')}"
                st.markdown(f"**Source:** {source}  \n[Refer to source]({google_link})")
        st.markdown("<hr>", unsafe_allow_html=True)
