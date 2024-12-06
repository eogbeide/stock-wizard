import streamlit as st
import datetime

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #87CEEB;  /* Sky blue */
        padding: 20px;
    }
    .button {
        background-color: #007bff;
        color: white;
        border: None;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Your MCAT Prep Companion")

def main():
    st.title("Welcome!")
    
    st.write("""
    These resources were curated to **assist pre-medical students in understanding MCAT concepts**. 
    These materials are provided **entirely free of charge as a service to the community**.
    """)

    st.write("""
    **Designed to enhance your learning experience**, these resources aim to support your understanding 
    **without replacing your current study methods and materials**.
    """)

    st.write("Happy studying!")

if __name__ == "__main__":
    main()

# Create a timestamp to force a refresh
today = datetime.datetime.now().date()
st.write(f"Last updated: {today}")

# Create buttons to navigate to the external apps
if st.button("Go to MCAT Topics"):
    st.markdown("[Open MCAT Topics](https://mcattopics.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Lab Review"):
    st.markdown("[Open MCAT Lab Review](https://mcatlabs.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Passage-Style Q&A"):
    st.markdown("[Open MCAT Passage-Stype Q&A](https://mcatpassages.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Scientific Reasoning Questions"):
    st.markdown("[Open MCAT Scientific Reasoning Questions](https://mcatreasoning.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Flashcards"):
    st.markdown("[Open MCAT Flash Cards](https://mcatflashcards.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Prep Quiz"):
    st.markdown("[Open MCAT Quiz Prep](https://mcatprep.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Companion ChatGPT"):
    st.markdown("[Open MCAT Companion](https://poe.com/MCATCompanion)", unsafe_allow_html=True)

if st.button("Learn About US Medical School Pre-reqs"):
    st.markdown("[Open Med Schools Prerequisites Page](https://medschool.streamlit.app/)", unsafe_allow_html=True)

# Instructions for refreshing all links daily
st.write("""
To refresh all links daily, ensure to run this app once a day to get the latest updates.
""")
