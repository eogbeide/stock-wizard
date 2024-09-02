import streamlit as st

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

# Create buttons to navigate to the external apps
if st.button("Go to MCAT Topics"):
    st.markdown("<iframe src='https://mcattopics.streamlit.app/' width='100%' height='600'></iframe>", unsafe_allow_html=True)

if st.button("Go to MCAT Lab Review"):
    st.markdown("<iframe src='https://mcatlabs.streamlit.app/' width='100%' height='600'></iframe>", unsafe_allow_html=True)

if st.button("Go to MCAT Flashcards"):
    st.markdown("<iframe src='https://mcatflashcards.streamlit.app/' width='100%' height='600'></iframe>", unsafe_allow_html=True)

if st.button("Go to MCAT Prep Quiz"):
    st.markdown("<iframe src='https://mcatprep.streamlit.app/' width='100%' height='600'></iframe>", unsafe_allow_html=True)

if st.button("Learn About US Medical School Pre-reqs"):
    st.markdown("<iframe src='https://medschool.streamlit.app/' width='100%' height='600'></iframe>", unsafe_allow_html=True)
