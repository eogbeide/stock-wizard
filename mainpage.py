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

import streamlit as st

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
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Topics](https://mcattopics.streamlit.app/)", unsafe_allow_html=True)


if st.button("Go to MCAT Lab Review"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Lab Review](https://mcatlabs.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Flashcards"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Flash Cards](https://mcatflashcards.streamlit.app/)", unsafe_allow_html=True)
    
if st.button("Go to MCAT Prep Quiz"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Quiz Prep](https://mcatprep.streamlit.app/)", unsafe_allow_html=True)

if st.button("Learn About US Medical School Pre-reqs"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open Med Schools Prerequisites Page](https://medschool.streamlit.app/)", unsafe_allow_html=True)
