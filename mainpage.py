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

# Create buttons to navigate to the external apps
if st.button("Go to MCAT Topics"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Topics](https://mcattopics.streamlit.app/)", unsafe_allow_html=True)
    
if st.button("Go to MCAT Prep"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Prep](https://mcatprep.streamlit.app/)", unsafe_allow_html=True)

if st.button("Learn About US Medical School Pre-reqs"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open Med Schools Prerequisites Page](https://medschool.streamlit.app/)", unsafe_allow_html=True)


