import streamlit as st

st.title("MCAT Resources Designed Just For You")

# Create buttons to navigate to the external apps
if st.button("Go to MCAT Prep"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Prep](https://mcatprep.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to MCAT Topics"):
    #st.write("You need to enable JavaScript to run this app.")
    st.markdown("[Open MCAT Topics](https://mcattopics.streamlit.app/)", unsafe_allow_html=True)
