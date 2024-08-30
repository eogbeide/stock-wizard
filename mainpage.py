import streamlit as st

# Define the pages for the two apps
pg = st.navigation([
    st.Page("https://mcatprep.streamlit.app/", label="MCAT Prep"),
    st.Page("https://mcattopics.streamlit.app/", label="MCAT Topics")
])

# Run the navigation
pg.run()
