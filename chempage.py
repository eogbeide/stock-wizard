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

st.title("Your Chem172 Prep Companion")

def main():
    st.title("Welcome!")
    
    st.write("""
    These resources were curated to **assist pre-medical students in understanding Chem concepts**. 
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
if st.button("Go to Chem172 Topics"):
    st.markdown("[Open Chem172 Topics](https://chem172notes.streamlit.app/)", unsafe_allow_html=True)

if st.button("Go to Chem172 Q&A"):
    st.markdown("[Open Chem172 Q&A](https://chem172questions.streamlit.app/)", unsafe_allow_html=True)
