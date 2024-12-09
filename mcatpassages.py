import streamlit as st
import pandas as pd
from gtts import gTTS
import os

st.title("Simple Text to Speech Converter")

# Text area for user input
text_area = st.text_area("Copy and paste text here to convert to speech:")

language = st.selectbox("Select language:", ["en", "fr", "ru", "hi", "es"])

if st.button("Convert"):
    if text_area:  # Check if there's text to convert
        audio_stream = gTTS(text=text_area, lang=language)
        audio_stream.save("output.mp3")  # Save the audio file
        st.success("Speech is generated successfully!")
        st.audio("output.mp3")  # Play the audio file
    else:
        st.warning("Please enter some text.")
        
@st.cache_data
# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labs.csv"
    df = pd.read_csv(url, encoding='ISO-8859-1')  # Specify encoding here
    return df

# Main function to run the app
def main():
    # Custom CSS to set font to Comic Sans MS and font size to 10
    st.markdown(
        """
        <style>
        body {
            font-family: 'Comic Sans MS';
            font-size: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("MCAT Labs")

    # Load the data
    df = load_data()

    # Clean column names
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace

    # Sidebar for subject selection
    if 'Subject' in df.columns:
        subjects = df['Subject'].unique()
        selected_subject = st.sidebar.selectbox("Select Subject:", subjects)

        # Filter data based on selected subject
        subject_data = df[df['Subject'] == selected_subject]

        # Sidebar for topic selection
        topics = subject_data['Topic'].unique()
        selected_topic = st.sidebar.selectbox("Select Topic:", topics)

        # Filter data based on selected topic
        topic_data = subject_data[subject_data['Topic'] == selected_topic]

        if not topic_data.empty:
            # Display Description in a box
            st.subheader("Description")
            description = topic_data['Description'].values[0]
            st.info(description)  # Use st.info for a box

            # Clean description for TTS
            cleaned_description = description.replace('*', '').replace('#', '').strip()
            if st.button("Read Description Aloud"):
                audio_stream = gTTS(text=cleaned_description, lang='en')
                audio_stream.save("description.mp3")  # Save the audio file
                st.audio("description.mp3")  # Play the audio file
                os.remove("description.mp3")  # Remove the audio file after playing

            # Display Questions and Answers in an expander
            st.subheader("Questions and Answers")
            with st.expander("View Questions and Answers"):
                questions_answers = topic_data['Questions and Answers'].values[0]
                st.write(questions_answers)  # Display questions and answers

                # Clean questions and answers for TTS
                cleaned_qa = questions_answers.replace('*', '').replace('#', '').strip()
                if st.button("Read Questions and Answers Aloud"):
                    audio_stream = gTTS(text=cleaned_qa, lang='en')
                    audio_stream.save("qa.mp3")  # Save the audio file
                    st.audio("qa.mp3")  # Play the audio file
                    os.remove("qa.mp3")  # Remove the audio file after playing

        else:
            st.write("No data available for the selected topic.")
    else:
        st.error("The 'Subject' column is missing from the data.")

if __name__ == "__main__":
    main()
