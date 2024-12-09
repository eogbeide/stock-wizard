import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile

@st.cache_data
# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labs.csv"
    df = pd.read_csv(url, encoding='ISO-8859-1')  # Specify encoding here
    return df

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    if text:
        audio_stream = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp_file:
            audio_stream.save(tmp_file.name)  # Save the audio file to a temporary file
            tmp_file.seek(0)  # Move to the beginning of the file
            return tmp_file.name  # Return the file name for playback
    return None

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
            description_text = topic_data['Description'].values[0]
            st.info(description_text)  # Use st.info for a box

            # Button to convert Description to speech
            if st.button("Read Description"):
                audio_file = text_to_speech(description_text)
                if audio_file:
                    st.audio(audio_file)  # Play the audio file

            # Display Questions and Answers in an expander
            st.subheader("Questions and Answers")
            questions_answers_text = topic_data['Questions and Answers'].values[0]
            with st.expander("View Questions and Answers"):
                st.write(questions_answers_text)  # Display questions and answers

            # Button to convert Questions and Answers to speech
            if st.button("Read Questions and Answers"):
                audio_file = text_to_speech(questions_answers_text)
                if audio_file:
                    st.audio(audio_file)  # Play the audio file

        else:
            st.write("No data available for the selected topic.")
    else:
        st.error("The 'Subject' column is missing from the data.")

if __name__ == "__main__":
    main()
