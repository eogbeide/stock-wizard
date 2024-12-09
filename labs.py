import streamlit as st
import pandas as pd
from gtts import gTTS

@st.cache_data
# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labs.csv"
    df = pd.read_csv(url, encoding='ISO-8859-1')  # Specify encoding here
    return df

# Main function to run the app
def main():
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
            # Display Description
            description = topic_data['Description'].values[0]
            st.write(description)  # Display description as plain text

            # Convert Description to speech
            if st.button("Read Description Aloud"):
                audio_stream = gTTS(text=description, lang='en')  # Default to English
                audio_stream.save("description.mp3")
                st.audio("description.mp3")  # Play the audio file

            # Display Questions and Answers
            questions_answers = topic_data['Questions and Answers'].values[0]
            st.write("Questions and Answers:")
            with st.expander("View Questions and Answers"):
                st.write(questions_answers)  # Display questions and answers

            # Convert Questions and Answers to speech
            if st.button("Read Questions and Answers Aloud"):
                audio_stream = gTTS(text=questions_answers, lang='en')  # Default to English
                audio_stream.save("qa.mp3")
                st.audio("qa.mp3")  # Play the audio file

        else:
            st.write("No data available for the selected topic.")
    else:
        st.error("The 'Subject' column is missing from the data.")

if __name__ == "__main__":
    main()
