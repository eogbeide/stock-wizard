import streamlit as st
import pandas as pd
from urllib.error import URLError
from gtts import gTTS
import os

@st.cache_data
def read_questions_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        return df
    except UnicodeDecodeError:
        st.error("Error decoding the CSV file. Trying a different encoding.")
        return pd.DataFrame()  # Return an empty DataFrame on failure
    except URLError as e:
        st.error(f"Error fetching data: {e.reason}")
        return pd.DataFrame()  # Return an empty DataFrame on failure
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Main function to run the Streamlit app
def main():
    # URL to the CSV file on GitHub
    file_path = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcattopics.csv"

    # Read questions from the CSV file
    df = read_questions_from_csv(file_path)

    if df.empty:
        st.write("No data available.")
        return

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()  # Normalize case and strip spaces

    # Rename columns to match expected names
    df.rename(columns={
        's/n': 'serial_number',
        'subject': 'subject',
        'topic': 'topic',
        'question': 'question',
        'explanation': 'explanation'
    }, inplace=True)

    # Check if required columns are present
    required_columns = ['subject', 'topic', 'question', 'explanation']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return

    st.title("MCAT Topics Explanation")

    # Create sidebars for Subject and Topic selection
    subjects = df['subject'].unique()
    selected_subject = st.sidebar.selectbox("Select Subject:", ["All"] + list(subjects))

    if selected_subject != "All":
        df = df[df['subject'] == selected_subject]

    topics = df['topic'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", ["All"] + list(topics))

    if selected_topic != "All":
        df = df[df['topic'] == selected_topic]

    # Initialize session state for question index
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    total_questions = len(df)

    # Adjust question index if it's out of bounds
    if total_questions == 0:
        st.write("No questions available for the selected subject and topic.")
        return

    # Ensure the question index is within the valid range
    st.session_state.question_index = min(st.session_state.question_index, total_questions - 1)

    # Display the current question in a box
    question_to_display = df.iloc[st.session_state.question_index]
    st.markdown("### Question")
    st.success(question_to_display['question'])

    # Display the explanation in an expander
    with st.expander("View Explanation"):
        st.write(question_to_display['explanation'])
        
        # Button to read the explanation text
        if st.button("Read Explanation Text"):
            explanation_text = question_to_display['explanation']
            audio_stream = gTTS(text=explanation_text, lang='en')
            audio_file_path = "explanation_output.mp3"
            audio_stream.save(audio_file_path)  # Save the audio file for the explanation
            st.success("Explanation text is being read!")
            st.audio(audio_file_path)  # Play the audio file

    st.markdown("### Navigation")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            if st.session_state.question_index > 0:
                st.session_state.question_index -= 1

    with col2:
        if st.button("Next"):
            if st.session_state.question_index < total_questions - 1:
                st.session_state.question_index += 1

    # Reset button
    if st.button("Reset"):
        st.session_state.question_index = 0

if __name__ == "__main__":
    main()
