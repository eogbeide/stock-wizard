import streamlit as st
import pandas as pd
from gtts import gTTS
import os
import tempfile

# Load data from Excel on GitHub
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"  # Link to the Excel file
    try:
        quiz_data = pd.read_excel(url)
        return quiz_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Load the quiz data
quiz_data = load_data()

# Debugging: Display the columns in the DataFrame
#st.write("Columns in quiz_data:", quiz_data.columns.tolist())

# Sidebar for subject and topic selection
st.sidebar.title('Quiz Navigation')
if not quiz_data.empty:
    try:
        subject = quiz_data['Subject'].unique()
        selected_subject = st.sidebar.selectbox('Select Subject', subject)

        # Filter questions based on selected subject
        filtered_quiz = quiz_data[quiz_data['Subject'] == selected_subject]
        topic = filtered_quiz['Topic'].unique()
        selected_topic = st.sidebar.selectbox('Select Topic', topic)

        # Further filter questions based on selected topic
        filtered_quiz = filtered_quiz[filtered_quiz['Topic'] == selected_topic]

        # Initialize session state for tracking question index and answer
        if 'question_index' not in st.session_state:
            st.session_state.question_index = 0
        if 'selected_answer' not in st.session_state:
            st.session_state.selected_answer = None
        if 'submitted' not in st.session_state:
            st.session_state.submitted = False

        # Function to display the current question and associated passage
        def display_question(index):
            if index < len(filtered_quiz):
                question_row = filtered_quiz.iloc[index]
                
                # Display the passage, formatting it into paragraphs with spacing
                st.write("### Passage:")
                passage = question_row['Passage'].replace('\n', '<br><br>')  # Replace newlines with <br> for spacing
                st.markdown(passage, unsafe_allow_html=True)  # Display formatted passage
                
                # Text-to-speech button
                if st.button("Read Passage Aloud"):
                    passage_text = question_row['Passage']
                    tts = gTTS(text=passage_text, lang='en')
                    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                        tts.save(f"{tmp_file.name}.mp3")
                        # Play the audio
                        st.audio(f"{tmp_file.name}.mp3")

                # Display the question
                st.write(f"### Question {index + 1}: {question_row['Question']}")
                
                # Prepare all options from the Answer column
                options = question_row['Answer'].split(';')  # Assuming answers are separated by semicolons
                options = [option.strip() for option in options]  # Clean up options

                # Show all options in a single radio button
                st.write("Click to select your answer:")
                selected_option = st.radio("Select your answer:", options, key="answer_radio")

                # Submit button to check the answer
                if st.button('Submit'):
                    st.session_state.submitted = True
                    st.session_state.selected_answer = selected_option  # Store selected answer

                    correct_answer = options[0].strip()  # First option is assumed to be the correct answer
                    if st.session_state.selected_answer == correct_answer:
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.write(f"**Explanation:** {question_row['Explanation']}")

                return True
            else:
                st.write("Quiz completed! Thank you for participating.")
                return False

        # Display the current question and passage
        if display_question(st.session_state.question_index):
            col1, col2 = st.columns(2)

            with col1:
                if st.button('Back'):
                    if st.session_state.question_index > 0:
                        st.session_state.question_index -= 1
                        st.session_state.selected_answer = None  # Reset selected answer

            with col2:
                if st.button('Next'):
                    if st.session_state.question_index < len(filtered_quiz) - 1:
                        st.session_state.question_index += 1
                        st.session_state.selected_answer = None  # Reset selected answer

    except KeyError as e:
        st.error(f"Column not found: {e}")
else:
    st.warning("No quiz data available.")
