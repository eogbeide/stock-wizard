import streamlit as st
import pandas as pd
import re

# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/flashcards.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Main function to run the app
def main():
    st.title("Flashcards")

    # Load the data
    df = load_data()

    # Check if the DataFrame is empty
    if df.empty:
        st.error("No data available. Please check the CSV file URL.")
        return

    # Clean column names
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace
    st.write(df.columns)  # Debugging: print column names to check available columns

    # Check if 'Topic' exists
    if 'Topic' not in df.columns:
        st.error("The 'Topic' column does not exist in the data.")
        return

    # Sidebar for topic selection
    topics = df['Topic'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", topics)

    # Filter data based on selected topic
    topic_data = df[df['Topic'] == selected_topic]

    # Initialize session state for question index
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []  # Store question-answer pairs
        st.session_state.question_index = 0  # Initialize question index
        st.session_state.show_answer = [False]  # Track if the answer should be shown for each question

    # Display the current question
    if not topic_data.empty:
        current_question = topic_data.iloc[0]  # We only take the first row for the selected topic
        
        # Extract questions and answers using regex
        st.session_state.qa_pairs = re.findall(
            r'Flashcard \d+:\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\s*Flashcard \d+:|$)', 
            current_question['Questions and Answers'], 
            re.DOTALL
        )

        if st.session_state.qa_pairs:
            # Get the current question-answer pair
            question, answer = st.session_state.qa_pairs[st.session_state.question_index]
            st.subheader(question.strip())

            # Answer display logic
            if st.session_state.show_answer[st.session_state.question_index]:
                st.info(answer.strip())
            else:
                # Button to show the answer
                if st.button("Show Answer"):
                    st.session_state.show_answer[st.session_state.question_index] = True  # Set flag to show the answer for the current question

            # Navigation buttons
            col1, col2 = st.columns(2)

            # Back Button
            with col1:
                if st.button("Back"):
                    if st.session_state.question_index > 0:
                        st.session_state.question_index -= 1
                        st.session_state.show_answer.append(False)  # Reset answer display for new question
                    else:
                        st.session_state.question_index = 0  # Remain at the first question

            # Next Button
            with col2:
                if st.button("Next"):
                    if st.session_state.question_index < len(st.session_state.qa_pairs) - 1:
                        st.session_state.question_index += 1
                        st.session_state.show_answer.append(False)  # Reset answer display for new question
                    else:
                        st.session_state.question_index = len(st.session_state.qa_pairs) - 1  # Stay at the last question

        else:
            st.error("No valid question and answer pairs found in the format. Please check the CSV file.")

    else:
        st.write("No flashcards available for the selected topic.")

if __name__ == "__main__":
    main()
