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

    # Sidebar for subject selection
    subjects = df['Subject'].unique()
    selected_subject = st.sidebar.selectbox("Select Subject:", subjects)

    # Filter data based on selected subject
    subject_data = df[df['Subject'] == selected_subject]

    # Initialize session state for question index
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    # Display the current question
    if not subject_data.empty:
        current_question = subject_data.iloc[st.session_state.question_index]
        
        # Extract questions and answers using regex
        qa_pairs = re.findall(r'Flashcard \d+:\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\s*Flashcard \d+:|$)', current_question['Questions and Answers'], re.DOTALL)

        if qa_pairs:
            question, answer = qa_pairs[0]  # Get the first question-answer pair
            st.subheader(question.strip())

            # Answer display logic
            if st.button("Show Answer"):
                st.info(answer.strip())

            # Navigation buttons
            col1, col2 = st.columns(2)

            # Back Button
            with col1:
                if st.button("Back"):
                    if st.session_state.question_index > 0:
                        st.session_state.question_index -= 1

            # Next Button
            with col2:
                if st.button("Next"):
                    if st.session_state.question_index < len(subject_data) - 1:
                        st.session_state.question_index += 1
        else:
            st.error("No valid question and answer pairs found in the format. Please check the CSV file.")

    else:
        st.write("No flashcards available for the selected subject.")

if __name__ == "__main__":
    main()
