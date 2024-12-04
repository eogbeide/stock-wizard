import streamlit as st
import pandas as pd
import re


# Create a timestamp to force a refresh
#today = datetime.datetime.now().date()
#st.write(f"Last updated: {today}")

@st.cache_data
# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/flashcards.csv"
    try:
        df = pd.read_csv(url, encoding='ISO-8859-1')
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

    # Sidebar for topic selection based on selected subject
    if not subject_data.empty:
        topics = subject_data['Topic'].unique()
        selected_topic = st.sidebar.selectbox("Select Topic:", topics)

        # Filter data based on selected topic
        topic_data = subject_data[subject_data['Topic'] == selected_topic]

        # Initialize session state for question index
        if 'current_row_index' not in st.session_state:
            st.session_state.current_row_index = 0  # Start with the first row

        # Display the current question
        if st.session_state.current_row_index < len(topic_data):
            current_question_row = topic_data.iloc[st.session_state.current_row_index]  # Get the current row
            
            # Extract questions and answers using regex
            st.session_state.qa_pairs = re.findall(
                r'Flashcard \d+:\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\s*Flashcard \d+:|$)', 
                current_question_row['Questions and Answers'], 
                re.DOTALL
            )

            if st.session_state.qa_pairs:
                # Initialize question index if not set
                if 'question_index' not in st.session_state:
                    st.session_state.question_index = 0  # Initialize question index to zero

                # Get the current question-answer pair
                question, answer = st.session_state.qa_pairs[st.session_state.question_index]
                flashcard_number = st.session_state.question_index + 1  # Flashcard number based on question index
                st.subheader(f"Flashcard {flashcard_number}: {question.strip()}")  # Display Flashcard # with question

                # Expandable dropdown for the answer
                with st.expander("Show Answer"):
                    st.info(answer.strip())

                # Navigation buttons
                col1, col2 = st.columns(2)

                # Back Button
                with col1:
                    if st.button("Back"):
                        if st.session_state.question_index > 0:
                            st.session_state.question_index -= 1
                        else:
                            st.session_state.question_index = 0  # Remain at the first question

                # Next Button
                with col2:
                    if st.button("Next"):
                        if st.session_state.question_index < len(st.session_state.qa_pairs) - 1:
                            st.session_state.question_index += 1
                        else:
                            # Move to the next row if available
                            st.session_state.current_row_index += 1
                            st.session_state.question_index = 0  # Reset question index for the next row

            else:
                st.error("No valid question and answer pairs found in the format. Please check the CSV file.")

        else:
            st.write("No more flashcards available for the selected topic.")

    else:
        st.write("No topics available for the selected subject.")

if __name__ == "__main__":
    main()
