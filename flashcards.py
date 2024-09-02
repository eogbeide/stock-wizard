import streamlit as st
import pandas as pd

# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/flashcards.csv"
    df = pd.read_csv(url)
    return df

# Main function to run the app
def main():
    st.title("Flashcards")

    # Load the data
    df = load_data()

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
        st.subheader(current_question['Questions'])

        # Answer display logic
        if st.button("Show Answer"):
            st.info(current_question['Answers'])

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
        st.write("No flashcards available for the selected subject.")

if __name__ == "__main__":
    main()
