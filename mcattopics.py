import streamlit as st
import pandas as pd
from urllib.error import URLError

# Function to read questions from CSV
def read_questions_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # Try with UTF-8 encoding
        return df
    except UnicodeDecodeError:
        st.error("Error decoding the CSV file. Trying a different encoding.")
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback to ISO-8859-1
            return df
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on failure
    except URLError as e:
        st.error(f"Error fetching data: {e.reason}")
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
    df.columns = df.columns.str.strip()  # Remove spaces
    df.columns = df.columns.str.lower()   # Normalize case

    # Check if required columns are present
    required_columns = ['subject', 'topics', 'question', 'explanation']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return

    st.title("MCAT Topics Quiz")

    # Create sidebars for Subject and Topic selection
    subjects = df['subject'].unique()
    selected_subject = st.sidebar.selectbox("Select Subject:", ["All"] + list(subjects))

    if selected_subject != "All":
        df = df[df['subject'] == selected_subject]

    topics = df['topics'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", ["All"] + list(topics))

    if selected_topic != "All":
        df = df[df['topics'] == selected_topic]

    # Select the number of questions to display
    num_questions = st.sidebar.selectbox("Select number of questions to display:", [20, 40, 60], index=0)

    # Initialize session state
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    # Calculate the total number of questions
    total_questions = len(df)

    # Display current question range
    start_index = st.session_state.question_index
    end_index = start_index + num_questions
    questions_to_display = df.iloc[start_index:end_index]

    # Display questions and explanations
    for index, row in questions_to_display.iterrows():
        st.write(f"**Question {index + 1}**: {row['question']}")
        st.write(f"**Explanation**: {row['explanation']}")

    # Navigation buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            if st.session_state.question_index > 0:
                st.session_state.question_index -= num_questions

    with col2:
        if st.button("Next"):
            if end_index < total_questions:
                st.session_state.question_index += num_questions

    # Reset button
    if st.button("Reset"):
        st.session_state.question_index = 0

if __name__ == "__main__":
    main()
