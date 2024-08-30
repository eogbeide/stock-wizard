import streamlit as st
import pandas as pd

# Function to read questions from CSV
def read_questions_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Main function to run the Streamlit app
def main():
    # URL to the CSV file on GitHub
    file_path = "mcattopics.csv"  # Update with your actual GitHub URL

    # Read questions from the CSV file
    df = read_questions_from_csv(file_path)

    st.title("MCAT Topics Quiz")

    # Create sidebars for Subject and Topic selection
    subjects = df['Subject'].unique()
    selected_subject = st.sidebar.selectbox("Select Subject:", ["All"] + list(subjects))

    if selected_subject != "All":
        df = df[df['Subject'] == selected_subject]

    topics = df['Topics'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", ["All"] + list(topics))

    if selected_topic != "All":
        df = df[df['Topics'] == selected_topic]

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
        st.write(f"**Question {index + 1}**: {row['Question']}")
        st.write(f"**Explanation**: {row['Explanation']}")

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
