import streamlit as st
import pandas as pd

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
            # Initialize session state for question index
            if 'question_index' not in st.session_state:
                st.session_state.question_index = 0

            # Display Description in a box
            st.subheader("Description")
            st.info(topic_data['Description'].values[st.session_state.question_index])  # Display current description

            # Display Questions and Answers in an expander
            st.subheader("Questions and Answers")
            with st.expander("View Questions and Answers"):
                st.write(topic_data['Questions and Answers'].values[st.session_state.question_index])  # Display current questions and answers

            # Navigation buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Back"):
                    if st.session_state.question_index > 0:
                        st.session_state.question_index -= 1

            with col2:
                if st.button("Next"):
                    if st.session_state.question_index < len(topic_data) - 1:
                        st.session_state.question_index += 1

        else:
            st.write("No data available for the selected topic.")
    else:
        st.error("The 'Subject' column is missing from the data.")

if __name__ == "__main__":
    main()
