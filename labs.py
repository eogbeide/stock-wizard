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

    # Initialize session state
    if 'subject_index' not in st.session_state:
        st.session_state.subject_index = 0
    if 'topic_index' not in st.session_state:
        st.session_state.topic_index = 0

    # Sidebar for subject selection
    if 'Subject' in df.columns:
        subjects = df['Subject'].unique()
        selected_subject = subjects[st.session_state.subject_index]

        # Filter data based on selected subject
        subject_data = df[df['Subject'] == selected_subject]

        # Sidebar for topic selection
        topics = subject_data['Topic'].unique()
        selected_topic = topics[st.session_state.topic_index]

        # Filter data based on selected topic
        topic_data = subject_data[subject_data['Topic'] == selected_topic]

        if not topic_data.empty:
            # Display Description in a box
            st.subheader("Description")
            st.info(topic_data['Description'].values[0])  # Use st.info for a box

            # Display Questions and Answers in an expander
            st.subheader("Questions and Answers")
            with st.expander("View Questions and Answers"):
                st.write(topic_data['Questions and Answers'].values[0])  # Display questions and answers

            # Navigation buttons
            col1, col2 = st.columns(2)

            # Back Button
            with col1:
                if st.button("Back Topic"):
                    if st.session_state.topic_index > 0:
                        st.session_state.topic_index -= 1
                    else:
                        st.session_state.subject_index = (st.session_state.subject_index - 1) % len(subjects)
                        st.session_state.topic_index = 0

            # Next Button
            with col2:
                if st.button("Next Topic"):
                    if st.session_state.topic_index < len(topics) - 1:
                        st.session_state.topic_index += 1
                    else:
                        st.session_state.subject_index = (st.session_state.subject_index + 1) % len(subjects)
                        st.session_state.topic_index = 0

        else:
            st.write("No data available for the selected topic.")
    else:
        st.error("The 'Subject' column is missing from the data.")

if __name__ == "__main__":
    main()
