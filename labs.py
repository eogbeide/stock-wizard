import streamlit as st
import pandas as pd


# Create a timestamp to force a refresh
#today = datetime.datetime.now().date()
#st.write(f"Last updated: {today}")

# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labs.csv"
    df = pd.read_csv(url, encoding='ISO-8859-1')  # Specify encoding here
    return df

# Main function to run the app
def main():
    # Custom CSS to set font to Comic Sans MS and font size to 10
    st.markdown(
        """
        <style>
        body {
            font-family: 'Comic Sans MS';
            font-size: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
            # Display Description in a box
            st.subheader("Description")
            st.info(topic_data['Description'].values[0])  # Use st.info for a box

            # Display Questions and Answers in an expander
            st.subheader("Questions and Answers")
            with st.expander("View Questions and Answers"):
                st.write(topic_data['Questions and Answers'].values[0])  # Display questions and answers

        else:
            st.write("No data available for the selected topic.")
    else:
        st.error("The 'Subject' column is missing from the data.")

if __name__ == "__main__":
    main()
