import streamlit as st
import pandas as pd

# Load data from CSV on GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/labs.csv"  # Update with your GitHub URL
    df = pd.read_csv(url)
    return df

# Main function to run the app
def main():
    st.title("MCAT Labs")

    # Load the data
    df = load_data()

    # Clean column names
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace

    # Sidebar for topic selection
    topics = df['Topic'].unique()
    selected_topic = st.sidebar.selectbox("Select Topic:", topics)

    # Filter data based on selected topic
    topic_data = df[df['Topic'] == selected_topic]

    if not topic_data.empty:
        st.subheader("Description")
        st.write(topic_data['Description'].values[0])  # Display the description

        st.subheader("Questions and Answers")
        st.write(topic_data['Questions and Answers'].values[0])  # Display questions and answers
    else:
        st.write("No data available for the selected topic.")

if __name__ == "__main__":
    main()
