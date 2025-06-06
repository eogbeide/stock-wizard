import streamlit as sts
import pandas as pd
import re


# Create a timestamp to force a refresh
#today = datetime.datetime.now().date()
#st.write(f"Last updated: {today}")

#@st.cache_data
# Load data from CSV on GitHub
def load_data():
    url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/quiz.csv"
    try:
        quiz_data = pd.read_csv(url, encoding='ISO-8859-1')
        return quiz_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Sidebar for subject and topic selection
st.sidebar.title('Quiz Navigation')
subject = quiz_data['Subject'].unique()
selected_subject = st.sidebar.selectbox('Select Subject', subject)

# Filter questions based on selected subject
filtered_quiz = quiz_data[quiz_data['Subject'] == selected_subject]
topic = filtered_quiz['Topic'].unique()
selected_topic = st.sidebar.selectbox('Select Topic', topic)

# Further filter questions based on selected topic
filtered_quiz = filtered_quiz[filtered_quiz['Topic'] == selected_topic]

# Initialize session state for tracking question index
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0

# Function to display the current question and associated passage
def display_question(index):
    if index < len(filtered_quiz):
        question_row = filtered_quiz.iloc[index]
        
        # Display the passage
        st.write(f"### Passage:")
        st.write(question_row['Passage'])  # Show the passage
        
        # Display the question
        st.write(f"### Question {index + 1}: {question_row['Question']}")
        
        # Radio buttons for answers
        options = [question_row['Answer'], "Option 2", "Option 3", "Option 4"]  # Adjust options as needed
        answer = st.radio("Select your answer:", options)
        
        if st.button('Submit'):
            if answer == question_row['Answer']:
                st.success("Correct!")
            else:
                st.error("Incorrect!")
            st.write(f"**Explanation:** {question_row['Explanation']}")
            
            # Move to the next question
            st.session_state.question_index += 1
    else:
        st.write("Quiz completed! Thank you for participating.")
        st.session_state.question_index = 0  # Reset for future quizzes

# Display the current question and passage
display_question(st.session_state.question_index)
