import streamlit as st
import pandas as pd

# Load data from Excel on GitHub
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/quiz.xlsx"  # Link to the Excel file
    try:
        quiz_data = pd.read_excel(url)
        return quiz_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Load the quiz data
quiz_data = load_data()

# Debugging: Display the columns in the DataFrame
#st.write("Columns in quiz_data:", quiz_data.columns.tolist())

# Sidebar for subject and topic selection
st.sidebar.title('Quiz Navigation')
if not quiz_data.empty:
    try:
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
    except KeyError as e:
        st.error(f"Column not found: {e}")
else:
    st.warning("No quiz data available.")
