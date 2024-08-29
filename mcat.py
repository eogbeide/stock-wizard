import streamlit as st
import pandas as pd

class Question:
    def __init__(self, text, choices, answer, explanation, serial_number):
        self.text = text
        self.choices = choices
        self.answer = answer
        self.explanation = explanation
        self.serial_number = serial_number

def read_questions_from_csv(file_path):
    questions = []
    df = pd.read_csv(file_path)

    # Check for expected columns
    expected_columns = ['S/N', 'Question', 'C', 'D', 'E', 'F', 'G', 'H']
    for col in expected_columns:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return []

    for index, row in df.iterrows():
        question_text = row['Question']
        choices = [row['C'], row['D'], row['E'], row['F']]
        answer = row['G'].replace('Answer:', '').strip()  # Clean the answer
        explanation = row['H']  # Explanation from Column H
        serial_number = row['S/N']  # S/N from Column A
        questions.append(Question(question_text, choices, answer, explanation, serial_number))

    return questions

def display_question(question):
    # Display the S/N and question text
    st.write(f"**{question.serial_number}. {question.text}**")
    
    # Create labeled choices for radio buttons without repeating headers
    labeled_choices = [f"{chr(65 + i)}) {choice.strip()}" for i, choice in enumerate(question.choices)]
    
    # Display choices
    user_answer = st.radio("Select your answer:", labeled_choices, key="answer_select")
    return user_answer

def main():
    file_path = "mcatss.csv"  # Path to your CSV file

    # Read questions from the CSV file
    quiz_questions = read_questions_from_csv(file_path)
    
    if not quiz_questions:
        st.write("No questions available.")
        return
    
    st.title("Multiple Choice Quiz")

    # Initialize session state variables
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.user_answer = None
        st.session_state.show_explanation = False
        st.session_state.correct_answers = 0  # Initialize correct_answers

    question_index = st.session_state.question_index
    question = quiz_questions[question_index]

    if st.session_state.show_explanation:
        # Check if the selected answer matches the expected answer
        user_answer_index = ord(st.session_state.user_answer[0]) - 65  # Adjust for A, B, C, D
        user_answer = question.choices[user_answer_index].strip()  # Get the selected choice and strip spaces
        
        # Compare with the correct answer (cleaned)
        if user_answer.lower() == question.answer.lower():  # Case-insensitive comparison
            st.success("Correct!")
            st.session_state.correct_answers += 1  # Update score
        else:
            st.error(f"Wrong! The correct answer is: {question.answer}.")
        
        # Show explanation
        st.write(question.explanation)

        # Navigation buttons
        col1, col2 = st.columns(2)

        with col1:
            next_disabled = st.session_state.question_index >= len(quiz_questions) - 1
            if st.button("Next Question", disabled=next_disabled):
                st.session_state.question_index += 1
                st.session_state.user_answer = None
                st.session_state.show_explanation = False

                if st.session_state.question_index >= len(quiz_questions):
                    st.write("You have completed the quiz!")
                    st.write(f"Your score: {st.session_state.correct_answers}/{len(quiz_questions)}")
                    st.session_state.question_index = 0  # Reset for a new round
                    st.session_state.correct_answers = 0  # Reset score

        with col2:
            back_disabled = st.session_state.question_index == 0
            if st.button("Back", disabled=back_disabled):
                st.session_state.question_index -= 1
                st.session_state.user_answer = None
                st.session_state.show_explanation = False

    else:
        user_answer = display_question(question)

        submit_disabled = user_answer is None  # No selection made
        if st.button("Submit", disabled=submit_disabled):
            st.session_state.user_answer = user_answer
            st.session_state.show_explanation = True

if __name__ == "__main__":
    main()
