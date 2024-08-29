import streamlit as st
import pandas as pd

class Question:
    def __init__(self, text, choices, answer, explanation):
        self.text = text
        self.choices = choices
        self.answer = answer
        self.explanation = explanation

def read_questions_from_csv(file_path):
    questions = []
    df = pd.read_csv(file_path)

    for index, row in df.iterrows():
        question_text = row['Question']
        choices = [row['C'], row['D'], row['E'], row['F']]  # Adjusted for new columns
        answer = row['G'].strip()  # Answer from Column G
        explanation = row['H']  # Explanation from Column H
        questions.append(Question(question_text, choices, answer, explanation))

    return questions

def display_question(question):
    st.write(question.text)
    
    # Create labeled choices for radio buttons
    labeled_choices = [f"{chr(67 + i)}) {choice}" for i, choice in enumerate(question.choices)]
    
    # Display choices
    user_answer = st.radio("Select your answer:", labeled_choices, key="answer_select")
    return user_answer

def main():
    file_path = "mcat.csv"  # Path to your CSV file

    # Read questions from the CSV file
    quiz_questions = read_questions_from_csv(file_path)
    
    st.title("Multiple Choice Quiz")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.user_answer = None
        st.session_state.show_explanation = False

    question_index = st.session_state.question_index
    question = quiz_questions[question_index]

    if st.session_state.show_explanation:
        # Check if the selected answer matches the expected answer
        user_answer_index = ord(st.session_state.user_answer[0]) - 67  # Adjust for C, D, E, F
        user_answer = question.choices[user_answer_index]  # Get the selected choice
        
        if user_answer == question.answer:
            st.success("Correct!")
        else:
            st.error(f"Wrong! The correct answer is: {question.answer}.")
        
        # Show explanation
        st.write("Explanation:")
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
                    st.session_state.question_index = 0  # Reset for a new round

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
