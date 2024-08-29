import streamlit as st
import docx

class Question:
    def __init__(self, text, choices, answer, explanation):
        self.text = text
        self.choices = choices
        self.answer = answer
        self.explanation = explanation

def read_questions_from_docx(file_path):
    questions = []
    doc = docx.Document(file_path)

    question_text = ""
    choices = []
    answer = ""
    explanation = ""

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Question"):
            if question_text:
                questions.append(Question(question_text, choices, answer, explanation))
            question_text = text
            choices = []
            answer = ""
            explanation = ""
        elif text.startswith("A)") or text.startswith("B)") or text.startswith("C)") or text.startswith("D)"):
            choices.append(text)  # Keep the full text for choices
        elif text.startswith("Answer:"):
            answer = text.split(":")[1].strip()
        elif text.startswith("Explanation:"):
            explanation = text.split(":", 1)[1].strip()

    if question_text:
        questions.append(Question(question_text, choices, answer, explanation))

    return questions

def display_question(question):
    st.write(question.text)
    
    # Create explicit A, B, C, D labeled choices for radio buttons
    labeled_choices = [f"A) {question.choices[0].split(') ')[1]}",
                       f"B) {question.choices[1].split(') ')[1]}",
                       f"C) {question.choices[2].split(') ')[1]}",
                       f"D) {question.choices[3].split(') ')[1]}"]
    
    # Create a radio button for the choices
    user_answer = st.radio("Select your answer (A, B, C, D):", labeled_choices, key="answer_select")
    return user_answer

def main():
    file_path = "mcat.docx"  # Path to your .docx file

    # Read questions from the docx file
    quiz_questions = read_questions_from_docx(file_path)
    
    st.title("Multiple Choice Quiz")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.user_answer = None
        st.session_state.show_explanation = False

    question_index = st.session_state.question_index
    question = quiz_questions[question_index]

    if st.session_state.show_explanation:
        correct_answer = question.answer.strip().split(",")  # Assuming answers are comma-separated
        if st.session_state.user_answer in correct_answer:
            st.success("Correct!")
        else:
            st.error(f"Wrong! The correct answer is: {', '.join(correct_answer)}.")
        
        # Show explanation
        st.write("Explanation:")
        st.write(question.explanation)

        # Button to move to the next question
        col1, col2 = st.columns(2)

        with col1:
            # Disable the button if it's the last question
            next_disabled = st.session_state.question_index >= len(quiz_questions) - 1
            if st.button("Next Question", disabled=next_disabled):
                st.session_state.question_index += 1
                st.session_state.user_answer = None
                st.session_state.show_explanation = False

                # Reset if at the end of the quiz
                if st.session_state.question_index >= len(quiz_questions):
                    st.write("You have completed the quiz!")
                    st.session_state.question_index = 0  # Reset for a new round

        with col2:
            # Disable the button if at the first question
            back_disabled = st.session_state.question_index == 0
            if st.button("Back", disabled=back_disabled):
                st.session_state.question_index -= 1
                st.session_state.user_answer = None
                st.session_state.show_explanation = False

    else:
        user_answer = display_question(question)

        # Store the selected answer only if a selection has been made
        submit_disabled = user_answer is None  # No selection made
        if st.button("Submit", disabled=submit_disabled):
            st.session_state.user_answer = user_answer
            st.session_state.show_explanation = True

if __name__ == "__main__":
    main()
