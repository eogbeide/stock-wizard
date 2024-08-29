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
            choices.append(text)
        elif text.startswith("Answer:"):
            answer = text.split(":")[1].strip()
        elif text.startswith("Explanation:"):
            explanation = text.split(":", 1)[1].strip()

    if question_text:
        questions.append(Question(question_text, choices, answer, explanation))

    return questions

def display_question(question):
    st.write(question.text)
    # Use a unique key for the radio button; users can select only one option
    user_answer = st.radio("Select your answer:", question.choices, key="radio_answer")
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
        correct_answer = question.answer.strip()
        if st.session_state.user_answer == correct_answer:
            st.success("Correct!")
        else:
            st.error(f"Wrong! The correct answer is: {correct_answer}.")
        
        # Show explanation
        st.write("Explanation:")
        st.write(question.explanation)

        # Button to move to the next question
        if st.button("Next Question"):
            st.session_state.question_index += 1
            st.session_state.user_answer = None
            st.session_state.show_explanation = False

            if st.session_state.question_index >= len(quiz_questions):
                st.write("You have completed the quiz!")
                st.session_state.question_index = 0  # Reset for a new round

    else:
        user_answer = display_question(question)

        # Store the selected answer only if a selection has been made
        if st.button("Submit"):
            st.session_state.user_answer = user_answer
            st.session_state.show_explanation = True

if __name__ == "__main__":
    main()
