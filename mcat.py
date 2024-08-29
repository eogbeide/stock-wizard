import streamlit as st
import docx

class Question:
    def __init__(self, text, choices, answer, explanation):
        self.text = text
        self.choices = choices
        self.answer = answer
        self.explanation = explanation

def clean_and_format_questions(file_path):
    questions = []
    doc = docx.Document(file_path)

    question_text = ""
    choices = []
    answer = ""
    explanation = ""

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Question"):
            if question_text and len(choices) == 4:  # Ensure exactly 4 choices
                questions.append((question_text, choices, answer, explanation))
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

    # Check the last question
    if question_text and len(choices) == 4:
        questions.append((question_text, choices, answer, explanation))

    # Format each question for Streamlit
    formatted_questions = []
    for question_text, choices, answer, explanation in questions:
        formatted_choices = [
            f"A) {choices[0].split(') ')[1]}",
            f"B) {choices[1].split(') ')[1]}",
            f"C) {choices[2].split(') ')[1]}",
            f"D) {choices[3].split(') ')[1]}"
        ]
        formatted_questions.append((question_text, formatted_choices, answer, explanation))

    return formatted_questions

def display_question(question):
    st.write(question[0])  # Display the question text
    
    # Create radio buttons for the choices
    user_answer = st.radio("Select your answer (A, B, C, D):", question[1], key="answer_select")
    return user_answer

def main():
    file_path = "mcat.docx"  # Path to your .docx file

    # Clean and format questions from the docx file
    quiz_questions = clean_and_format_questions(file_path)

    # Check if there are any valid questions
    if not quiz_questions:
        st.error("No valid questions available for the quiz. Please check your .docx file format.")
        return

    st.title("Multiple Choice Quiz")

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
        st.session_state.user_answer = None
        st.session_state.show_explanation = False

    question_index = st.session_state.question_index
    question = quiz_questions[question_index]

    if st.session_state.show_explanation:
        correct_answer = question[2].strip().split(",")  # Assuming answers are comma-separated
        if st.session_state.user_answer in correct_answer:
            st.success("Correct!")
        else:
            st.error(f"Wrong! The correct answer is: {', '.join(correct_answer)}.")
        
        # Show explanation
        st.write("Explanation:")
        st.write(question[3])

        # Buttons to navigate questions
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
