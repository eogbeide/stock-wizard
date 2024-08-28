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

def take_quiz(questions):
    score = 0
    question_index = 0

    while question_index < len(questions):
        question = questions[question_index]

        st.write(question.text)

        # User selects an answer with a unique key
        user_answer = st.radio("Select your answer:", question.choices, key=f"radio_{question_index}")

        if st.button("Submit", key=f"submit_{question_index}"):
            st.write(f"You selected: {user_answer}")

            correct_answer = question.answer.strip()
            if user_answer == correct_answer:
                st.write("Correct!")
                score += 1
            else:
                st.write(f"Wrong! The correct answer is: {correct_answer}.")
            
            # Show explanation
            st.write("Explanation:")
            st.write(question.explanation)

            question_index += 1

            if question_index < len(questions):
                st.success("Next question:")
            else:
                st.write(f"\nYour final score: {score}/{len(questions)}")
                break

    st.write(f"\nYour score: {score}/{len(questions)}")

if __name__ == "__main__":
    file_path = "mcat.docx"  # Path to your .docx file

    # Read questions from the docx file
    quiz_questions = read_questions_from_docx(file_path)
    
    st.title("Multiple Choice Quiz")
    take_quiz(quiz_questions)
