import requests
import docx
import streamlit as st

class Question:
    def __init__(self, text, choices, answer, explanation):
        self.text = text
        self.choices = choices
        self.answer = answer
        self.explanation = explanation

def download_docx(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("mcat.docx", "wb") as f:
            f.write(response.content)
        return "mcat.docx"
    else:
        st.error("Failed to download the document.")
        return None

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

        # Create separate radio buttons for each option A, B, C, and D
        user_answer = st.radio("Select your answer:", question.choices, key=f"answer_{question_index}")

        if st.button("Submit", key=f"submit_{question_index}"):
            st.write(f"You selected: {user_answer}")

            correct_answer = question.answer.strip()
            if user_answer == correct_answer:
                st.write("Correct!")
                score += 1
            else:
                st.write(f"Wrong! The correct answer is {correct_answer}.")
            
            # Show explanation
            st.write("Explanation:")
            st.write(question.explanation)

            # Move to the next question
            question_index += 1

            if question_index < len(questions):
                st.success("Next question:")
            else:
                st.write(f"\nYour final score: {score}/{len(questions)}")
                break

    st.write(f"\nYour score: {score}/{len(questions)}")

if __name__ == "__main__":
    raw_docx_url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcat.docx"
    
    # Download the docx file
    file_path = download_docx(raw_docx_url)
    
    if file_path:
        quiz_questions = read_questions_from_docx(file_path)
        
        st.title("Multiple Choice Quiz")
        take_quiz(quiz_questions)
