import requests
import docx
import streamlit as st
import os

class Question:
    def __init__(self, text, choices, answer, explanation):
        self.text = text
        self.choices = choices
        self.answer = answer
        self.explanation = explanation

    def check_answer(self, user_answer):
        return user_answer == self.answer


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

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():  # Only process non-empty lines
            parts = paragraph.text.split('|')
            if len(parts) == 7:  # Expecting: Question|Choice1|Choice2|Choice3|Choice4|CorrectChoice|Explanation
                question_text = parts[0].strip()
                choices = [parts[i].strip() for i in range(1, 5)]
                answer = parts[5].strip()
                explanation = parts[6].strip()
                questions.append(Question(question_text, choices, answer, explanation))

    return questions


def take_quiz(questions):
    score = 0

    for question in questions:
        st.write(question.text)
        user_answer = st.selectbox("Select your answer:", question.choices)

        # Show the user's selected answer
        st.write(f"You selected: {user_answer}")

        if user_answer == question.choices[int(question.answer) - 1]:  # Correct answer check
            st.write("Correct!")
            score += 1
        else:
            st.write(f"Wrong! The correct answer is {question.choices[int(question.answer) - 1]}.")
            st.write(f"Explanation: {question.explanation}")

    st.write(f"\nYour score: {score}/{len(questions)}")


if __name__ == "__main__":
    raw_docx_url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/mcat.docx"
    
    # Download the docx file
    file_path = download_docx(raw_docx_url)
    
    if file_path:
        quiz_questions = read_questions_from_docx(file_path)
        
        st.title("Multiple Choice Quiz")
        take_quiz(quiz_questions)
