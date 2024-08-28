import requests
import docx
import streamlit as st

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

    question_text = ""
    choices = []
    answer = ""
    explanation = ""

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Question"):
            if question_text:  # Save the previous question if it exists
                questions.append(Question(question_text, choices, answer, explanation))
            
            # Reset for the new question
            question_text = text
            choices = []
            answer = ""
            explanation = ""
        elif text.startswith("A)") or text.startswith("B)") or text.startswith("C)") or text.startswith("D)"):
            choices.append(text)
        elif text.startswith("Answer:"):
            answer = text.split(":")[1].strip()[0]  # Get the letter of the answer (A, B, C, or D)
        elif text.startswith("Explanation:"):
            explanation = text.split(":", 1)[1].strip()  # Get the explanation text

    # Append the last question
    if question_text:
        questions.append(Question(question_text, choices, answer, explanation))

    return questions


def take_quiz(questions):
    score = 0
    for question in questions:
        st.write(question.text)
        user_answer = st.selectbox("Select your answer:", question.choices, key=question.text)

        if st.button("Submit", key=question.text + "_submit"):
            # Show the user's selected answer
            st.write(f"You selected: {user_answer}")

            correct_answer = question.choices[ord(question.answer) - ord('A')]  # Get the correct answer choice
            if user_answer == correct_answer:  # Correct answer check
                st.write("Correct!")
                score += 1
            else:
                st.write(f"Wrong! The correct answer is {correct_answer}.")
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
