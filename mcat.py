import docx

def clean_and_format_docx(input_file, output_file):
    doc = docx.Document(input_file)
    formatted_questions = []

    question_text = ""
    choices = []
    answer = ""
    explanation = ""

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Question"):
            if question_text and len(choices) == 4:  # Ensure exactly 4 choices
                formatted_questions.append((question_text, choices, answer, explanation))
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

    # Check the last question
    if question_text and len(choices) == 4:
        formatted_questions.append((question_text, choices, answer, explanation))

    # Write to a new document
    new_doc = docx.Document()
    for question, choices, answer, explanation in formatted_questions:
        new_doc.add_paragraph(question)
        for choice in choices:
            new_doc.add_paragraph(choice)
        new_doc.add_paragraph(f"Answer: {answer}")
        new_doc.add_paragraph(f"Explanation: {explanation}")
        new_doc.add_paragraph()  # Add a blank line for spacing

    new_doc.save(output_file)

if __name__ == "__main__":
    input_file = "mcat.docx"  # Original file
    output_file = "formatted_mcat.docx"  # Cleaned output file
    clean_and_format_docx(input_file, output_file)
