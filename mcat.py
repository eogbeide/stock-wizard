import pandas as pd
from docx import Document
import re

def extract_questions_from_docx(docx_file):
    document = Document(docx_file)
    questions = []
    
    # Regular expressions to match the question format
    question_pattern = re.compile(r'Question (\d+): (.*)')
    answer_pattern = re.compile(r'Answer: ([A-D]\)) (.*)')
    explanation_pattern = re.compile(r'Explanation:\s*(.*)')
    
    current_question = None
    options = {}
    
    for para in document.paragraphs:
        # Match question
        question_match = question_pattern.match(para.text.strip())
        if question_match:
            if current_question:
                # Save the previous question before starting a new one
                questions.append((current_question, options['A'], options['B'], options['C'], options['D'], answer, explanation))
            
            # Start a new question
            question_number = question_match.group(1)
            current_question = question_match.group(2)
            options = {}
            answer = None
            explanation = None
            
        # Match options A-D
        if current_question and para.text.strip() in ['A)', 'B)', 'C)', 'D)']:
            option_letter = para.text.strip()[0]
            option_text = para.text.strip()[3:]  # Get text after "A) "
            options[option_letter] = option_text
        
        # Match answer
        answer_match = answer_pattern.match(para.text.strip())
        if answer_match:
            answer = answer_match.group(1).strip()  # A), B), C), or D)

        # Match explanation
        explanation_match = explanation_pattern.match(para.text.strip())
        if explanation_match:
            explanation = explanation_match.group(1).strip()
    
    # Append the last question
    if current_question:
        questions.append((current_question, options.get('A'), options.get('B'), options.get('C'), options.get('D'), answer, explanation))
    
    return questions

def save_to_csv(questions, csv_file):
    # Prepare DataFrame
    df = pd.DataFrame(questions, columns=['S/N', 'Question', 'C', 'D', 'E', 'F', 'G', 'H'])
    # Save to CSV
    df.to_csv(csv_file, index=False)

def main():
    docx_file = 'mcat.docx'  # Path to your DOCX file
    csv_file = 'mcat.csv'      # Path to save the CSV file
    
    questions = extract_questions_from_docx(docx_file)
    
    # Transform the questions to include serial numbers
    questions_with_serial = [(i + 1,) + q for i, q in enumerate(questions)]
    
    save_to_csv(questions_with_serial, csv_file)
    print(f'Successfully saved questions to {csv_file}')

if __name__ == "__main__":
    main()
