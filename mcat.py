import streamlit as st

# Sample text with question, options, answer, and explanation.
text = """
Question 1:
Which of the following correctly describes the two main divisions of the vertebrate nervous system?
A) Central and peripheral nervous systems
B) Somatic and autonomic nervous systems
C) Sympathetic and parasympathetic nervous systems
D) Cerebral and cerebellar nervous systems
Answer: A) Central and peripheral nervous systems
Explanation:
The vertebrate nervous system is organized into two main divisions: the central nervous system (CNS), which includes the brain and spinal cord, and the peripheral nervous system (PNS), which consists of all the nerves that extend from the CNS to the rest of the body.
"""

# Function to parse the question and options
def parse_question(text):
    lines = text.strip().split('\n')
    question = lines[1].strip()
    options = [line.strip() for line in lines[2:6]]
    answer = lines[7].strip().split(": ")[1]
    explanation = lines[9].strip()
    return question, options, answer, explanation

# Extract information from the text
question, options, correct_answer, explanation = parse_question(text)

# Streamlit app layout
st.title("Multiple Choice Question")
st.write(question)

# User selects an answer
user_answer = st.radio("Select your answer:", options)

# Submit button
if st.button("Submit"):
    st.write(f"You selected: {user_answer}")
    
    # Check the selected answer
    if user_answer == correct_answer:
        st.write("Correct!")
    else:
        st.write(f"Wrong! The correct answer is: {correct_answer}.")
    
    # Show explanation
    st.write("Explanation:")
    st.write(explanation)
