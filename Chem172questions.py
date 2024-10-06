import streamlit as st
import pandas as pd
import requests
from io import StringIO  # Import StringIO from the io module


# Load the CSV file from GitHub
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/Chem172_questions.csv'  # Raw URL for the CSV
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad requests
    return pd.read_csv(StringIO(response.text))  # Use StringIO from io

# Main function
def main():
    # Load data
    data = load_data()
    
    # Sidebar for chapter selection
    st.sidebar.title("Select Chapter")
    chapters = data['Chapter'].unique()
    selected_chapter = st.sidebar.selectbox("Choose a Chapter", chapters)

    # Filter data based on selected chapter
    chapter_data = data[data['Chapter'] == selected_chapter]

    # Initialize session state for question index
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    # Display current question
    current_question = chapter_data.iloc[st.session_state.question_index]
    
    st.title(f"Chapter: {selected_chapter}")
    st.subheader(f"Question {st.session_state.question_index + 1}: {current_question['Question']}")
    
    if st.button("Show Answer"):
        st.write(current_question['Answer and Explanation'])

    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.question_index > 0:
            if st.button("Back"):
                st.session_state.question_index -= 1
    
    with col2:
        if st.session_state.question_index < len(chapter_data) - 1:
            if st.button("Next"):
                st.session_state.question_index += 1

# Run the app
if __name__ == "__main__":
    main()





<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Equations Display</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
        }
        .equation {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            transition: background 0.3s;
        }
        .equation:hover {
            background-color: #f0f0f0;
        }
        .details {
            display: none;
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Math Equations</h1>

    <div class="equation">
        <strong>Equation 1:</strong> E = mc²
        <button onclick="toggleDetails(1)">Show Details</button>
        <div class="details" id="details-1">
            <p><strong>Answer:</strong> This formula shows the equivalence of mass and energy.</p>
            <p><strong>Explanation:</strong> E represents energy, m represents mass, and c represents the speed of light.</p>
        </div>
    </div>

    <div class="equation">
        <strong>Equation 2:</strong> a² + b² = c²
        <button onclick="toggleDetails(2)">Show Details</button>
        <div class="details" id="details-2">
            <p><strong>Answer:</strong> This is the Pythagorean theorem.</p>
            <p><strong>Explanation:</strong> It relates the lengths of the sides of a right triangle.</p>
        </div>
    </div>

    <div class="equation">
        <strong>Equation 3:</strong> F = ma
        <button onclick="toggleDetails(3)">Show Details</button>
        <div class="details" id="details-3">
            <p><strong>Answer:</strong> This is Newton's second law of motion.</p>
            <p><strong>Explanation:</strong> F represents force, m represents mass, and a represents acceleration.</p>
        </div>
    </div>
</div>

<script>
    function toggleDetails(equationNumber) {
        const details = document.getElementById(`details-${equationNumber}`);
        if (details.style.display === "none" || details.style.display === "") {
            details.style.display = "block";
        } else {
            details.style.display = "none";
        }
    }
</script>

</body>
</html>
