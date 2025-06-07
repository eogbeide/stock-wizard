import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile

# Load data from Excel on GitHub
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/Product.xlsx"  # Update with your actual URL
    try:
        # Load the data
        data = pd.read_excel(url)
        
        # Clean the data by dropping rows with any empty cells
        data.dropna(inplace=True)
        
        # Reset the index after dropping rows
        data.reset_index(drop=True, inplace=True)
        
        st.write(data)  # Display the DataFrame for debugging
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails

# Load the product data
product_data = load_data()

if not product_data.empty:
    # Sidebar for category and subcategory selection
    st.sidebar.title('Interview Navigation')
    
    categories = product_data['Category'].unique()
    selected_category = st.sidebar.selectbox('Select Category', categories)

    subcategories = product_data[product_data['Category'] == selected_category]['Subcategory'].unique()
    selected_subcategory = st.sidebar.selectbox('Select Subcategory', subcategories)

    # Filter questions based on selected category and subcategory
    filtered_data = product_data[(product_data['Category'] == selected_category) & 
                                  (product_data['Subcategory'] == selected_subcategory)]

    # Initialize session state for question navigation
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    # Function to display the current question and provide options
    def display_question(index):
        if index < len(filtered_data):
            row = filtered_data.iloc[index]
            st.write(f"### Interviewer: {row['Interviewer']}")
            st.write(f"### Interviewee: {row['Interviewee']}")

            # Text-to-speech for interviewer and interviewee
            if st.button("Read Interviewer Aloud"):
                tts = gTTS(text=row['Interviewer'], lang='en')
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tts.save(f"{tmp_file.name}.mp3")
                    st.audio(f"{tmp_file.name}.mp3")

            if st.button("Read Interviewee Aloud"):
                tts = gTTS(text=row['Interviewee'], lang='en')
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tts.save(f"{tmp_file.name}.mp3")
                    st.audio(f"{tmp_file.name}.mp3")

            # Dropdown for interviewee response
            response_options = ["Select Response"] + row['Interviewee'].split(';')  # Assuming multiple responses are separated by semicolons
            selected_response = st.selectbox('Select Interviewee Response', response_options)

            return True
        else:
            st.write("End of Interview. Thank you!")
            return False

    # Display the current question
    if display_question(st.session_state.question_index):
        col1, col2 = st.columns(2)

        with col1:
            if st.button('Back'):
                if st.session_state.question_index > 0:
                    st.session_state.question_index -= 1

        with col2:
            if st.button('Next'):
                if st.session_state.question_index < len(filtered_data) - 1:
                    st.session_state.question_index += 1

else:
    st.warning("No data available.")
