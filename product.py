import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile

# Load data from Excel on GitHub
def load_data():
    url = "https://github.com/eogbeide/stock-wizard/raw/main/Product.xlsx"
    try:
        df = pd.read_excel(url)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

product_data = load_data()

if product_data.empty:
    st.warning("No data available.")
    st.stop()

st.sidebar.title('Interview Navigation')

# 1) Category filter
categories = product_data['Category'].unique()
selected_category = st.sidebar.selectbox('Select Category', categories)

# 2) Subcategory filter
sub_df = product_data[product_data['Category'] == selected_category]
subcategories = sub_df['Subcategory'].unique()
selected_subcategory = st.sidebar.selectbox('Select Subcategory', subcategories)

# 3) QuestionType filter
qt_df = sub_df[sub_df['Subcategory'] == selected_subcategory]
question_types = qt_df['QuestionType'].unique()
selected_qtype = st.sidebar.selectbox('Select QuestionType', question_types)

# Apply all three filters
filtered_data = product_data[
    (product_data['Category'] == selected_category) &
    (product_data['Subcategory'] == selected_subcategory) &
    (product_data['QuestionType'] == selected_qtype)
]

if filtered_data.empty:
    st.warning("No interview entries for that combination.")
    st.stop()

# Initialize navigation state
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0

def play_aloud(text: str):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")

def display_interview(idx: int):
    row = filtered_data.iloc[idx]
    interviewer = row['Interviewer']
    interviewee = row['Interviewee']

    st.markdown(f"#### Interviewer:\n> {interviewer}")
    st.markdown(f"#### Interviewee:\n> {interviewee}")

    if st.button("🔊 Read Aloud", key=f"read_{idx}"):
        play_aloud(f"Interviewer says: {interviewer}. Interviewee replies: {interviewee}")
    return True

# Show current interview segment
if display_interview(st.session_state.question_index):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back"):
            if st.session_state.question_index > 0:
                st.session_state.question_index -= 1
    with col2:
        if st.button("Next ▶"):
            if st.session_state.question_index < len(filtered_data) - 1:
                st.session_state.question_index += 1
            else:
                st.success("End of Interview. Thank you!")
