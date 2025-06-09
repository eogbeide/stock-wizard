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
        # Rename for clarity
        df.rename(columns={'Interviewer': 'QuestionType',
                           'Interviewee': 'Interview'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

product_data = load_data()
if product_data.empty:
    st.warning("No data available.")
    st.stop()

st.sidebar.title('Interview Navigation')

# 1) Category
categories = product_data['Category'].unique()
selected_category = st.sidebar.selectbox('Select Category', categories)

# 2) Subcategory
sub_df = product_data[product_data['Category'] == selected_category]
subcategories = sub_df['Subcategory'].unique()
selected_subcategory = st.sidebar.selectbox('Select Subcategory', subcategories)

# 3) QuestionType
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
    st.warning("No entries for that combination.")
    st.stop()

# Navigation state
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0

def play_aloud(text: str):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(tmp.name, format="audio/mp3")

def display_entry(idx: int):
    row = filtered_data.iloc[idx]
    st.markdown(f"### QuestionType:\n> {row['QuestionType']}")
    st.markdown(f"### Interview:\n> {row['Interview']}")
    if st.button("🔊 Read Aloud", key=f"read_{idx}"):
        play_aloud(f"Question type: {row['QuestionType']}. Interview: {row['Interview']}")
    return True

# Show current entry and nav buttons
if display_entry(st.session_state.question_index):
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
