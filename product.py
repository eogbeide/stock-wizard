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

        # Rename columns so you can refer to them as QuestionType / Interview
        df.rename(
            columns={
                'Interviewer': 'QuestionType',
                'Interviewee': 'Interview'
            },
            inplace=True
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

product_data = load_data()

if product_data.empty:
    st.warning("No data available.")
    st.stop()

st.sidebar.title('Interview Navigation')

# Category selector
categories = product_data['Category'].unique()
selected_category = st.sidebar.selectbox('Select Category', categories)

# Subcategory selector
sub_df = product_data[product_data['Category'] == selected_category]
subcategories = sub_df['Subcategory'].unique()
selected_subcategory = st.sidebar.selectbox('Select Subcategory', subcategories)

# Filter down to matching rows
filtered_data = product_data[
    (product_data['Category'] == selected_category) &
    (product_data['Subcategory'] == selected_subcategory)
]

if filtered_data.empty:
    st.warning("No entries for that Category/Subcategory.")
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
    if idx >= len(filtered_data):
        st.write("End of entries.")
        return False

    row = filtered_data.iloc[idx]
    qtype = row['QuestionType']
    interview = row['Interview']

    st.markdown(f"### QuestionType:\n> {qtype}")
    st.markdown(f"### Interview:\n> {interview}")

    if st.button("🔊 Read Aloud", key=f"read_{idx}"):
        play_aloud(f"Question type: {qtype}. Interview: {interview}")
    return True

# Show the current entry
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
