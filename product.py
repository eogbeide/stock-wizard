import streamlit as sts
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
        df.rename(columns={
            'Interviewer': 'QuestionType',
            'Interviewee': 'Interview'
        }, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

product_data = load_data()
if product_data.empty:
    st.warning("No data available.")
    st.stop()

st.sidebar.title('Interview Navigation')

# 1) Category selector
categories = product_data['Category'].unique()
selected_category = st.sidebar.selectbox('Select Category', categories)

# 2) Subcategory selector
sub_df = product_data[product_data['Category'] == selected_category]
subcategories = sub_df['Subcategory'].unique()
selected_subcategory = st.sidebar.selectbox('Select Subcategory', subcategories)

# 3) QuestionType selector
qt_df = sub_df[sub_df['Subcategory'] == selected_subcategory]
question_types = qt_df['QuestionType'].unique()
selected_qtype = st.sidebar.selectbox('Select QuestionType', question_types)

# Apply all three filters
filtered_data = product_data[
    (product_data['Category'] == selected_category) &
    (product_data['Subcategory'] == selected_subcategory) &
    (product_data['QuestionType'] == selected_qtype)
]

# Clamp the session index to valid range
max_idx = len(filtered_data) - 1
if max_idx < 0:
    st.warning("No entries for that combination.")
    st.stop()

if 'question_index' not in st.session_state:
    st.session_state.question_index = 0
st.session_state.question_index = min(st.session_state.question_index, max_idx)

def play_aloud(text: str):
    """Generate TTS, play it, then delete the temp file."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        st.audio(tmp.name, format="audio/mp3")

def display_entry(idx: int):
    """Render one entry at the given index with proper paragraph formatting."""
    if idx < 0 or idx > max_idx:
        st.write("End of entries.")
        return False

    row = filtered_data.iloc[idx]
    # QuestionType as a subheader
    st.subheader("Question Type")
    # Render paragraphs: replace double newlines with <br> for markdown
    qt_text = row['QuestionType'].strip().replace("\n\n", "  \n\n")
    st.markdown(qt_text)

    st.subheader("Interview")
    int_text = row['Interview'].strip().replace("\n\n", "  \n\n")
    st.markdown(int_text)

    if st.button("🔊 Read Aloud", key=f"read_{idx}"):
        # Combine with proper sentence spacing
        play_aloud(f"Question type: {row['QuestionType']}. Interview: {row['Interview']}")
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
            if st.session_state.question_index < max_idx:
                st.session_state.question_index += 1
            else:
                st.success("End of Interview. Thank you!")
