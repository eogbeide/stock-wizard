import streamlit as st
import pandas as pd

# Load the top 500 GRE words and their meanings from a CSV file
words_df = pd.read_csv('top_500_gre_words.csv', encoding='utf-8')

# Create a dictionary mapping words to meanings and example sentences
words_dict = {
    row['Word']: {'Meaning': row['Meaning'], 'Example': row['Example']}
    for _, row in words_df.iterrows()
}

# Streamlit app
def main():
    st.title("Top 500 GRE Words")
    st.write("Choose a word to get its meaning and an example sentence.")

    # Select a word from the dropdown
    selected_word = st.selectbox("Select a word", list(words_dict.keys()))

    # Display the meaning and example sentence
    st.write(f"Meaning: {words_dict[selected_word]['Meaning']}")
    st.write(f"Example: {words_dict[selected_word]['Example']}")

if __name__ == '__main__':
    main()
