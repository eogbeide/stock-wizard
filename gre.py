import streamlit as st
import pandas as pd

# Streamlit app
def main():
    st.title("Top 500 GRE Words")
    st.write("Choose a word to get its meaning.")

    try:
        # Open the CSV file and read its contents
        with open('top_500_gre_words.csv', 'r', encoding='utf-16-le', errors='ignore') as file:
            words_df = pd.read_csv(file)

        # Create a dictionary mapping words to meanings
        words_dict = {
            row['Word']: row['Meaning']
            for _, row in words_df.iterrows()
        }

        # Select a word from the dropdown
        selected_word = st.selectbox("Select a word", list(words_dict.keys()))

        if selected_word:
            # Display the meaning of the selected word
            st.write(f"Meaning: {words_dict[selected_word]}")
        else:
            st.write("Please select a word.")

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please check the file's encoding.")

if __name__ == '__main__':
    main()
