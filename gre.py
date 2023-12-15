import streamlit as st
import pandas as pd
import urllib
from PyDictionary import PyDictionary
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Streamlit app
def main():
    st.title("Top GRE Words by Engr. Manny")
    st.write("Choose a word to get its meaning and example usage.")

    try:
        # Get the raw URL of the CSV file on GitHub
        raw_csv_url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/top_500_gre_words.csv"

        # Download the CSV file and read its contents
        csv_content = urllib.request.urlopen(raw_csv_url)
        words_df = pd.read_csv(csv_content, encoding="cp1252")

        # Create a dictionary mapping words to meanings and example sentences
        words_dict = {}
        for _, row in words_df.iterrows():
            word = row['Word']
            meaning = row['Meaning']
            example_sentence = row['Example Sentence']
            words_dict[word] = {'meaning': meaning, 'example_sentence': example_sentence}

        # Select a word from the dropdown
        selected_word = st.selectbox("Select a word", list(words_dict.keys()))

        if selected_word:
            # Display the meaning of the selected word
            st.write(f"Meaning: {words_dict[selected_word]['meaning']}")

            # Display example sentences
            example_sentence = words_dict[selected_word]['example_sentence']
            if example_sentence:
                sentences = sent_tokenize(example_sentence)
                st.write("Example Sentences:")
                for sentence in sentences:
                    st.write(sentence)
            else:
                st.write("Example sentences not available.")

        else:
            st.write("Please select a word.")

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please check the file's encoding.")

if __name__ == '__main__':
    main()
