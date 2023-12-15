pip install nltk
import streamlit as st
import pandas as pd
import urllib3
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# Streamlit app
def main():
    st.title("Top GRE Words by Engr. Manny")
    st.write("Choose a word to get its meaning.")

    try:
        # Get the raw URL of the CSV file on GitHub
        raw_csv_url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/top_500_gre_words.csv"

        # Download the CSV file and read its contents
        csv_content = urllib.request.urlopen(raw_csv_url)
        words_df = pd.read_csv(csv_content, encoding="cp1252'")

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

            # Get example sentences from WordNet
            synsets = wordnet.synsets(selected_word)
            example_sentences = []
            for synset in synsets:
                for example in synset.examples():
                    example_sentences.append(example)

            if example_sentences:
                st.write("Example Sentences:")
                for sentence in example_sentences[:5]:
                    st.write(sentence)
            else:
                st.write("Example sentences not available.")

        else:
            st.write("Please select a word.")

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please check the file's encoding.")

if __name__ == '__main__':
    main()
