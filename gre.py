import streamlit as st
import pandas as pd
import urllib
from PyDictionary import PyDictionary

dictionary = PyDictionary()

# Streamlit app
def main():
    st.title("Top GRE Words by Engr. Manny")
    st.write("Choose a word to get its meaning and an example usage.")

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

            # Get example usage from online dictionary
            dictionary = PyDictionary()
            example_usage = dictionary.meaning(selected_word)

            if example_usage:
                st.write(f"Example Usage: {example_usage[selected_word][0]}")
            else:
                st.write("Example usage not available.")

        else:
            st.write("Please select a word.")

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please check the file's encoding.")

if __name__ == '__main__':
    main()
