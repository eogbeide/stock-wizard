import streamlit as st
import pandas as pd
import urllib

# Streamlit app
def main():
    st.title("Top GRE Words by Manny")
    st.write("Choose a word to get its meaning.")

    try:
        # Get the raw URL of the CSV file on GitHub
        raw_csv_url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/top_500_gre_words.csv"

        # Download the CSV file and read its contents
        csv_content = urllib.request.urlopen(raw_csv_url)
        words_df = pd.read_csv(csv_content, encoding="cp1252")

        # Create a dictionary mapping words to meanings
        words_dict = {
            row['Word']: row['Meaning']
            for _, row in words_df.iterrows()
        }

        # Create a sidebar for selecting alphabets
        alphabets = sorted(set(word[0] for word in words_dict.keys()))
        selected_alphabet = st.sidebar.selectbox("Select an alphabet", alphabets)

        # Filter the words based on the selected alphabet
        filtered_words = {word: meaning for word, meaning in words_dict.items() if word.startswith(selected_alphabet)}

        if filtered_words:
            # Select a word from the filtered words
            selected_word = st.selectbox("Select a word", list(filtered_words.keys()))

            if selected_word:
                # Display the meaning of the selected word
                st.write(f"Meaning: {filtered_words[selected_word]}")
            else:
                st.write("Please select a word.")
        else:
            st.write("No words found for the selected alphabet.")

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please check the file's encoding.")

if __name__ == '__main__':
    main()
