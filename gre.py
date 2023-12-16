import streamlit as st
import pandas as pd
import urllib
import re

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

        # Clean the words and meanings
        words_df['Word'] = words_df['Word'].apply(clean_text)
        words_df['Meaning'] = words_df['Meaning'].apply(clean_text)

        # Create a dictionary mapping words to meanings
        words_dict = {
            row['Word']: row['Meaning']
            for _, row in words_df.iterrows()
        }

        # Group words by alphabet bins
        alphabet_bins = {}
        for word in words_dict.keys():
            letter = word[0].upper()
            if letter in alphabet_bins:
                alphabet_bins[letter].append(word)
            else:
                alphabet_bins[letter] = [word]

        # Select an alphabet bin from the dropdown
        selected_bin = st.selectbox("Select an alphabet bin", sorted(alphabet_bins.keys()))

        # Filter the words based on the selected alphabet bin
        filtered_words = {word: words_dict[word] for word in alphabet_bins[selected_bin]}

        if filtered_words:
            # Select a word from the filtered words
            selected_word = st.selectbox("Select a word", list(filtered_words.keys()))

            if selected_word:
                # Display the meaning of the selected word
                st.write(f"Meaning: {filtered_words[selected_word]}")
            else:
                st.write("Please select a word.")
        else:
            st.write("No words found for the selected alphabet bin.")

    except UnicodeDecodeError:
        st.error("Error: Unable to decode the CSV file. Please check the file's encoding.")

def clean_text(text):
    # Remove HTML tags and entities from the text
    cleaned_text = re.sub('<[^<]+?>', '', text)
    cleaned_text = re.sub('&\w+;', '', cleaned_text)
    cleaned_text = cleaned_text.replace('", [', '')
    return cleaned_text.strip()

if __name__ == '__main__':
    main()
