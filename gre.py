import streamlit as st
import requests
from bs4 import BeautifulSoup

# Function to scrape the top 500 GRE words from the website
def scrape_gre_words():
    url = "https://www.majortests.com/gre/wordlist.php"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    word_table = soup.find("table", {"class": "wordlist"})
    if word_table is None:
        return []

    rows = word_table.find_all("tr")[1:]  # Skip the table header row

    words = []
    for row in rows:
        cells = row.find_all("td")
        word = cells[1].text.strip()
        meaning = cells[2].text.strip()
        example = cells[3].text.strip()
        words.append({"Word": word, "Meaning": meaning, "Example": example})

    return words

# Streamlit app
def main():
    st.title("Top 500 GRE Words")
    st.write("Choose a word to get its meaning and an example sentence.")

    # Scrape the top 500 GRE words
    words = scrape_gre_words()

    # Create a dictionary mapping words to meanings and example sentences
    words_dict = {word["Word"]: {"Meaning": word["Meaning"], "Example": word["Example"]} for word in words}

    # Select a word from the dropdown
    selected_word = st.selectbox("Select a word", list(words_dict.keys()))
    
    # Check if the selected word exists in the dictionary
    if selected_word in words_dict:
        # Display the meaning and example sentence
        st.write(f"Meaning: {words_dict[selected_word]['Meaning']}")
        st.write(f"Example: {words_dict[selected_word]['Example']}")
    else:
        # Word not found in the dictionary
        st.write("Word not found.")

    # Display the meaning and example sentence
    st.write(f"Meaning: {words_dict[selected_word]['Meaning']}")
    st.write(f"Example: {words_dict[selected_word]['Example']}")

if __name__ == '__main__':
    main()
