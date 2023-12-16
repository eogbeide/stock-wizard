import streamlit as st
import PyPDF2
import re

# Streamlit app
def main():
    st.title("Top SAT Words by Manny")
    st.write("Choose a word to get its meaning.")

    try:
        # Get the raw URL of the PDF file on GitHub
        raw_pdf_url = "https://raw.githubusercontent.com/eogbeide/stock-wizard/main/SAT_Vocabulary_List.pdf"

        # Download the PDF file and read its contents
        pdf_content = st.binary_resource(raw_pdf_url)
        pdf_file = open_pdf(pdf_content)

        # Extract words and meanings from the PDF
        words_dict = extract_words_and_meanings(pdf_file)

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

    except PyPDF2.PdfReadError:
        st.error("Error: Unable to read the PDF file.")
    except urllib.error.URLError:
        st.error("Error: Unable to fetch the PDF file. Please check the URL or your internet connection.")

def open_pdf(pdf_content):
    # Open the PDF file with PyPDF2
    pdf_file = PyPDF2.PdfFileReader(pdf_content)
    return pdf_file

def extract_words_and_meanings(pdf_file):
    # Extract words and meanings from the PDF using regular expressions
    words_dict = {}
    for page_num in range(pdf_file.numPages):
        page = pdf_file.getPage(page_num)
        text = page.extractText()
        matches = re.findall(r"(?i)(\w+)\s-\s(.+?)(?=\s\w+\s-|$)", text)
        for match in matches:
            word = match[0]
            meaning = match[1]
            words_dict[word] = meaning.strip()
    return words_dict

if __name__ == '__main__':
    main()
