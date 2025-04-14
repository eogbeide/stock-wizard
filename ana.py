import streamlit as st
import requests
from llama_index import GPTSimpleVectorIndex

# Function to fetch and read PDF from a GitHub URL
def fetch_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        return "temp.pdf"
    else:
        st.error(f"Failed to fetch PDF from {url}")
        return None

# Load PDFs from the GitHub repository
def load_documents(pdf_filenames):
    documents = []
    base_url = 'https://raw.githubusercontent.com/eogbeide/stock-wizard/main/'
    for filename in pdf_filenames:
        pdf_url = f"{base_url}{filename.strip()}"
        pdf_path = fetch_pdf(pdf_url)
        if pdf_path:
            # Assuming a different method to read PDF content
            with open(pdf_path, "rb") as f:
                content = f.read()  # Read PDF content
                documents.append(content)  # Append raw content or processed text
    return documents

# Streamlit app
def main():
    st.title("PDF Q&A with LlamaIndex")

    # Input for PDF filenames
    pdf_filenames = st.text_area("Enter PDF filenames (comma-separated):", "")
    pdf_filenames = [filename.strip() for filename in pdf_filenames.split(",") if filename.strip()]

    if st.button("Load PDFs"):
        if pdf_filenames:
            documents = load_documents(pdf_filenames)
            index = GPTSimpleVectorIndex(documents)
            st.session_state.index = index  # Save index in session state
            st.success("Documents loaded successfully!")
        else:
            st.error("Please enter valid PDF filenames.")

    question = st.text_input("Ask a question about the PDFs:")

    if st.button("Get Answer"):
        if 'index' in st.session_state:
            answer = st.session_state.index.query(question)
            st.write(answer)
        else:
            st.error("Please load PDFs first.")

if __name__ == "__main__":
    main()
