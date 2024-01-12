import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd

# Function to scrape prerequisites for a specific medical school
def scrape_prerequisites(school):
    # URL of the AAMC website page containing medical school prerequisites
    url = "https://www.aamc.org/admissions/medical-school-admissions/prerequisites"

    # Send a GET request to the website
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the specific HTML elements containing the prerequisites information
    prerequisites_elements = soup.find_all("div", class_="prerequisites")

    # Create a list to store the prerequisite data
    prerequisites_data = []

    # Iterate over the prerequisites elements and extract the requirements for the selected school
    for element in prerequisites_elements:
        school_name = element.find("h2").text.strip()
        if school_name.lower() == school.lower():
            state = element.find("h3").text.strip()
            course = element.find("strong").text.strip()
            requirement = element.find_all("p")
            required_or_recommended = requirement[0].text.strip()
            lab = requirement[1].text.strip()
            additional_info = requirement[2].text.strip()

            prerequisites_data.append({
                "State": state,
                "Medical School": school_name,
                "Course": course,
                "Required or Recommended": required_or_recommended,
                "Lab?": lab,
                "Additional Info": additional_info
            })

    return prerequisites_data

# Streamlit app
def main():
    # Title and description
    st.title("Medical School Prerequisites")
    st.write("Select a medical school to view its prerequisites.")

    # List of medical schools (example)
    schools = ["School A", "School B", "School C"]

    # Selectbox to choose a medical school
    selected_school = st.selectbox("Select a school", schools)

    if st.button("Show Prerequisites"):
        # Scrape prerequisites for the selected school
        prerequisites = scrape_prerequisites(selected_school)
        
        # Display prerequisites
        if prerequisites:
            st.subheader(f"Prerequisites for {selected_school}")
            df = pd.DataFrame(prerequisites)
            st.write(df)
        else:
            st.write("No prerequisites found for the selected school.")

if __name__ == "__main__":
    main()
