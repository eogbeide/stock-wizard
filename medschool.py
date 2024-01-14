import streamlit as st
import pandas as pd

# Welcome Page
st.title("Welcome to US Medical Schools Prerequisite AI Wiz")
st.write(" - This Prerequiste Wizard is based on AAMC data obatined from https://students-residents.aamc.org/media/7041/download")
st.write(" - Always cross-validate with the school's website")

# First page: Medical schools with filter options
st.header("Selected Medical School Prerequisites")
st.write(" - Double click on Additional Info Column to see details")

# Load the CSV file
df = pd.read_csv("Medical_School_Requirements3.csv", encoding='unicode_escape')

# Remove leading and trailing spaces from all string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Create a list of distinct medical school names
school_options = df['Medical School'].unique().tolist()

# Create a selectbox to choose a state
selected_state = st.sidebar.selectbox("Select State", df['State'].unique())

# Filter the DataFrame based on the selected state
state_filtered_df = df[df['State'] == selected_state]

# Create a selectbox to choose a medical school within the selected state
selected_school = st.sidebar.selectbox("Select Medical School", state_filtered_df['Medical School'].unique())


# Display the selected medical school and state
st.write(f"Selected Medical School: {selected_school}")
st.write(f"Selected State: {selected_state}")

# Search for the selected school on Wikipedia
search_query = "List of medical schools in the United States to view website for"
search_query += " " + selected_school.replace(" ", "_")

# Create the Wikipedia link
#school_wikipedia_link = "https://members.aamc.org/eweb/DynamicPage.aspx?site=AAMC&webcode=AAMCOrgSearchResult&orgtype=Medical%20School"

# Display the link to the medical school's Wikipedia page
#st.write(f"Click: {school_wikipedia_link} for more details about {search_query}")
exact_school = st.write(f"{search_query}")
st.link_button(f"Go to AAMC website for {search_query}", "https://members.aamc.org/eweb/DynamicPage.aspx?site=AAMC&webcode=AAMCOrgSearchResult&orgtype=Medical%20School")


# Filter the DataFrame based on the selected school and exclude State and Medical School columns
filtered_df = state_filtered_df[state_filtered_df['Medical School'] == selected_school].drop(columns=['State', 'Medical School'])

# Check if 'Credit Hours' column exists in the filtered DataFrame
if 'Credit Hours' in filtered_df.columns:
    # Convert 'Credit Hours' column to numeric values
    filtered_df['Credit Hours'] = pd.to_numeric(filtered_df['Credit Hours'], errors='coerce')

    # Format the 'Credit Hours' column to display one significant figure
    filtered_df['Credit Hours'] = filtered_df['Credit Hours'].apply(lambda x: format(x, ".1f") if pd.notnull(x) else "")

    # Display the filtered DataFrame without the index column, with wrapped text in the 'Additional Info' column
    st.dataframe(filtered_df.reset_index(drop=True).style.set_properties(**{'white-space': 'pre-wrap'}))
else:
    st.write("No 'Credit Hours' column found in the filtered DataFrame.")

# Second page: Medical schools with filter options
st.subheader("Filtered Medical Schools by Required or Recommended Courses")

# Define the filter options
course_options = df['Course'].unique().tolist()
required_or_recommended_options = ['Required', 'Recommended']

# Select the filter options from the sidebar
selected_course = st.sidebar.selectbox("Choose Course", course_options)
selected_required_or_recommended = st.sidebar.selectbox("Required or Recommended?", required_or_recommended_options)

# Filter the DataFrame based on the selected options
filtered_schools_df = df[(df['Course'] == selected_course) & (df['Required or Recommended?'] == selected_required_or_recommended)]

# Select the columns to display
columns_to_display = ['Medical School', 'Lab?', 'Credit Hours', 'Additional Info']

# Display the filtered DataFrame with the selected columns, sorted by "Medical School"
st.dataframe(filtered_schools_df[columns_to_display].sort_values(by="Medical School").reset_index(drop=True))
