import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="University Recommendation System")

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the form inputs
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
gre = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)

# Make predictions
if st.button("Submit"):
    data = pd.DataFrame({'CGPA': [cgpa], 'GRE Score': [gre], 'TOEFL Score': [toefl]})
    prediction = model.predict([[cgpa, gre, toefl]])
    filtered_data = data[(data['CGPA'] <= cgpa) & (data['GRE Score'] <= gre) & (data['TOEFL Score'] <= toefl)]
    #print(f"Values less than {cgpa and gre and toefl}: \n{filtered_data}")
    st.write(filtered_data)
