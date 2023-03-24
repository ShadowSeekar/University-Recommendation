import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

st.set_page_config(page_title="University Recommendation System")

import joblib

# Load the saved model
model = joblib.load('model.pkl')

# Define the form inputs
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
gre = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)

# Make predictions
if st.button("Submit"):
    data = {'CGPA': [cgpa], 'GRE Score': [gre], 'TOEFL Score': [toefl]}
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    if prediction[0] == 1:
        st.write("You have a good chance of getting admission to the university!")
    else:
        st.write("Sorry, your chances of getting admission to the university are low.")
