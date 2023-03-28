import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
filename = 'random_forest_rgr_model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Define the function to make predictions
def predict_admission_chance(data):
    prediction = model.predict(data)[0]
    # Convert the prediction to percentage and round it to two decimal places
    prediction_percent = round(prediction * 100, 2)
    return prediction_percent

# Define the Streamlit app
def app():
    # Set the app title
    st.title('Admission Chance Prediction App')
    
    # Create the input form for the user
    form = st.form(key='admission_chance_form')
    gre_score = form.number_input('GRE Score', min_value=250, max_value=340, step=1)
    toefl_score = form.number_input('TOEFL Score', min_value=70, max_value=120, step=1)
    university_rating = form.slider('University Rating', min_value=1, max_value=5, step=1, value=3)
    sop = form.slider('Statement of Purpose Strength (out of 5)', min_value=0.0, max_value=5.0, step=0.1, value=3.0)
    lor = form.slider('Letter of Recommendation Strength (out of 5)', min_value=0.0, max_value=5.0, step=0.1, value=3.0)
    cgpa = form.number_input('CGPA', min_value=5.0, max_value=10.0, step=0.01)
    research = form.selectbox('Research Experience', ['No', 'Yes'])
    submit_button = form.form_submit_button(label='Predict')
    
    # Check if the user has clicked the Predict button
    if submit_button:
        # Preprocess the user input
        research = 1 if research == 'Yes' else 0
        data = [[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]]
        data = pd.DataFrame(data, columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
        
        # Make a prediction and display it to the user
        prediction = predict_admission_chance(data)
        st.success(f'Your chance of admission is: {prediction}%')

# Run the Streamlit app
if __name__ == '__main__':
    app()