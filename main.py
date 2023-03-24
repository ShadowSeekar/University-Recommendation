import streamlit as st

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def app():
    st.title('University Admission Predictor')
    st.write('Enter your details to get admission prediction')

# Define the input form
form = st.form(key='input_form')
cgpa = form.number_input('CGPA', min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
gre = form.number_input('GRE Score', min_value=0, max_value=340, step=1)
toefl = form.number_input('TOEFL Score', min_value=0, max_value=120, step=1)
submit_button = form.form_submit_button('Submit')

# Get the prediction result
if submit_button:
    result = model.predict([[cgpa, gre, toefl]])
    if result == 1:
        st.write('Congratulations! You are likely to get admission.')
    else:
        st.write('Sorry, you are not likely to get admission.')

if name == 'main':
    app()