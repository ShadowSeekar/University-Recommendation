import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Load the trained KNN model from disk
with open('kNNr_model2.pkl', 'rb') as file:
    knn = pickle.load(file)

# Load the dataset used for training the model
df_train = pd.read_csv('student_data.csv')
df_train = df_train.drop(['SOP', 'LOR'], axis=1)

# Separate the features and target variable
X_train = df_train[['GRE Score', 'TOEFL Score', 'CGPA', 'Research']]
y_train = df_train['Chance of Admit']

# Fit a StandardScaler instance to the training data and use it to scale the user input
scaler = StandardScaler()
scaler.fit(X_train)

# Define a function to get the top 5 institutions with predicted chances of admission that are lower than the user input
def get_top_institutions(user_input):
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    predicted_chance_of_admit = knn.predict(user_input)[0]
    df_train['Predicted Chance of Admit'] = knn.predict(scaler.transform(X_train))
    df_grouped = df_train.groupby('Institution').mean().sort_values(by='Predicted Chance of Admit', ascending=False)
    top_institutions = df_grouped[df_grouped['Predicted Chance of Admit'] < predicted_chance_of_admit].head(5)
    return top_institutions[['Predicted Chance of Admit']]

# Create the web app using Streamlit
st.title('Admission Predictor')

st.write('Enter the values of the following features:')
gre_score = st.slider('GRE Score', 260, 340, 320)
toefl_score = st.slider('TOEFL Score', 80, 120, 110)
#sop = st.slider('SOP', 1.0, 5.0, 3.0)
#lor = st.slider('LOR', 1.0, 5.0, 3.0)
cgpa = st.slider('CGPA', 6.0, 10.0, 8.0)
research = st.radio('Research', ['Yes', 'No'])
if research == 'Yes':
    research = 1
else:
    research = 0


# Get the top 5 institutions with predicted chances of admission that are lower than the user input
user_input = [gre_score, toefl_score, sop, lor, cgpa, research]
top_institutions = get_top_institutions(user_input)

if top_institutions.empty:
    st.write('No institutions found with your chances of admission in database.')
else:
    st.write('Institutions with predicted chances of admission:')
    st.table(top_institutions)

