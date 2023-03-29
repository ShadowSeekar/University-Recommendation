import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="University Recommendation System")
st.markdown("# University Recommendation System")
st.sidebar.markdown("# Recommendation System")

# Load the saved model
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# Define the form inputs
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, format="%.2f")
gre = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)

# Make predictions
if st.button("Submit"):
    data = pd.read_csv('data_uni.csv')
    prediction = loaded_model.predict([[cgpa, gre, toefl]])
    filtered_data = data[(data['CGPA'] <= cgpa) & (data['GRE Score'] <= gre) & (data['TOEFL Score'] <= toefl)]
    dfs = pd.DataFrame(filtered_data, columns=['world_rank', 'institution', 'country', 'quality_of_education',
                                               'quality_of_faculty', 'influence', 'patents', 'GRE Score', 'TOEFL Score', 'CGPA'])
    dfs.rename(columns={'world_rank': 'World Rank', 'institution': 'Institution', 'country': 'Country',
                         'quality_of_education': 'Quality of Education', 'quality_of_faculty': 'Quality of Faculty',
                         'influence': 'Influence', 'patents': 'Patents'}, inplace=True)
    dfi = dfs.reset_index(drop=True)
    dfi.index = dfi.index + 1
    st.dataframe(dfi)

    csv = dfi.to_csv().encode('utf-8')

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Universities.csv',
        mime='text/csv',
    )
