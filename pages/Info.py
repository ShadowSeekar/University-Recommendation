import streamlit as st

st.markdown("# How is this different from the previous one? ❄️")
st.sidebar.markdown("# Working and Explaination ❄️")
st.markdown('#### It uses Random Forest machine learning algorithm to predict results.')
st.write("Random forest is a type of machine learning algorithm used for classification problems, where we want to predict a categorical label for each data point based on some input features.")
st.write("In this code, the input features are the CGPA, GRE Score, and TOEFL Score of students, and the goal is to predict whether each student will get admission to a university or not. The admission decision is based on whether the student's CGPA, GRE Score, and TOEFL Score are all below certain thresholds.")
st.write("The code uses a dataset containing these features and admission labels, and splits it into a training set and a testing set. It then trains a random forest classifier using the training set, where the classifier is a model that learns to map the input features to the corresponding admission label. The random forest classifier works by constructing a large number of decision trees, each of which makes a prediction based on a subset of the input features.")
st.write("The final prediction of the random forest is then determined by aggregating the predictions of all the individual decision trees. By training this random forest classifier on the labeled data, the code aims to learn a model that can accurately predict the admission decision for new, unseen data based on their CGPA, GRE Score, and TOEFL Score.")