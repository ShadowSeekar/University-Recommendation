# 🎓 University Recommendation System

A Machine Learning-powered web app that predicts your chance of admission to universities based on academic scores, and recommends institutions you’re most likely to get into.  
🚀 Built with Scikit-Learn, K-NN Regression, and deployed using Streamlit.  

🔗 Live Demo: https://shadowseekar-university-recommendation-recommender-jgvaut.streamlit.app/

# 🔍 Features
✔ Hyperparameter tuning using GridSearchCV
✔ Model validation with Cross-Validation
✔ User input prediction via the command line
✔ Deployed Streamlit web app for interactive use
✔ CSV download of recommended universities

# 📂 Dataset
The model is trained on a dataset with features similar to:
- Feature	Description
- GRE Score	Standardized test score (max 340)
- TOEFL Score	Standardized English test score (max 120)
- CGPA	Undergraduate grade
- Chance of Admit	Target variable

# 🛠 Model Details
Algorithm -	K-Nearest Neighbors Regressor
Scaling	- StandardScaler
Validation	- 10-Fold Cross-Validation
Best Score	~92% (varies)
Final R2 Score	~0.8 on training data

# 🚀 Usage
A) Run Locally
Clone the repo:
```
git clone https://github.com/<your-username>/university-recommendation-system.git

```

Install dependencies:
```
pip install -r requirements.txt
```
Train and test your model:
```
python university_model.py
```

B) Run the Streamlit App
Ensure that the saved model file (random_forest_model.pkl or kNNr_model.pkl) and the dataset (data_uni.csv) are located in the project folder.
```
streamlit run app.py
```
📥 You can also download the recommended universities as a CSV!

# 🗂 Project Structure
```
├── README.md
├── requirements.txt
├── university_model.py        # Model training + CLI prediction
├── app.py                     # Streamlit web app
├── student_data.csv           # Training data for model
├── dataset.csv                # Dataset for inference
├── data_uni.csv               # Univ ranking & score data
├── kNNr_model.pkl             # Saved KNN Regressor model
└── random_forest_model.pkl    # Alternative model used in Streamlit

```
