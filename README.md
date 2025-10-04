# 🏏 IPL Win Predictor Web App – Machine Learning Project

I built this IPL Win Predictor web application to apply machine learning techniques for predicting match outcomes in the Indian Premier League. The project demonstrates the full pipeline from data preprocessing and model training to web app deployment using **Streamlit**.

---

## 📁 Files

- `app.py` – Streamlit web application file.
- `ipl_dataset.csv` – Dataset containing IPL match data.
- `model.ipynb` – Jupyter notebook with model training and evaluation.
- `training_columns_shaunak.joblib` – Serialized feature columns for model alignment.
- `forest_classifier_shaunak.joblib` – Serialized Random Forest model.
- `.gitattributes` / `README.md` – Repository configuration and documentation.

---

## 📌 Project Objectives

- Load and explore IPL match data.
- Clean and preprocess features for machine learning.
- Train multiple regression and classification models to predict match outcomes.
- Evaluate models and select the best performing one (Random Forest).
- Serialize the trained model for reuse in a web app.
- Build an interactive Streamlit frontend for user input and dynamic predictions.

---

## 🔧 Technologies and Libraries Used

- **Python** – Core programming language.
- **Pandas / NumPy** – Data manipulation and numerical operations.
- **Scikit-learn** – Model training and evaluation.
- **Joblib** – Model serialization.
- **Seaborn / Matplotlib** – Data visualization.
- **Streamlit** – Web app framework for deploying interactive ML apps.

---

## 🔍 Workflow Overview

### 1. 📥 Data Loading and Preprocessing
- Imported IPL match data in Google Colab.
- Removed irrelevant columns and filtered consistent teams.
- Dropped early overs (<5) to focus on meaningful match scenarios.
- Engineered features like `runs_left`, `balls_left`, `wickets_remaining`, `crr`, `rrr`.
- Created a binary target variable `is_win` for classification.

### 2. 🤖 Model Training
- Trained multiple models: Decision Tree, Linear Regression, Lasso, SVM, Neural Network, Random Forest.
- Evaluated models using Accuracy, Precision, Recall, and F1-score.
- Random Forest classifier performed the best.
- Serialized the trained Random Forest model and feature columns using **Joblib**.

### 3. 🧹 Feature Engineering
- Encoded categorical features: `batting_team`, `bowling_team`, `venue`.
- Combined numerical and encoded categorical features for training.
- Saved training columns to ensure correct input alignment in the web app.

### 4. 🚀 Web App Development
- Developed the frontend in VSCode using **Streamlit**.
- Created interactive input fields for:
  - Batting and bowling teams
  - Venue
  - Target score, current score, overs completed, wickets lost
- Calculated win probability dynamically using the trained Random Forest model.

### 5. 📈 Prediction
- Prepared user input and aligned with training columns.
- Predicted win probabilities for batting and bowling teams.
- Displayed results interactively on the Streamlit interface.

---

## 🎯 Outcome

- Achieved a high-performing Random Forest classifier for IPL win prediction.
- Enabled dynamic match outcome prediction in a user-friendly web app.
- Reinforced understanding of feature engineering, model training, serialization, and ML deployment.

---

## 💡 Use Cases

- Predicting IPL match outcomes in real-time.
- Data-driven insights for sports analytics.
- Interactive ML web applications showcasing model deployment.

---

## 🚀 How to Run

1. Clone this repository or download the files.
2. Install required Python packages:
   ```bash
   pip install pandas numpy scikit-learn==1.7.1 seaborn matplotlib streamlit joblib
   
## Run the Streamlit app:
bash
streamlit run app.py
Select teams, venue, target, score, overs, and wickets to get dynamic win probabilities.

# (https://ipl-winning-team-predicition-app-fpstegzxppwp3g3fdhypjh.streamlit.app/)

## 📊 Key Learning Outcomes
Hands-on experience with feature engineering for sports data.

Training and evaluating multiple machine learning models.

Model serialization using Joblib for deployment.

Streamlit app development for real-time predictions and interactive user input.
