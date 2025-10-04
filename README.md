# 🏏 IPL Win Predictor Web App – Machine Learning Project

I built and deployed an interactive IPL Win Predictor web application to predict match outcomes in the Indian Premier League. This project demonstrates the complete workflow from data preprocessing and machine learning model training to deployment using **Streamlit**, with a live app accessible online.

---

## 🌐 Live App

The app is deployed and accessible at:  
[https://ipl-winning-team-predicition-app-fpstegzxppwp3g3fdhypjh.streamlit.app/](https://ipl-winning-team-predicition-app-fpstegzxppwp3g3fdhypjh.streamlit.app/)

---

## 📁 Repository Files

- `app.py` – Streamlit web application file.
- `ipl_dataset.csv` – IPL match dataset used for model training.
- `model.ipynb` – Jupyter notebook with preprocessing, feature engineering, and model training.
- `training_columns_shaunak.joblib` – Serialized feature columns for input alignment.
- `forest_classifier_shaunak.joblib` – Serialized Random Forest classifier.
- `.gitattributes` / `README.md` – Repository configuration and documentation.

GitHub Repository: [Shaunak-Kunde / IPL Win Predictor](https://github.com/Shaunak-Kunde/IPL-Win-Predictor)

---

## 📌 Project Objectives

- Load, clean, and preprocess IPL match data.
- Engineer meaningful features for predicting match outcomes.
- Train multiple machine learning models and evaluate their performance.
- Select the best performing model (Random Forest) and serialize it.
- Develop an interactive web app with Streamlit to predict dynamic win probabilities.

---

## 🔧 Technologies and Libraries Used

- **Python** – Core programming language.
- **Pandas / NumPy** – Data handling and numerical computations.
- **Scikit-learn** – Model training, evaluation, and serialization.
- **Joblib** – Saving and loading ML models and training columns.
- **Seaborn / Matplotlib** – Data visualization.
- **Streamlit** – Web app development and deployment.

---

## 🔍 Workflow Overview

### 1. 📥 Data Loading and Cleaning
- Imported IPL match data and removed irrelevant columns.
- Filtered for consistent teams and removed early overs (<5).
- Created a binary target variable `is_win` for classification tasks.

### 2. 🧹 Feature Engineering
- Calculated features like `runs_left`, `balls_left`, `wickets_remaining`, `crr`, and `rrr`.
- Encoded categorical variables such as `batting_team`, `bowling_team`, and `venue`.
- Combined numerical and categorical features for model training.

### 3. 🤖 Model Training and Evaluation
- Trained multiple models: Decision Tree, Linear Regression, Lasso, SVM, Neural Network, Random Forest.
- Evaluated models using Accuracy, Precision, Recall, and F1-score.
- Random Forest classifier emerged as the best performing model.
- Serialized the model and training columns using **Joblib** for app deployment.

### 4. 🚀 Web App Development
- Built the app with **Streamlit** to allow users to select:
  - Batting and bowling teams
  - Venue
  - Target score, current score, overs completed, wickets lost
- Calculated dynamic win probabilities for user inputs using the trained Random Forest model.
- Deployed the app online for public access.

---

## 🎯 Key Outcomes

- Developed a **highly accurate Random Forest classifier** for IPL win prediction.
- Built a **live, interactive Streamlit web application** accessible globally.
- Reinforced practical skills in **feature engineering, model training, serialization, and ML deployment**.
- Learned to integrate ML models with a user-friendly web interface.

---

## 💡 Use Cases

- Predicting IPL match outcomes in real-time.
- Sports analytics and data-driven strategy planning.
- Showcasing machine learning deployment in interactive web apps.

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Shaunak-Kunde/IPL-Win-Predictor.git
Navigate to the project directory:

bash
Copy code
cd IPL-Win-Predictor
Install dependencies:

bash
Copy code
pip install pandas numpy scikit-learn==1.7.1 seaborn matplotlib streamlit joblib
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Use the interactive frontend to select teams, venue, target, score, overs, and wickets for dynamic win predictions.

## 📊 Learning Outcomes

# Hands-on experience with data cleaning and feature engineering for sports data.

# Training and evaluating multiple machine learning models for classification.

# Model serialization and deployment using Joblib and Streamlit.

# Creating interactive, real-time prediction web applications.
