# 📊 ChurnIQ — Customer Churn Prediction

An interactive machine-learning application that predicts customer churn risk using XGBoost and helps businesses identify customers who may leave their services.

## 🚀 Live Demo

https://customer-churn-prediction-biqkerg4mysctrkt6gajs8.streamlit.app/

Create an account or log in to explore the customer churn prediction dashboard.

## 📌 Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave can help businesses develop better customer-retention strategies.

ChurnIQ uses machine learning to estimate churn probability based on customer information and presents the results through an interactive Streamlit dashboard.

## ✨ Features

* 🔐 User registration, login, and logout
* 📊 Interactive customer churn prediction dashboard
* 🤖 XGBoost-based churn prediction
* 📈 Churn probability and risk classification
* 🟢 Low, 🟡 Medium, and 🔴 High risk categories
* 👤 Customer information input
* 📉 Model evaluation and feature-importance analysis

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* SQLite
* Joblib

## 🧠 Machine Learning

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

XGBoost was selected as the primary model based on the overall model-comparison results.

The model was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

### Final XGBoost Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 80.27% |
| Precision | 65.19% |
| Recall    | 55.08% |
| F1-score  | 59.71% |
| ROC-AUC   | 85.41% |

*These scores reflect the recorded evaluation results for the model version used in this project.*

## ⚙️ Run Locally

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Ensure the trained model files are in the project directory.
4. Run `python -m streamlit run app.py`.
5. Open the local URL displayed in the terminal.

## 🔒 Authentication

The application includes account registration and login using password hashing and a local SQLite database.

The authentication implementation is intended for academic demonstration and is not a production-grade identity system.

## 🎯 Project Objective

To demonstrate how machine learning can be used to estimate customer churn risk and present predictions through an accessible web application.

## 👩‍💻 Developer

**Shreya Shree N**

MCA Student | Machine Learning | Python | Web Applications
