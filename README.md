# Employee Salary Prediction

## Overview
This project is a machine learning–based web application that predicts whether an employee earns more than $50K or not based on demographic and professional details.

The application uses a trained classification model and provides predictions through a simple web interface.

---

## Problem Statement
Given employee details such as age, education, occupation, and work experience, predict whether the salary category is:
- **> 50K**
- **≤ 50K**

---

## Technologies Used
- Python  
- Scikit-learn  
- Streamlit  
- Pandas  
- NumPy  
- Joblib  

---

## Project Workflow
1. Collected and explored real-world census dataset  
2. Performed data preprocessing:
   - Handled missing values
   - Encoded categorical variables
   - Removed invalid records
3. Trained multiple classification models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
4. Selected the best-performing model based on accuracy
5. Deployed the model using Streamlit for real-time predictions

---

## Application Output
- Takes employee details as input
- Predicts salary category (**>50 or ≤50K**) instantly
  
---

## How to Run the Project
1. Clone the repository
  bash
   git clone https://github.com/ThanmaiReddy07/Employee-Salary-Prediction.git
2. Install required libraries
  bash
   pip install -r requirements.txt
3. Run the streamlit app
  bash
   streamlit run app.py

---

## Key Learnings
- End-to-end ML project implementation
- Data preprocessing and feature encoding
- Model comparison and evaluation
- Deploying ML models using Streamlit
