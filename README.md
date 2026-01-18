# 💼 Employee Salary Prediction

## 📌 Overview
This project is a **machine learning–based web application** that predicts whether an employee earns **more than $50K or not** based on demographic and professional details.  
The application trains multiple classification models, selects the best one based on accuracy, and provides predictions through a simple **Streamlit web interface**.  
It also includes **Matplotlib visualizations** for model comparison and feature distributions.

---

## 🎯 Problem Statement
Given employee details such as age, education, occupation, and work experience, predict whether the salary category is:

- **> 50K**
- **≤ 50K**

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- Joblib  
- Streamlit  
- Matplotlib  

---

## 🧩 Project Workflow
1. **Data Collection & Exploration**  
   - Used the Adult Census dataset.  

2. **Data Preprocessing**  
   - Handled missing values.  
   - Encoded categorical variables.  
   - Removed invalid records.  

3. **Model Training & Evaluation**  
   - Trained multiple classification models:  
     - Logistic Regression  
     - Random Forest  
     - Gradient Boosting  
   - Compared accuracy scores and selected the best-performing model.  
   - Visualized model performance and feature distributions using Matplotlib.  

4. **Deployment**  
   - Integrated the best model into a Streamlit app.  
   - Enabled real-time predictions via a user-friendly interface.  
   - Supported both single predictions and batch CSV uploads.  

---

## 📊 Application Output
- Takes employee details as input.  
- Predicts salary category (**>50K or ≤50K**) instantly.  
- Displays visualizations for model comparison and feature distributions.  
- Supports batch CSV uploads with downloadable prediction results.  

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/ThanmaiReddy07/Employee-Salary-Prediction.git
   cd Employee-Salary-Prediction

2. Install required libraries:
   ```bash
   pip install -r requirements.txtbash

3. Run the Streamlit app:
   ```bash
   streamlit run app.py

---

📚 Key Learnings- End-to-end ML project implementation.
- Data preprocessing and feature encoding.
- Model comparison and evaluation.
- Visualizing results with Matplotlib.
- Deploying ML models using Streamlit.
- Building reproducible workflows with GitHub
