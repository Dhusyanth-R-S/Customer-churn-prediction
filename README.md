# 📊 Customer Churn Prediction Using Machine Learning

An end-to-end ML project predicting telecom customer churn with **full preprocessing**, **EDA**, **multiple model training**, **hyperparameter tuning (GridSearchCV & RandomizedSearchCV)**, and **final pipeline selection**.  
A complete industry-style workflow.

---

## 🧠 Project Overview

Customer churn prediction helps telecom companies prevent customer loss.  
This project implements a **full ML lifecycle**, including:

- Data cleaning  
- Exploratory Data Analysis  
- Feature engineering  
- Preprocessing with ColumnTransformer  
- Multiple model training  
- **Hyperparameter tuning using GridSearchCV & RandomizedSearchCV**  
- Model comparison  
- Selecting the best-performing model  
- Saving the final pipeline as a `.pkl` file  

---

## 📁 Dataset

**Dataset:** Telco Customer Churn  
**Rows:** 7043  
**Target Column:** `Churn`  
**Problem Type:** Binary Classification  

### Key Features
- Customer demographics  
- Phone & internet service details  
- Contract & payment types  
- Monthly and total charge patterns  

### Data Cleaning Summary
- Removed `customerID`  
- Converted `total_charges` to numeric (with coercion)  
- Filled remaining missing values  
- Standardized column names  

Dataset stored in:

[Click here for Dataset](https://github.com/Dhusyanth-R-S/Customer-churn-prediction/blob/main/WA_Fn-UseC_-Telco-Customer-Churn.csv)

---

## 🛠️ Project Workflow

### 1️⃣ Data Cleaning
- Converted columns to proper types  
- Replaced missing values  
- Unified naming convention  
- Removed irrelevant identifiers  

---

### 2️⃣ Exploratory Data Analysis (EDA)
EDA included:

- Churn distribution  
- Distribution of numerical columns  
- Churn vs monthly charges  
- Churn vs contract type  
- Churn vs tenure  
- Identification of patterns contributing to churn  

Visualized using **Matplotlib & Seaborn**.

---

## 🔧 Preprocessing

A **ColumnTransformer** was used to apply:

| Feature Type   | Technique                         |
|----------------|-----------------------------------|
| Numerical      | StandardScaler                    |
| Categorical    | OneHotEncoder / OrdinalEncoder    |
| Target         | LabelEncoder                      |

All transformations were wrapped inside a **single ML pipeline**.

---

## 🤖 Model Development

### 🧪 Models Tried (Before Picking the Best One)

Trained and evaluated multiple algorithms:

- Logistic Regression  
- K-Nearest Neighbors  
- Support Vector Classifier  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- XGBoost Classifier 

### 🔍 Hyperparameter Tuning

Used:

- **GridSearchCV** to exhaustively search best parameters  
- **RandomizedSearchCV** to efficiently explore wide ranges  
- 5-fold cross-validation  
- Scoring based on accuracy/precision/recall  

### 🏆 Best Model Selected

After comparing all models:

- **RandomForestClassifier (tuned version)**  
  showed **consistently better performance**, balanced generalization, and stability.

<img width="1458" height="491" alt="image" src="https://github.com/user-attachments/assets/ab19172c-ce88-4079-a01c-d7137c877077" />


This tuned RF model was placed inside the final pipeline.

---

## 📈 Model Evaluation

Evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

**Note:**  
Like typical churn datasets, the minority class (Churn = Yes) is hard and recall was moderate even after tuning — this is normal without oversampling methods.

---

## Top 10 features affecting churn

<img width="809" height="393" alt="image" src="https://github.com/user-attachments/assets/4378ae81-0aaa-4dec-9f52-27b34a7fc729" />

---
## 💾 Saving the Final Model

The complete pipeline (preprocessing + tuned model) was saved as:

```
models/churn_model.pkl
```

Using:

```python
joblib.dump(rf_pipeline, "models/churn_model.pkl")
```

## ▶️ How to Use the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load("models/churn_model.pkl")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)
pred = model.predict(X.head())

print(pred)
```

---

## 📦 Project Structure

```
customer_churn_prediction/
│
├── README.md
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_model.pkl
├── customer_churn_project_NoteBook.ipynb
├── customer_churn_report.pdf
└── customer_churn_presentation.pdf

---

## 🔍 Key Insights

- Month-to-month contract customers churn the most  
- High monthly charges increase churn probability  
- Longer-tenure customers are more stable  
- Automatic model selection + tuning significantly improves performance  
- Pipelines ensure reproducibility  

---

## 👨‍💻 Author

**Dhusyanth R S**  
Machine Learning Practitioner

