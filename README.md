# 📊 Customer Churn Prediction Using Machine Learning

A complete end-to-end machine learning project predicting **customer churn** for a telecom company using industry-standard preprocessing, feature engineering, pipeline creation, and model evaluation techniques.

---

## 🧠 Project Overview

Customer churn is a major business challenge for telecom companies.  
This project builds a machine learning model that predicts whether a customer is likely to **churn**, enabling proactive retention strategies.

The workflow includes:

- Data cleaning  
- Exploratory Data Analysis (EDA)  
- Preprocessing with ColumnTransformer  
- ML pipeline creation  
- Model training & evaluation  
- Saving the final pipeline as a `.pkl` file  

---

## 📁 Dataset

**Dataset:** Telco Customer Churn  
**Total Rows:** 7043  
**Target Column:** `Churn`  
**Problem Type:** Binary Classification

### Key Features
- Demographics  
- Services (Internet, phone, security, streaming)  
- Contract and payment details  
- Monthly & total charges  

### Key Cleaning Steps
- Removed non-predictive `customerID`  
- Converted `total_charges` to numeric  
- Treated missing values  
- Standardized naming format  

Dataset is stored under:

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 🛠️ Project Workflow

### 1️⃣ Data Cleaning
- Removed irrelevant columns  
- Converted numeric columns stored as strings  
- Checked missing values and handled them  
- Standardized column naming  

---

### 2️⃣ Exploratory Data Analysis (EDA)

EDA included:

- Churn distribution  
- Numerical feature distributions  
- Correlation patterns  
- Churn vs Contract type  
- Churn vs Tenure  
- Visualizations using Matplotlib & Seaborn  

---

### 3️⃣ Feature Preprocessing

Performed using a **ColumnTransformer**:

| Feature Type   | Technique                         |
|----------------|-----------------------------------|
| Numerical      | StandardScaler                    |
| Categorical    | OneHotEncoder / OrdinalEncoder    |
| Target         | LabelEncoder                      |

All preprocessing steps were integrated into a **single pipeline**.

---

### 4️⃣ Model Development

**Model used:** `RandomForestClassifier`

Reasons:
- Handles mixed types well  
- Robust to noise and outliers  
- Strong baseline for churn prediction  

A complete ML pipeline was created:

```
ColumnTransformer → RandomForestClassifier
```

---

### 5️⃣ Model Evaluation

Metrics evaluated:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

Note:  
Churn datasets are highly imbalanced; minority class performance is moderate without techniques like SMOTE or class reweighting.

---

## 💾 Saving the Final Model

The entire pipeline (preprocessing + model) was saved as:

```
models/churn_model.pkl
```

Using:

```python
joblib.dump(rf_pipeline, "models/churn_model.pkl")
```

This allows direct prediction without re-running preprocessing.

---

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
├── data/
│     └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebook/
│     └── churn_project.ipynb
│
├── models/
│     └── churn_model.pkl
│
├── README.md
└── requirements.txt
```

---

## 🔍 Key Insights & Learnings

- Month-to-month contracts strongly correlate with churn  
- High monthly charges increase churn likelihood  
- Longer-tenure customers are less likely to churn  
- Class imbalance affects recall for the churn class  
- Pipelines ensure reproducibility and deployment readiness  

---

## 🚀 Future Improvements

- Apply SMOTE or class-weighting  
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Try advanced models (XGBoost, LightGBM, CatBoost)  
- Add feature importance visualizations  
- Deploy using Streamlit / FastAPI  

---

## 👨‍💻 Author

**Dhusyanth R S**  
Machine Learning Practitioner

