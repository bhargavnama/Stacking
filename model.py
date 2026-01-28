import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write("This system predicts loan approval using a Stacking Ensemble Machine Learning model.")

st.markdown("---")

# ---------------------------------------------------------
# LOAD REAL DATASET
# ---------------------------------------------------------
# Replace your CSV file name here
df = pd.read_csv("./train_u6lujuX_CVtuZ9i (1).csv")

# Required columns
required_cols = ["Self_Employed", "ApplicantIncome", "LoanAmount", "Credit_History", "Loan_Status"]
df = df[required_cols].dropna()

# Encode Self_Employed
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})

# Encode Loan_Status
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Training data
train_data = df[["Self_Employed", "ApplicantIncome", "LoanAmount", "Credit_History"]]
train_labels = df["Loan_Status"]

# Compute min–max from REAL dataset
income_min, income_max = train_data["ApplicantIncome"].min(), train_data["ApplicantIncome"].max()
loan_min, loan_max = train_data["LoanAmount"].min(), train_data["LoanAmount"].max()

# StandardScaler for model training
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data)

# ---------------------------------------------------------
# STACKING MODEL DEFINITION
# ---------------------------------------------------------
base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('rf', RandomForestClassifier(n_estimators=150))
]

meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

stack_model.fit(X_train, train_labels)

st.success("✔ Model Trained Successfully Using Real Dataset")

st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR INPUTS (ONLY 4 FEATURES)
# ---------------------------------------------------------
st.sidebar.header("📥 Applicant Details (Only Required Features)")

# Input 1 – Self Employed
self_emp = st.sidebar.selectbox("Self-Employed", ["No", "Yes"])
self_emp_val = 1 if self_emp == "Yes" else 0

# Input 2 – Applicant Income
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)

# Input 3 – Loan Amount
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

# Input 4 – Credit History
credit_history = st.sidebar.radio("Credit History", ("Yes", "No"))
credit_history_val = 1 if credit_history == "Yes" else 0

st.markdown("---")

# ---------------------------------------------------------
# MODEL ARCHITECTURE DISPLAY
# ---------------------------------------------------------
st.subheader("🧱 Stacking Model Architecture (Read-Only)")

st.info("""
### Base Models:
- Logistic Regression  
- Decision Tree  
- Random Forest  

### Meta Model:
- Logistic Regression  

This stacking setup allows multiple models to vote, and the meta-model learns from their predictions.
""")

st.markdown("---")

# ---------------------------------------------------------
# USER INPUT SCALING (MIN–MAX)
# ---------------------------------------------------------
def min_max_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

scaled_income = min_max_scale(applicant_income, income_min, income_max)
scaled_loan = min_max_scale(loan_amount, loan_min, loan_max)

# Final feature vector
user_data = np.array([[self_emp_val, scaled_income, scaled_loan, credit_history_val]])
user_data_scaled = scaler.transform(user_data)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Base model predictions
    lr_pred = base_models[0][1].fit(X_train, train_labels).predict(user_data_scaled)[0]
    dt_pred = base_models[1][1].fit(X_train, train_labels).predict(user_data_scaled)[0]
    rf_pred = base_models[2][1].fit(X_train, train_labels).predict(user_data_scaled)[0]

    # Final Stacking Decision
    final_pred = stack_model.predict(user_data_scaled)[0]
    final_prob = stack_model.predict_proba(user_data_scaled)[0][1] * 100

    # ---------------------------------------------------------
    # OUTPUT SECTION
    # ---------------------------------------------------------
    st.subheader("📊 Base Model Predictions")
    st.write(f"**Logistic Regression:** {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if rf_pred else 'Rejected'}")

    st.markdown("---")

    st.subheader("🧠 Final Stacking Decision")

    if final_pred == 1:
        st.success(f"✅ Loan Approved (Confidence: {final_prob:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {final_prob:.2f}%)")

    st.markdown("---")

    # ---------------------------------------------------------
    # BUSINESS EXPLANATION
    # ---------------------------------------------------------
    st.subheader("💼 Business Explanation")
    st.info("""
The model uses the applicant’s self-employment status, normalized income and loan amount values,
and credit history to evaluate repayment likelihood.

Each base model gives a prediction, and the stacking model combines them to enhance accuracy.

Thus, the system provides the final decision on loan approval.
""")

