import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import random

# Load model and preprocessor
model = joblib.load("fraud_detection_xgboost.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🚨 Fraud Detection App")
st.markdown("Enter transaction details to predict fraud.")

# ========== Input Section ==========

# Categorical inputs
original_inputs = {
    'type': st.selectbox("Transaction Type", ["Credit", "Debit"]),
    'channel': st.selectbox("Transaction Channel", ["ATM", "Branch", "Online"]),
    'occupation': st.selectbox("User Occupation", ["Doctor", "Engineer", "Retired", "Student"]),
}

# Numeric inputs (user-filled)
user_numeric = {
    'amount': st.number_input("Transaction Amount", min_value=0.0),
    'login_attempts': st.number_input("Login Attempts", min_value=0),
    'balance': st.number_input("Account Balance", min_value=0.0),
    'account_tx_count': st.number_input("Total Account Transactions", min_value=0),
    'account_avg_amount': st.number_input("Average Transaction Amount", min_value=0.0),
    'account_std_amount': st.number_input("Std Dev of Transaction Amounts", min_value=0.0),
    'amount_to_balance': st.number_input("Amount to Balance Ratio", min_value=0.0),
    'merchant_risk': st.slider("Merchant Risk (0–1)", 0.0, 1.0, 0.2),
    'location_risk': st.slider("Location Risk (0–1)", 0.0, 1.0, 0.3),
}

# ========== Auto Time Feature Section ==========

# Simulated values (or could be linked to real time logs in production)
now = datetime.now()
random.seed(now.microsecond)

original_inputs.update(user_numeric)

# Simulate time-based fields (these could also come from logs)
original_inputs['duration'] = random.uniform(1, 60)  # e.g., seconds since login
original_inputs['time_since_last_tx'] = random.uniform(0.5, 48)  # hours
original_inputs['device_usage_freq'] = random.uniform(0, 10)  # logins per day
original_inputs['ip_usage_freq'] = random.uniform(0, 10)  # logins per IP

# ========== Prediction Section ==========

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([original_inputs])
        
        # Preprocess and predict
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)[0]
        proba = model.predict_proba(X_transformed)[0][int(prediction)]

        label = "Fraudulent ❌" if prediction == 1 else "Legitimate ✅"
        st.success(f"Prediction: **{label}**\nConfidence: **{proba * 100:.2f}%**")

        # Show all values used in prediction
        with st.expander("🔍 Details of Transaction"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
