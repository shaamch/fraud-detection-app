import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Only need joblib now
import random
from datetime import datetime

# Load both files with joblib
model = joblib.load("fraud_detection_xgboost.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # Updated line

st.title("🚨 Smart Fraud Detection App")
st.markdown("Enter minimum transaction details. System will auto-fill the rest.")

# User inputs (only 4 inputs from user)
user_input = {
    'type': st.selectbox("Transaction Type", ["Credit", "Debit"]),
    'channel': st.selectbox("Transaction Channel", ["ATM", "Branch", "Online"]),
    'occupation': st.selectbox("User Occupation", ["Doctor", "Engineer", "Retired", "Student"]),
    'amount': st.number_input("Transaction Amount", min_value=0.0, value=100.0)
}

# Generate defaults for remaining features
auto_inputs = {
    'duration': random.uniform(1, 60),
    'login_attempts': random.randint(0, 5),
    'balance': random.uniform(1000, 50000),
    'account_tx_count': random.randint(5, 100),
    'account_avg_amount': random.uniform(50, 5000),
    'account_std_amount': random.uniform(10, 1000),
    'time_since_last_tx': random.uniform(0.1, 48),
    'amount_to_balance': user_input['amount'] / max(1, random.uniform(1000, 50000)),
    'device_usage_freq': random.uniform(0.1, 10),
    'ip_usage_freq': random.uniform(0.1, 10),
    'merchant_risk': random.uniform(0, 1),
    'location_risk': random.uniform(0, 1),
}

# Merge inputs
all_inputs = {**user_input, **auto_inputs}

# Predict button
if st.button("Predict Fraud"):
    try:
        input_df = pd.DataFrame([all_inputs])
        transformed_input = preprocessor.transform(input_df)
        prediction = model.predict(transformed_input)[0]
        prob = model.predict_proba(transformed_input)[0][int(prediction)]

        label = "Fraudulent ❌" if prediction == 1 else "Legitimate ✅"
        st.success(f"Prediction: **{label}**\nConfidence: **{prob * 100:.2f}%**")

        with st.expander("🔍 Inputs Used"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
