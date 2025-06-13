import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load the model and preprocessor
model = joblib.load("fraud_detection_xgboost.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🚨 Fraud Detection App")

st.markdown("Enter transaction details below to predict if it’s fraudulent.")

# Original features expected BEFORE preprocessing
original_inputs = {}

# === Manually input features ===
original_inputs['type'] = st.selectbox("Transaction Type", ["Credit", "Debit"])
original_inputs['channel'] = st.selectbox("Transaction Channel", ["ATM", "Branch", "Online"])
original_inputs['occupation'] = st.selectbox("User Occupation", ["Doctor", "Engineer", "Retired", "Student"])

numeric_features = [
    'amount', 'duration', 'login_attempts', 'balance', 'account_tx_count',
    'account_avg_amount', 'account_std_amount', 'time_since_last_tx',
    'amount_to_balance', 'device_usage_freq', 'ip_usage_freq',
    'merchant_risk', 'location_risk'
]

for feature in numeric_features:
    original_inputs[feature] = st.number_input(feature.replace('_', ' ').title(), min_value=0.0)

# === Auto-filled time-based features ===
now = datetime.now()
original_inputs['hour'] = now.hour
original_inputs['minute'] = now.minute
original_inputs['day'] = now.day
original_inputs['weekday'] = now.weekday()  # Monday = 0, Sunday = 6
original_inputs['month'] = now.month
original_inputs['weekofyear'] = now.isocalendar().week

st.write("⏱️ Time features filled automatically:")
st.text(f"Hour: {now.hour}, Minute: {now.minute}, Day: {now.day}, Weekday: {now.weekday()}, Month: {now.month}, Week #: {now.isocalendar().week}")

# === Prediction ===
if st.button("Predict"):
    try:
        # Create DataFrame with one row
        input_df = pd.DataFrame([original_inputs])

        # Preprocess
        X_transformed = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(X_transformed)[0]
        proba = model.predict_proba(X_transformed)[0][int(prediction)]

        label = "Fraudulent ❌" if prediction == 1 else "Legitimate ✅"
        st.success(f"Prediction: **{label}**\n\nConfidence: **{proba*100:.2f}%**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
