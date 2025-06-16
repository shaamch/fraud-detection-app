import streamlit as st
import joblib
import numpy as np

# Fix for scikit-learn version mismatch
from types import SimpleNamespace
import sklearn.compose

if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        __doc__ = "Marker class to remain columns"
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

if not hasattr(sklearn.compose._column_transformer, '_ColumnTransformer'):
    sklearn.compose._column_transformer._ColumnTransformer = SimpleNamespace

# Load the model and preprocessor
model = joblib.load("fraud_detection_xgboost.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Define your feature names (customize this list as per your model)
FEATURE_NAMES = ['amount', 'transaction_type', 'origin_account_age', 'destination_account_age']

st.title("🚨 Fraud Detection App")
st.markdown("Enter transaction details below:")

# Input fields
user_input = []
for feature in FEATURE_NAMES:
    value = st.text_input(f"{feature.replace('_', ' ').title()}")
    user_input.append(value)

if st.button("Predict"):
    try:
        # Prepare input
        X_input = np.array(user_input).reshape(1, -1)
        X_processed = preprocessor.transform(X_input)
        
        prediction = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][int(prediction)]

        label = "Fraudulent ❌" if prediction == 1 else "Legitimate ✅"
        st.success(f"Prediction: **{label}**\n\nConfidence: **{proba*100:.2f}%**")

    except Exception as e:
        st.error(f"Error: {e}")
