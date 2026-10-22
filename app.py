import streamlit as st
import numpy as np
import joblib

# 1. Saved Model aur Scaler ko Load karna
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Streamlit Page Design
st.set_page_config(page_title="Fraud Detection & Risk Scoring System", page_icon="🛡️", layout="centered")

st.title("🛡️ Online Payment Fraud Detection System")
st.write("Enter transaction details below to check the real-time risk score and automated decision.")

# 3. User Inputs (Form Fields matching dataset columns)
st.subheader("Transaction Parameters")

# Type dropdown (PaySim dataset types)
trans_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
type_mapping = {"PAYMENT": 0, "CASH_IN": 1, "CASH_OUT": 2, "DEBIT": 3, "TRANSFER": 4}
type_val = type_mapping.get(trans_type, 0)

step = st.number_input("Time Step (Hour representation)", min_value=1, value=1, step=1)
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0, step=50.0)
oldbalanceOrg = st.number_input("Sender's Old Balance ($)", min_value=0.0, value=5000.0, step=100.0)
newbalanceOrig = st.number_input("Sender's New Balance ($)", min_value=0.0, value=4000.0, step=100.0)
oldbalanceDest = st.number_input("Receiver's Old Balance ($)", min_value=0.0, value=0.0, step=100.0)
newbalanceDest = st.number_input("Receiver's New Balance ($)", min_value=0.0, value=0.0, step=100.0)

# 4. Prediction & Business Logic Button
if st.button("Analyze Transaction Risk", type="primary"):
    # Features array in exact training order: [type, step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    input_features = np.array([[type_val, step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
    
    # Scale the input
    scaled_input = scaler.transform(input_features)
    
    # Get probability of fraud
    fraud_prob = model.predict_proba(scaled_input)[0][1]
    risk_score = round(fraud_prob * 100, 2)
    
    # Display Results with Styling
    st.markdown("---")
    st.subheader("Analysis Results")
    
    st.metric(label="Calculated Fraud Risk Score", value=f"{risk_score} / 100")
    
    # Business Action logic
    if risk_score <= 30:
        st.success("Status: **APPROVE (Low Risk)** ✅ — Transaction looks safe to proceed.")
    elif risk_score <= 75:
        st.warning("Status: **REVIEW (Medium Risk)** ⚠️ — Additional OTP verification required.")
    else:
        st.error("Status: **BLOCK (High Risk)** ❌ — High probability of fraud! Transaction declined.")