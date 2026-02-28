import streamlit as st
import pickle
import numpy as np
import pandas as pd 

# Load saved model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection", layout="wide")

# ---------- CUSTOM CSS (Banking Theme) ----------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

.green-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 20px;
    font-weight: bold;
}

.red-card {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("💳 Credit Card Fraud Detection System")

st.write("Enter transaction details below 👇")

# ---------- INPUT SECTION ----------
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", value=100.0)

with col2:
    time = st.number_input("Transaction Time", value=1.0)

# ---------- RANDOM FEATURES ----------
if st.button("🎲 Generate Random Features"):
    random_features = np.random.normal(0, 1, 28)
    st.session_state["random_features"] = random_features
    st.success("Random features generated!")

if "random_features" not in st.session_state:
    st.session_state["random_features"] = np.random.normal(0, 1, 28)

# ---------- PREDICT ----------
if st.button("🔍 Predict Fraud"):

    features = st.session_state["random_features"]

    input_data = np.concatenate(([time], features, [amount]))
    input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(
            '<div class="red-card">⚠️ Fraudulent Transaction Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="green-card">✅ Valid Transaction</div>',
            unsafe_allow_html=True
        )

    # ---------- DASHBOARD ----------
    st.subheader("📊 Transaction Analytics")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Amount", f"₹{amount}")
        st.metric("Time", f"{time}")

    with col4:
        prob = model.predict_proba(input_data)[0][1]
        st.metric("Fraud Probability", f"{prob:.2%}")




   