import streamlit as st
import requests
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ================= SIDEBAR =================
st.sidebar.title("📌 Project Info")

st.sidebar.info("""
**Model:** XGBoost  
**Dataset:** Telco Customer Churn  
**Goal:** Predict customer churn  
**Tech Stack:** FastAPI + Streamlit  
""")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.subtitle {
    font-size: 18px;
    color: #7f8c8d;
}
.stButton>button {
    background-color: #2ecc71;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<p class="title">📊 Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict whether a customer will churn using Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# ================= LAYOUT =================
col1, col2 = st.columns(2)

# ===== LEFT =====
with col1:
    st.subheader("👤 Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.subheader("📞 Services")

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])

# ===== RIGHT =====
with col2:
    st.subheader("🎬 Streaming & Billing")

    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    st.subheader("💰 Charges")

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# ================= BUTTON =================
st.divider()
center_col = st.columns([1, 2, 1])[1]

with center_col:
    predict_button = st.button("🚀 Predict Churn")

# ================= PREDICTION =================
if predict_button:

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        # ✅ FINAL API URL (AUTO SWITCH)
        API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

        # 🔥 LOADING EFFECT
        with st.spinner("🔍 Predicting..."):
            response = requests.post(API_URL, json=data)

        result = response.json()

        st.divider()

        if "error" in result:
            st.error(result["error"])

        else:
            churn = result["churn"]
            prob = result["probability"]

            # 🔥 RESULT UI
            if churn:
                st.markdown(f"""
                <div style='background-color:#ff4b4b;padding:25px;border-radius:12px;color:white;text-align:center'>
                    <h2>⚠️ High Risk of Churn</h2>
                    <h3>Probability: {prob}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#2ecc71;padding:25px;border-radius:12px;color:white;text-align:center'>
                    <h2>✅ Customer Likely to Stay</h2>
                    <h3>Probability: {prob}</h3>
                </div>
                """, unsafe_allow_html=True)

            st.progress(prob)

    except Exception as e:
        st.error("⚠️ API not running. Please start FastAPI server.")