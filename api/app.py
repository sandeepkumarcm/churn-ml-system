from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="Customer Churn Prediction API")

# LOAD MODEL
model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# INPUT SCHEMA
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# HOME
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}

# PREDICT
@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_df = pd.DataFrame([data.dict()])

        # ================= FEATURE ENGINEERING =================

        input_df["tenure_group"] = pd.cut(
            input_df["tenure"],
            bins=[0,12,24,48,72],
            labels=["0-12","12-24","24-48","48-72"]
        )

        input_df["avg_monthly_spend"] = input_df["TotalCharges"] / input_df["tenure"]
        input_df["avg_monthly_spend"].replace([np.inf, -np.inf], 0, inplace=True)
        input_df["avg_monthly_spend"].fillna(0, inplace=True)

        service_cols = [
            "PhoneService","MultipleLines","InternetService",
            "OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies"
        ]

        input_df["service_count"] = 0
        for col in service_cols:
            input_df["service_count"] += input_df[col].apply(
                lambda x: 1 if x in ["Yes","DSL","Fiber optic"] else 0
            )

        input_df["contract_score"] = input_df["Contract"].map({
            "Month-to-month":0,
            "One year":1,
            "Two year":2
        })

        input_df["is_auto_payment"] = input_df["PaymentMethod"].apply(
            lambda x: 1 if "automatic" in x.lower() else 0
        )

        # ======================================================

        processed = preprocessor.transform(input_df)

        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        return {
            "churn": bool(prediction),
            "probability": round(float(probability), 4)
        }

    except Exception as e:
        return {"error": str(e)}