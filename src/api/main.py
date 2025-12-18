# src/api/main.py
from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import pandas as pd
import joblib  # use joblib to load the saved model

app = FastAPI(title="Credit Risk API")

# Load the best model directly with joblib
MODEL_PATH = "models/best_model.joblib"
model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict credit risk based on input features.
    """
    # Convert request data to DataFrame for the model
    input_df = pd.DataFrame([request.dict()])
    
    # Predict probability of risk
    prob = model.predict_proba(input_df)[0][1]
    
    # Generate label (0 = low risk, 1 = high risk)
    label = int(prob > 0.5)
    
    return PredictionResponse(risk_probability=prob, risk_label=label)


