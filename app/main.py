from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.schemas import LoanApplication, PredictionResponse
from src.predict import predict_loan

app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
)

MODEL_PATH = Path("models/best_model.joblib")


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: LoanApplication) -> PredictionResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained")

    values = {
        "Age": data.age,
        "Income": data.income,
        "LoanAmount": data.loan_amount,
        "CreditScore": data.credit_score,
        "MonthsEmployed": data.months_employed,
        "NumCreditLines": data.num_credit_lines,
        "InterestRate": data.interest_rate,
        "LoanTerm": data.loan_term,
        "DTIRatio": data.dti_ratio,
    }
    return PredictionResponse(prediction=predict_loan(values, MODEL_PATH))