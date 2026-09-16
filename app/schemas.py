from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
	age: int = Field(ge=18)
	income: float = Field(gt=0)
	loan_amount: float = Field(gt=0)
	credit_score: int = Field(ge=300, le=850)
	months_employed: int = Field(ge=0)
	num_credit_lines: int = Field(ge=0)
	interest_rate: float = Field(ge=0)
	loan_term: int = Field(gt=0)
	dti_ratio: float = Field(ge=0)


class PredictionResponse(BaseModel):
	prediction: int
