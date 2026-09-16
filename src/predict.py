from pathlib import Path

import joblib
import pandas as pd

from src.train import FEATURES


def predict_loan(
	features: dict[str, float | int], model_path: str | Path = "models/best_model.joblib"
) -> int:
	model = joblib.load(model_path)
	input_data = pd.DataFrame([features], columns=FEATURES)
	return int(model.predict(input_data)[0])
