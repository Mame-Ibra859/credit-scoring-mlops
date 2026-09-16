from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "models/best_model.joblib"
TARGET = "Default"
FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def main(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> str:
    df = pd.read_csv(data_path)
    missing_columns = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {sorted(missing_columns)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment("credit-scoring")
    candidates = {
        "random-forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    results = []
    for run_name, model in candidates.items():
        with mlflow.start_run(run_name=run_name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_param("model", type(model).__name__)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            if run_name == "random-forest":
                mlflow.sklearn.log_model(model, "model")
            else:
                mlflow.xgboost.log_model(model, "model")
            results.append((metrics["f1"], model))

    _, best_model = max(results, key=lambda result: result[0])
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_path)
    return str(output_path)


if __name__ == "__main__":
    main()