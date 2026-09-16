from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
	response = client.get("/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok"}


def test_predict_requires_trained_model(monkeypatch, tmp_path) -> None:
	monkeypatch.setattr("app.main.MODEL_PATH", tmp_path / "missing-model.joblib")
	response = client.post(
		"/predict",
		json={
			"age": 35,
			"income": 60000,
			"loan_amount": 50000,
			"credit_score": 700,
			"months_employed": 60,
			"num_credit_lines": 3,
			"interest_rate": 8.5,
			"loan_term": 36,
			"dti_ratio": 0.3,
		},
	)

	assert response.status_code == 503
