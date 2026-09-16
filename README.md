# Credit Scoring MLOps API

Ce projet entraîne deux modèles de classification sur des données de prêts,
compare leurs performances avec MLflow, sauvegarde le meilleur modèle et
l'expose via une API FastAPI conteneurisée.

## Architecture

- `data/loan_data.csv` : données d'entraînement et cible `Default`
- `src/train.py` : séparation train/test, Random Forest, XGBoost, métriques MLflow et sélection du meilleur F1-score
- `src/predict.py` : chargement du modèle sélectionné et prédiction
- `app/` : API FastAPI et schémas de validation
- `tests/` : tests de santé et de disponibilité du modèle
- `.github/workflows/ci.yml` : installation et exécution des tests

Le modèle utilise les colonnes numériques `Age`, `Income`, `LoanAmount`,
`CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`
et `DTIRatio`. La cible `Default` vaut `0` pour un prêt sans défaut et `1`
pour un prêt en défaut.

## Installation locale (WSL)

Python  3.14 est recommandé.

```powershell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entraînement et comparaison MLflow

```powershell
python -m src.train
mlflow ui
```

La commande crée `models/best_model.joblib` et les expériences dans `mlruns/`.
MLflow est disponible sur http://localhost:5000 pour comparer l'accuracy,
la précision, le recall et le F1-score des deux modèles.

## API

```powershell
uvicorn app.main:app --reload
```

Documentation interactive : http://localhost:8000/docs

Exemple de requête :

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"age":35,"income":60000,"loan_amount":50000,"credit_score":700,"months_employed":60,"num_credit_lines":3,"interest_rate":8.5,"loan_term":36,"dti_ratio":0.3}'
```

Réponse :

```json
{"prediction": 0}
```

## Tests

```powershell
pytest
```

## Docker

L'image entraîne le modèle pendant le build puis démarre l'API :

```powershell
docker compose up --build
```

L'API est disponible sur http://localhost:8000.

