# Kelompok 2 - Latihan 8 Dashboard

Dash dashboard for customer churn analysis and prediction.

## Model selector

The default "Tanpa model" option shows actual churn data, raw-data charts, and a customer table without predictions. Choose Model A (Logistic Regression), Model B (Logistic Regression + Random Oversampling), or Model C (Gradient Boosting) to update the risk charts and retention table. Single-customer prediction is available after selecting a model.

## Run locally

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
pip install -r requirements.txt
python sesi8/app_final_dashboard.py
```

Open http://127.0.0.1:8053/.

The dashboard reads `data/customer_churn.csv`, `utils/model_utils.py`, and the saved models in `sesi8/top3_models_cv_roc_auc/`.