# Kelompok 2 - Latihan 8 Dashboard

Dash dashboard for customer churn analysis and prediction.

## Model selector

Model C (Gradient Boosting, `churn_model.joblib`) is listed first in the dropdown and selected by default. The "Tanpa model" option shows actual churn data, raw-data charts, and a customer table without predictions. Choose Model A (Logistic Regression), Model B (Logistic Regression + Random Oversampling), or Model C to update the risk charts and retention table. Single-customer prediction is available after selecting a model.

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

## CSV upload

Use the sidebar to upload a UTF-8 CSV with up to 5,000 customers (10 MB). `data/customer_churn.csv` and `data/new_customers.csv` are example formats. The model feature columns are required; `customer_id`, `signup_date`, `churn`, and `total_charges` are optional. Missing `total_charges` is calculated from tenure and monthly charges.

The overview filters, charts, and table use the uploaded rows. If the file has no `churn` column, the raw view shows customer distributions and model views show predictions without an actual churn rate. Uploaded data stays in the current browser tab; click **Gunakan data bawaan** to restore the bundled dataset.