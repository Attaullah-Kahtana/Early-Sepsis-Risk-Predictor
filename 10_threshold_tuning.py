import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

def test_thresholds(model_path, test_file, model_name):
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_file)

    drop_cols = ["PatientID", "SepsisLabel"]
    if "Source" in test_df.columns:
        drop_cols.append("Source")

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["SepsisLabel"]

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"Threshold sweep for: {model_name}")
    print(f"{'='*60}")

    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        print(f"\n--- Threshold = {threshold} ---")
        print(classification_report(y_test, y_pred, zero_division=0))

# Full-data model
test_thresholds(
    r"C:\sepsis proj\full_model.pkl",
    r"C:\sepsis proj\full_test_features.csv",
    "Full-Data Model"
)

# Vitals-only model
test_thresholds(
    r"C:\sepsis proj\vitals_model.pkl",
    r"C:\sepsis proj\vitals_test_features.csv",
    "Vitals-Only Model"
)