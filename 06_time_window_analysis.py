import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_lead_time(train_file, test_file, hours_before, model_name):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Shift the label backward: predict sepsis N hours before it happens
    # This requires the data to be sorted by PatientID and ICULOS (hour index)
    for df in [train_df, test_df]:
        df.sort_values(["PatientID", "ICULOS"], inplace=True)
        df["FutureSepsisLabel"] = df.groupby("PatientID")["SepsisLabel"].shift(-hours_before)
        df["FutureSepsisLabel"] = df["FutureSepsisLabel"].fillna(0)

    drop_cols = ["PatientID", "SepsisLabel", "FutureSepsisLabel"]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["FutureSepsisLabel"]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["FutureSepsisLabel"]

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    print(f"{model_name} — Predicting {hours_before}h ahead: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    return auroc, auprc

print("=== FULL-DATA MODEL ===")
for h in [6, 12]:
    evaluate_lead_time(
        r"C:\sepsis proj\full_train_features.csv",
        r"C:\sepsis proj\full_test_features.csv",
        h, "Full-Data"
    )

print("\n=== VITALS-ONLY MODEL ===")
for h in [6, 12]:
    evaluate_lead_time(
        r"C:\sepsis proj\vitals_train_features.csv",
        r"C:\sepsis proj\vitals_test_features.csv",
        h, "Vitals-Only"
    )