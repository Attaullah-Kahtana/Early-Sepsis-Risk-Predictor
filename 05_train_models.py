import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import joblib

def train_and_evaluate(train_file, test_file, model_name, save_model_path):
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    drop_cols = ["PatientID", "SepsisLabel"]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["SepsisLabel"]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["SepsisLabel"]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)

    print(f"\nAUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model to disk for later use (SHAP, further analysis)
    joblib.dump(model, save_model_path)
    print(f"Model saved to: {save_model_path}")

    return model, X_test, y_test

# Full-data model (labs + vitals)
full_model, full_X_test, full_y_test = train_and_evaluate(
    r"C:\sepsis proj\full_train_features.csv",
    r"C:\sepsis proj\full_test_features.csv",
    "Full-Data Model (with labs + features)",
    r"C:\sepsis proj\full_model.pkl"
)

# Vitals-only model (resource-limited)
vitals_model, vitals_X_test, vitals_y_test = train_and_evaluate(
    r"C:\sepsis proj\vitals_train_features.csv",
    r"C:\sepsis proj\vitals_test_features.csv",
    "Vitals-Only Model (resource-limited + features)",
    r"C:\sepsis proj\vitals_model.pkl"
)