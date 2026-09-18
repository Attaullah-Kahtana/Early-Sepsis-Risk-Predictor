import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

# This assumes a 'Source' column exists identifying setA vs setB
# If it doesn't exist yet, it must be added during data loading (see note above)

def check_subgroup_performance(model_path, test_file, source_col="Source"):
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_file)

    if source_col not in test_df.columns:
        print(f"WARNING: '{source_col}' column not found. Cannot run subgroup analysis without hospital source info.")
        return

    drop_cols = ["PatientID", "SepsisLabel", source_col]
    for source_value in test_df[source_col].unique():
        subset = test_df[test_df[source_col] == source_value]
        X_subset = subset.drop(columns=drop_cols)
        y_subset = subset["SepsisLabel"]

        y_pred_proba = model.predict_proba(X_subset)[:, 1]
        auroc = roc_auc_score(y_subset, y_pred_proba)
        print(f"Source = {source_value}: AUROC = {auroc:.4f}, n_patients = {subset['PatientID'].nunique()}")

check_subgroup_performance(r"C:\sepsis proj\full_model.pkl", r"C:\sepsis proj\full_test_features.csv")
