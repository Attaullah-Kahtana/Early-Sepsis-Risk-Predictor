


import matplotlib
matplotlib.use("Agg")

import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib


def run_shap_analysis(model_path, test_file, model_name, sample_size=1000):
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_file)

    drop_cols = ["PatientID", "SepsisLabel"]
    X_test = test_df.drop(columns=drop_cols)

    # Use a sample for speed (SHAP can be slow on full datasets)
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot: shows which features matter most overall
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(f"C:\\sepsis proj\\shap_summary_{model_name}.png", dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot for {model_name}")

run_shap_analysis(r"C:\sepsis proj\full_model.pkl", r"C:\sepsis proj\full_test_features.csv", "full_data")
run_shap_analysis(r"C:\sepsis proj\vitals_model.pkl", r"C:\sepsis proj\vitals_test_features.csv", "vitals_only")
