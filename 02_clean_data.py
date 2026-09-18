import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\sepsis proj\combined_data.csv")

print(f"Original shape: {df.shape}")

clinically_important = ["Lactate"]  # sepsis ka sabse strong indicator, hamesha rakhna hai

missing_pct = df.isnull().sum() / len(df) * 100
cols_to_drop = missing_pct[missing_pct > 95].index.tolist()

cols_to_drop = [col for col in cols_to_drop if col not in clinically_important]

print(f"\nDropping columns (>95% missing, excluding clinically important): {cols_to_drop}")
df_full = df.drop(columns=cols_to_drop)

print(f"Shape after dropping high-missing columns: {df_full.shape}")


exclude_cols = ["PatientID", "Age", "Gender", "HospAdmTime", "ICULOS", "SepsisLabel", "Unit1", "Unit2"]
fill_cols = [col for col in df_full.columns if col not in exclude_cols]

df_full = df_full.sort_values(["PatientID", "ICULOS"])
df_full[fill_cols] = df_full.groupby("PatientID")[fill_cols].ffill()

print("\nMissing % after forward-fill:")
print((df_full[fill_cols].isnull().sum() / len(df_full) * 100).round(2))


for col in fill_cols:
    median_val = df_full[col].median()
    df_full[col] = df_full[col].fillna(median_val)

print("\nMissing % after median fill (should be 0 now):")
print((df_full[fill_cols].isnull().sum() / len(df_full) * 100).round(2))

df_full.to_csv(r"C:\sepsis proj\full_data_clean.csv", index=False)
print("\nSaved: full_data_clean.csv")

vitals_cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", 
                "Age", "Gender", "HospAdmTime", "ICULOS", "SepsisLabel", "PatientID"]

vitals_cols = [col for col in vitals_cols if col in df_full.columns]

df_vitals = df_full[vitals_cols]
df_vitals.to_csv(r"C:\sepsis proj\vitals_only_clean.csv", index=False)
print("Saved: vitals_only_clean.csv")

print(f"\nFull data columns: {list(df_full.columns)}")
print(f"Vitals-only columns: {list(df_vitals.columns)}")