import pandas as pd
import os
import glob


base_folder = r"C:\sepsis proj\Training Data-20260904T113657Z-1-001"

psv_files = glob.glob(os.path.join(base_folder, "training_setA", "*.psv")) + \
            glob.glob(os.path.join(base_folder, "training_setB", "*.psv"))


print(f"Total files found: {len(psv_files)}")

all_data = []

for file in psv_files:
    patient_id = os.path.basename(file).replace(".psv", "")
    df = pd.read_csv(file, sep="|")
    df["PatientID"] = patient_id
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

print(f"Combined dataframe shape: {combined_df.shape}")
print(f"Total unique patients: {combined_df['PatientID'].nunique()}")

sepsis_positive = combined_df[combined_df["SepsisLabel"] == 1]["PatientID"].nunique()
total_patients = combined_df["PatientID"].nunique()
print(f"Patients with sepsis at some point: {sepsis_positive} out of {total_patients} ({sepsis_positive/total_patients*100:.2f}%)")

print("\nMissing values per column (%):")
print((combined_df.isnull().sum() / len(combined_df) * 100).round(2))

combined_df.to_csv(r"C:\sepsis proj\combined_data.csv", index=False)
print("\nSaved combined data to combined_data.csv")