import pandas as pd
import os
import glob

base_folder = r"C:\sepsis proj\Training Data-20260904T113657Z-1-001\Training Data"

# Har patient ID ko uske source (setA/setB) se map karo
source_map = {}

for file in glob.glob(os.path.join(base_folder, "training_setA", "*.psv")):
    patient_id = os.path.basename(file).replace(".psv", "")
    source_map[patient_id] = "setA"

for file in glob.glob(os.path.join(base_folder, "training_setB", "*.psv")):
    patient_id = os.path.basename(file).replace(".psv", "")
    source_map[patient_id] = "setB"

print(f"Total patients mapped: {len(source_map)}")

# Har file mein Source column add karo jahan zaroori hai
files_to_patch = [
    r"C:\sepsis proj\full_test_features.csv",
    r"C:\sepsis proj\full_train_features.csv",
    r"C:\sepsis proj\vitals_test_features.csv",
    r"C:\sepsis proj\vitals_train_features.csv",
]

for file_path in files_to_patch:
    df = pd.read_csv(file_path)
    df["Source"] = df["PatientID"].map(source_map)
    df.to_csv(file_path, index=False)
    print(f"Updated: {file_path} — Source column added")