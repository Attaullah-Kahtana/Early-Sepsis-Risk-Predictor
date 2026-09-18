import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file, train_output, test_output):
    df = pd.read_csv(input_file)
    
    # Unique patient IDs 
    patient_ids = df["PatientID"].unique()
    
    # Patient-level split (80% train, 20% test)
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
   
    train_df = df[df["PatientID"].isin(train_ids)]
    test_df = df[df["PatientID"].isin(test_ids)]
    
    print(f"\n{input_file}:")
    print(f"Train patients: {len(train_ids)}, Train rows: {train_df.shape[0]}")
    print(f"Test patients: {len(test_ids)}, Test rows: {test_df.shape[0]}")
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"Saved: {train_output}, {test_output}")

# Full-data version split 
split_data(
    r"C:\sepsis proj\full_data_clean.csv",
    r"C:\sepsis proj\full_train.csv",
    r"C:\sepsis proj\full_test.csv"
)

# Vitals-only version split 
split_data(
    r"C:\sepsis proj\vitals_only_clean.csv",
    r"C:\sepsis proj\vitals_train.csv",
    r"C:\sepsis proj\vitals_test.csv"
)