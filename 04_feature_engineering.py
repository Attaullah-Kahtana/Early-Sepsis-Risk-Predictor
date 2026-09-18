import pandas as pd

def add_features(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df = df.sort_values(["PatientID", "ICULOS"])
    
    
    exclude_cols = ["PatientID", "Age", "Gender", "HospAdmTime", "ICULOS", "SepsisLabel", "Unit1", "Unit2"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Creating rolling features for: {feature_cols}")
    
    for col in feature_cols:
       
        df[f"{col}_rolling_mean"] = df.groupby("PatientID")[col].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        
        df[f"{col}_diff"] = df.groupby("PatientID")[col].transform(
            lambda x: x.diff().fillna(0)
        )
       
        df[f"{col}_rolling_std"] = df.groupby("PatientID")[col].transform(
            lambda x: x.rolling(window=6, min_periods=1).std().fillna(0)
        )
    
    print(f"New shape: {df.shape}")
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

add_features(
    r"C:\sepsis proj\full_train.csv",
    r"C:\sepsis proj\full_train_features.csv"
)
add_features(
    r"C:\sepsis proj\full_test.csv",
    r"C:\sepsis proj\full_test_features.csv"
)

add_features(
    r"C:\sepsis proj\vitals_train.csv",
    r"C:\sepsis proj\vitals_train_features.csv"
)
add_features(
    r"C:\sepsis proj\vitals_test.csv",
    r"C:\sepsis proj\vitals_test_features.csv"
)