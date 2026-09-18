import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Early Sepsis Risk Predictor", page_icon="🏥", layout="centered")

st.title("🏥 Early Sepsis Risk Predictor")
st.markdown("A machine learning tool to estimate sepsis risk from ICU patient vitals — built for both full-resource and resource-limited hospital settings.")

st.divider()

# Model selection
model_choice = st.radio(
    "Select model version:",
    ["Vitals-Only (No labs needed — resource-limited)", "Full-Data (Includes lab values)"]
)

st.subheader("Enter Patient Vitals")

col1, col2 = st.columns(2)
with col1:
    hr = st.number_input("Heart Rate (HR)", min_value=0.0, max_value=250.0, value=85.0)
    o2sat = st.number_input("Oxygen Saturation (O2Sat %)", min_value=0.0, max_value=100.0, value=97.0)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
    sbp = st.number_input("Systolic BP (SBP)", min_value=0.0, max_value=250.0, value=120.0)
with col2:
    map_val = st.number_input("Mean Arterial Pressure (MAP)", min_value=0.0, max_value=200.0, value=80.0)
    dbp = st.number_input("Diastolic BP (DBP)", min_value=0.0, max_value=150.0, value=75.0)
    resp = st.number_input("Respiratory Rate (Resp)", min_value=0.0, max_value=60.0, value=18.0)
    age = st.number_input("Age", min_value=0, max_value=120, value=50)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

hosp_adm_time = st.number_input("Hours since hospital admission", min_value=-500.0, max_value=0.0, value=-10.0)
iculos = st.number_input("Hours spent in ICU so far", min_value=0.0, max_value=500.0, value=5.0)

lab_values = {}
if "Full-Data" in model_choice:
    st.subheader("Lab Values")
    col3, col4 = st.columns(2)
    with col3:
        lab_values["BaseExcess"] = st.number_input("Base Excess", value=0.0)
        lab_values["FiO2"] = st.number_input("FiO2", value=0.21)
        lab_values["pH"] = st.number_input("pH", value=7.4)
        lab_values["PaCO2"] = st.number_input("PaCO2", value=40.0)
        lab_values["BUN"] = st.number_input("BUN", value=15.0)
        lab_values["Calcium"] = st.number_input("Calcium", value=9.0)
    with col4:
        lab_values["Chloride"] = st.number_input("Chloride", value=100.0)
        lab_values["Creatinine"] = st.number_input("Creatinine", value=1.0)
        lab_values["Glucose"] = st.number_input("Glucose", value=100.0)
        lab_values["Lactate"] = st.number_input("Lactate", value=1.5)
        lab_values["Magnesium"] = st.number_input("Magnesium", value=2.0)
        lab_values["Potassium"] = st.number_input("Potassium", value=4.0)
    lab_values["Hct"] = st.number_input("Hematocrit (Hct)", value=40.0)
    lab_values["Hgb"] = st.number_input("Hemoglobin (Hgb)", value=13.0)
    lab_values["WBC"] = st.number_input("WBC Count", value=8.0)
    lab_values["Platelets"] = st.number_input("Platelets", value=250.0)

st.divider()

if st.button("Predict Sepsis Risk", type="primary"):
    try:
        if "Vitals-Only" in model_choice:
            model = joblib.load("vitals_model.pkl")
            base_features = {
                "HR": hr, "O2Sat": o2sat, "Temp": temp, "SBP": sbp,
                "MAP": map_val, "DBP": dbp, "Resp": resp,
                "Age": age, "Gender": gender_val,
                "HospAdmTime": hosp_adm_time, "ICULOS": iculos
            }
        else:
            model = joblib.load("full_model.pkl")
            base_features = {
                "HR": hr, "O2Sat": o2sat, "Temp": temp, "SBP": sbp,
                "MAP": map_val, "DBP": dbp, "Resp": resp,
                **lab_values,
                "Age": age, "Gender": gender_val,
                "Unit1": 0, "Unit2": 0,
                "HospAdmTime": hosp_adm_time, "ICULOS": iculos
            }

        # Build a single-row dataframe matching model's expected feature order
        model_features = model.get_booster().feature_names
        input_df = pd.DataFrame([base_features])

        # Fill in any missing engineered/rolling features with the base value (best-effort approximation for single-point input)
        for col in model_features:
            if col not in input_df.columns:
                base_col = col.split("_rolling")[0].split("_diff")[0]
                input_df[col] = input_df[base_col] if base_col in input_df.columns else 0

        input_df = input_df[model_features]

        proba = model.predict_proba(input_df)[0, 1]
        risk_pct = proba * 100

        st.subheader("Result")
        if proba >= 0.40:
            st.error(f"⚠️ Elevated Sepsis Risk: {risk_pct:.1f}%")
            st.markdown("This patient's vitals pattern is consistent with elevated sepsis risk based on model training data. Clinical correlation and further monitoring recommended.")
        else:
            st.success(f"✅ Lower Sepsis Risk: {risk_pct:.1f}%")
            st.markdown("This patient's current vitals do not strongly match the sepsis risk pattern learned by the model.")

        st.caption("This is a research prototype trained on the PhysioNet/CinC 2019 Challenge dataset. It is not a diagnostic tool and should not be used for actual clinical decision-making.")

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Built as part of an early sepsis detection research project | Dataset: PhysioNet/CinC Challenge 2019")