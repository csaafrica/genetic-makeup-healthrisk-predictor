import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


bio_data = pd.read_csv(r'C:\Users\ryanj\Desktop\CSA2025\data\anthropometric_trait_gwas.csv')

bmi_mean = bio_data['BMI'].mean()
bmi_std = bio_data['BMI'].std()

# --- Load model ---
model = joblib.load("bmi_rf_model.pkl")

# --- Define preprocessing rules ---
features_to_scale = [
    'LDL_cholesterol', 'HDL_cholesterol', 'systolic_BP',
    'diastolic_BP', 'BMI', 'age', 'weight', 'height',
    'waist_circumference', 'hip_circumference'
]

one_hot_col = 'sex'
drop_cols = ['cohort', 'sex']
all_snps = [f"SNP_{i}" for i in range(1, 1001)]  # SNP_1 to SNP_1000

# --- App Interface ---
st.title("BMI Predictor App")
st.markdown("Upload a CSV file with the required features to predict BMI.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read uploaded data
    input_df = pd.read_csv(uploaded_file)

    # Ensure SNP columns exist
    for snp in all_snps:
        if snp not in input_df.columns:
            input_df[snp] = 0  # Fill missing SNPs with 0

    # One-hot encode 'sex' column
    if one_hot_col in input_df.columns:
        encoded = pd.get_dummies(input_df[one_hot_col], drop_first=True)
        input_df = pd.concat([input_df, encoded], axis=1)

    # Drop unnecessary columns
    input_df.drop(columns=[col for col in drop_cols if col in input_df.columns], inplace=True)

    # Scale numeric features (temporary, just for model input)
    scaler = StandardScaler()
    for col in features_to_scale:
        if col in input_df.columns:
            input_df[col] = scaler.fit_transform(input_df[[col]])

    # Align columns to training order
    try:
        original_cols = joblib.load("training_columns.pkl")
        input_df = input_df.reindex(columns=original_cols, fill_value=0)
    except:
        st.warning("⚠️ Missing `training_columns.pkl`. Please save the training column order during training.")

    # --- Predict ---
    prediction = model.predict(input_df)

    # Inverse transform BMI from scaled value to original units
    full_scaler = joblib.load("full_scaler.pkl")
    #bmi_index = features_to_scale.index("BMI")
    #bmi_mean = full_scaler.mean_[bmi_index]
    #bmi_std = full_scaler.scale_[bmi_index]
    actual_bmi = (prediction * bmi_std) + bmi_mean

    print("Predicted BMI values:", actual_bmi)

    # --- Output BMI and Category ---
    st.subheader("Predicted BMI & Category")

    for i, bmi_value in enumerate(actual_bmi):
        if bmi_value < 18.5:
            category = "Underweight"
            st.warning(f"🤏Person {i+1} is in the **Underweight** category.")
        elif 18.5 <= bmi_value < 25:
            category = "Normal weight"
            st.success(f"💚 Person {i+1} is in the **Normal weight** category.")
        elif 25 <= bmi_value < 30:
            category = "Overweight"
            st.warning(f"⚠️ Person {i+1} is in the **Overweight** category.")
        else:
            category = "Obese"
            st.warning(f"🚨 Person {i+1} is in the **Obese** category.")

        st.markdown(f"**Person {i+1}**")
        st.write(f"Predicted BMI: **{bmi_value:.2f}**")
        st.write(f"BMI Category: **{category}**")
        st.markdown("---")