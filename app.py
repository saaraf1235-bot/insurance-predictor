import joblib
import streamlit as st
import pandas as pd
import numpy as np

model = joblib.load("insurance_model.pkl")
scaler = joblib.load("insurance_scaler.pkl")
columns = joblib.load("insurance_columns.pkl")

st.set_page_config(page_title="Insurance Charge Predictor", layout="centered")
st.title(" Medical Insurance Charge Predictor")
st.write("Enter details to predict annual insurance charges")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 65, 30)
    bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
    children = st.number_input("No. of Children", 0, 5, 0)
with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Charges"):
    input_dict = {
        'age': age, 'bmi': bmi, 'children': children,
        'sex_male': 1 if sex == "male" else 0,
        'smoker_yes': 1 if smoker == "yes" else 0,
        'region_northwest': 1 if region == "northwest" else 0,
        'region_southeast': 1 if region == "southeast" else 0,
        'region_southwest': 1 if region == "southwest" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"**Estimated Annual Insurance Charges: ${prediction:,.2f}**")