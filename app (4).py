
import streamlit as st
import pandas as pd
import pickle

# Load trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Diabetes Prediction Using Logistic Regression")

st.write("Enter the patient information below:")

pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.warning("Model Prediction: Diabetes Positive")
    else:
        st.success("Model Prediction: Diabetes Negative")

    st.write(
        f"Predicted probability of positive class: {probability:.2%}"
    )

    st.caption(
        "This application demonstrates a machine-learning model and is not a medical diagnosis."
    )
