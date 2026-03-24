import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # fallback
import sys

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("Heart Disease Risk Prediction")
st.markdown("Enter patient data to estimate the probability of heart disease.")

# Load the pre-trained pipeline
@st.cache_resource
def load_model():
    model_path = "cat_pipeline.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Define input fields based on the original dataset
st.sidebar.header("Patient Information")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", options=["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120, step=1)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200, step=1)

    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        resting_ecg = st.selectbox("Resting ECG Results", options=["Normal", "ST", "LVH"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150, step=1)
        exercise_angina = st.selectbox("Exercise Induced Angina", options=["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict")

# Prepare input data
if submitted:
    # Create a dictionary with the input values
    input_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    # Convert to DataFrame (order must match training features)
    df = pd.DataFrame([input_data])

    # Ensure column order matches the training data (the pipeline expects them in that order)
    # The original order (from the pickle metadata) is:
    expected_columns = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
    ]
    df = df[expected_columns]

    # Predict
    try:
        # The pipeline returns an array of predictions
        pred_proba = model.predict_proba(df)[0, 1]  # probability of class 1 (disease)
        pred_class = model.predict(df)[0]

        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("Probability of Heart Disease", f"{pred_proba:.2%}")
        col2.metric("Predicted Class", "High Risk" if pred_class == 1 else "Low Risk")

        # Additional visual feedback
        if pred_class == 1:
            st.error("⚠️ The model suggests a **high risk** of heart disease. Please consult a doctor.")
        else:
            st.success("✅ The model suggests a **low risk** of heart disease. Keep up the healthy lifestyle!")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Optional: Display model info
with st.expander("About this app"):
    st.markdown("""
    This app uses a **CatBoost** model trained on heart disease data.
    The model was prepared with a scikit-learn pipeline that handles missing values,
    categorical encoding, and feature scaling automatically.

    **Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)