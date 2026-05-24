import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="mkrish2025/Machine-Failure-Prediction", filename="best_mc_failure_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of machine failures and classifies whether an engine requires maintenance or is operating normally.
Please enter the machine sensor details below to get a prediction.
""")

EngineRPM = st.number_input("Engine RPM", min_value=60, max_value=2500, value=70)
LubOilPressure = st.number_input("Lub Oil Pressure", min_value=0.003, max_value=8, value=0.1,format="%.3f")
FuelPressure = st.number_input("Fuel Pressure", min_value=0.003, max_value=21, value=0.01,format="%.3f")
CoolantPressure = st.number_input("Coolant Pressure", min_value=0.002, max_value=8, value=0.2,format="%.3f")
LubOilTemperature = st.number_input("Lub Oil Temperature", min_value=70, max_value=90, value=86,format="%.3f")
CoolantTemperature = st.number_input("Coolant Temperature", min_value=70, max_value=200, value=100,format="%.3f")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "engine_rpm": EngineRPM,
    "lub_oil_pressure": LubOilPressure,
    "fuel_pressure": FuelPressure,
    "coolant_pressure": CoolantPressure,
    "lub_oil_temp": LubOilTemperature,
    "coolant_temp": CoolantTemperature

}])

st.subheader("Raw Input Data")
st.dataframe(input_data)

# Prediction
if st.button("Predict"):
    try:

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"❌ Machine is likely to fail and requires maintenance (Confidence: {prediction_proba:.2f})")
        else:
            st.warning(f"✅ Machine is likely in good condition  (Confidence: {1 - prediction_proba:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


