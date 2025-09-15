import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd
import os

# Check if model and test data files exist
if not os.path.exists('powerconsumption_model.pkl'):
    st.error("Error: powerconsumption_model.pkl not found. Please run train_model.py first.")
    st.stop()
if not os.path.exists('test_data.pkl'):
    st.error("Error: test_data.pkl not found. Please run train_model.py first.")
    st.stop()

# Load the model and test data
try:
    model = joblib.load('powerconsumption_model.pkl')
    test_data = joblib.load('test_data.pkl')
except Exception as e:
    st.error(f"Error loading model or test data: {str(e)}")
    st.stop()

# Define dependent variables for labeling predictions
dependent_vars = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Hour', 'DayOfWeek', 'Month']

# Streamlit app
st.title("Power Consumption Prediction")

# Input fields
st.header("Enter Power Consumption Values")
col1, col2, col3 = st.columns(3)
with col1:
    zone1 = st.number_input("Power Consumption Zone 1 (kWh)", min_value=0.0, step=0.1, value=0.0)
with col2:
    zone2 = st.number_input("Power Consumption Zone 2 (kWh)", min_value=0.0, step=0.1, value=0.0)
with col3:
    zone3 = st.number_input("Power Consumption Zone 3 (kWh)", min_value=0.0, step=0.1, value=0.0)

# Predict button
if st.button("Predict"):
    try:
        # Prepare input for prediction
        input_data = np.array([[zone1, zone2, zone3]])
        prediction = model.predict(input_data)[0]
        
        # Display predictions
        st.header("Predictions")
        st.write(f"Temperature: {prediction[0]:.2f} °C")
        st.write(f"Humidity: {prediction[1]:.2f} %")
        st.write(f"Wind Speed: {prediction[2]:.2f} m/s")
        st.write(f"General Diffuse Flows: {prediction[3]:.2f} W/m²")
        st.write(f"Diffuse Flows: {prediction[4]:.2f} W/m²")
        st.write(f"Hour: {int(prediction[5])} (0-23)")
        st.write(f"Day of Week: {int(prediction[6])} (0=Mon, 6=Sun)")
        st.write(f"Month: {int(prediction[7])} (1=Jan, 12=Dec)")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

