import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

# Streamlit app
st.title("FWI Prediction App")
st.write("Enter the input parameters to predict")

# Input fields
Temperature = st.number_input("Temperature", value=20.0)
RH = st.number_input("Relative Humidity (RH)", value=50.0)
Ws = st.number_input("Wind Speed (Ws)", value=5.0)
Rain = st.number_input("Rain", value=0.0)
FFMC = st.number_input("FFMC", value=85.0)
DMC = st.number_input("DMC", value=50.0)
ISI = st.number_input("ISI", value=10.0)
Classes = st.selectbox("Classes", [0.0, 1.0])   # Example: 0 = low risk, 1 = high risk
Region = st.selectbox("Region", [1.0, 2.0, 3.0, 4.0]) # Example region codes

if st.button("Predict FWI"):
    # Prepare and scale input
    new_data_scaled = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    result = ridge_model.predict(new_data_scaled)

    st.success(f"{result[0]:.2f}")
