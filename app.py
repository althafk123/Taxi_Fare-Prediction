
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="🚕",
    layout="centered"
)

# Load the trained model
model_path = Path(__file__).parent / "taxi_fare_model.pkl"
model = joblib.load(model_path)

# App heading
st.title("🚕 Taxi Fare Prediction")
st.write("Machine Learning Based Taxi Fare Estimator")
st.divider()

st.subheader("Enter Your Trip Details")

# Input fields
distance = st.number_input(
    "Trip Distance (km)",
    min_value=0.1,
    max_value=500.0,
    value=10.0,
    step=0.5
)

duration = st.number_input(
    "Trip Duration (minutes)",
    min_value=1.0,
    max_value=1000.0,
    value=30.0
)

passengers = st.number_input(
    "Passenger Count",
    min_value=1,
    max_value=10,
    value=1,
    step=1
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Weekday", "Weekend"]
)

traffic = st.selectbox(
    "Traffic Conditions",
    ["Low", "Medium", "High"]
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rainy", "Snowy"]
)

st.divider()

# Prediction button
if st.button("Predict Taxi Fare", type="primary"):

    input_data = pd.DataFrame([{
        "Trip_Distance_km": distance,
        "Time_of_Day": time_of_day,
        "Day_of_Week": day_of_week,
        "Passenger_Count": passengers,
        "Traffic_Conditions": traffic,
        "Weather": weather,
        "Trip_Duration_Minutes": duration
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Taxi Fare: {prediction:.2f}")

    st.info(
        "This is an estimated fare based on your trained ML model. "
        "It is not a live taxi fare quotation."
    )
