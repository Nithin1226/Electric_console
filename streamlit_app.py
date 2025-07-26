from datetime import datetime
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/forecast_model.pkl")

st.title("🔌 Electricity Consumption Forecaster")
st.write("Enter date and time to forecast electricity consumption:")

# 🛠️ Use date_input and time_input
selected_date = st.date_input("Select Date", value=datetime.now().date())
selected_time = st.time_input("Select Time", value=datetime.now().time())

# Combine into a full datetime
input_datetime = datetime.combine(selected_date, selected_time)

# Extract features
hour = input_datetime.hour
day = input_datetime.day
month = input_datetime.month
weekday = input_datetime.weekday()

features = pd.DataFrame([[hour, day, month, weekday]], columns=["hour", "day", "month", "weekday"])

# Predict
if st.button("Forecast Consumption"):
    prediction = model.predict(features)
    st.success(f"⚡ Predicted Consumption: {prediction[0]:.2f} kWh")
