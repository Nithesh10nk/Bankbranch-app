import streamlit as st
import numpy as np
import joblib


# Load saved model and scaler

model = joblib.load("bank_branch_rf_model.pkl")
scaler = joblib.load("scaler.pkl")


# App title

st.set_page_config(page_title="Bank Branch Performance", layout="centered")
st.title(" Bank Branch Performance Analysis")
st.write("Predict Monthly Revenue using Machine Learning")

# User Inputs 

staff = st.number_input("Staff Count", min_value=1, max_value=500, value=50)
area = st.number_input("Branch Area (sqft)", min_value=500, max_value=50000, value=3000)
marketing = st.number_input("Marketing Spend", min_value=0, value=200000)
atm = st.number_input("ATM Count", min_value=0, max_value=20, value=3)
population = st.number_input("Population Density", min_value=0, value=10000)
competitors = st.number_input("Competitor Branches", min_value=0, value=5)
income = st.number_input("Average Income", min_value=0, value=30000)

city_type = st.selectbox("City Type", ["Rural", "Urban", "SemiUrban"])

# Encode City Type (same as training) 

city_urban = 1 if city_type == "Urban" else 0
city_semi = 1 if city_type == "SemiUrban" else 0

#  Prediction 

if st.button("Predict Revenue"):
    input_data = np.array([[staff, area, marketing, atm,
                            population, competitors, income,
                            city_urban, city_semi]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f" Predicted Monthly Revenue: ₹ {prediction[0]:,.2f}")

    # Debug info (optional – can remove later)
    st.write("Input Shape:", input_data.shape)
