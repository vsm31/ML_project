import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("UHI_effect.pkl")

scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌍 Urban Heat Island (UHI) Severity Prediction")
st.write("Enter environmental data to predict UHI severity level in Indian cities.")

# User inputs
city = st.selectbox("Select City", ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune'])
temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=50.0, value=35.0)
pop_density = st.number_input("Population Density (per sq km)", min_value=1000, max_value=50000, value=10000)
green_cover = st.slider("Green Cover (%)", min_value=0, max_value=100, value=20)
industrial_activity = st.selectbox("Industrial Activity Level", ['Low', 'Medium', 'High'])
vehicular_emissions = st.slider("Vehicular Emissions Index", min_value=0.0, max_value=500.0, value=150.0)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=60)
land_surface_temp = st.number_input("Land Surface Temperature (°C)", min_value=20.0, max_value=60.0, value=40.0)
nighttime_temp = st.number_input("Nighttime Temperature (°C)", min_value=10.0, max_value=40.0, value=30.0)

# Convert categorical inputs
city_dict = {'Mumbai': 0, 'Delhi': 1, 'Bangalore': 2, 'Chennai': 3, 'Pune': 4}
industrial_dict = {'Low': 0, 'Medium': 1, 'High': 2}

# Prepare input data
input_data = np.array([[
    temperature, pop_density, green_cover, 
    industrial_dict[industrial_activity], vehicular_emissions, humidity,
    land_surface_temp, nighttime_temp, city_dict[city]
]])

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict UHI Severity"):
    prediction = model.predict(input_data_scaled)[0]
    st.success(f"Predicted UHI Severity: {prediction}")
