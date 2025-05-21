import streamlit as st
import torch
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import joblib
from src.model import LSTMModel
import pandas as pd
import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
st.set_page_config(page_title="India 5G Bandwidth Predictor", page_icon="📡")
@st.cache_resource
def load_model_and_scaler():
    model = LSTMModel(input_size=7)
    model.load_state_dict(torch.load("models/lstm_model.pth"))
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()
geolocator = Nominatim(user_agent="my_bandwidth_app")

st.title("📡 India 5G Bandwidth Predictor")
option = st.radio("Choose Input Mode", ("📍 Enter Location Name", "🧭 Enter Latitude and Longitude"))

lat, lon = None, None

if option == "📍 Enter Location Name":
    place = st.text_input("Enter a city/place in India")
    if place:
        try:
            loc = geolocator.geocode(place + ", India", timeout=10)
            if loc:
                if "India" in loc.address:
                    lat, lon = loc.latitude, loc.longitude
                    st.success(f"📍 Found: {loc.address}")
                    st.write(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
                else:
                    st.error("❌ Invalid location: Please enter a place within India.")
            else:
                st.error("❌ Invalid location: Please enter a place within India.")
        except GeocoderTimedOut:
            st.error("Geocoder timed out. Please try again.")
else:
    lat = st.number_input("Latitude", value=23.0)
    lon = st.number_input("Longitude", value=80.0)

    # Reverse geocode to show location name
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        if location and "India" in location.address:
            st.success(f"📍 Approximate location: {location.address}")
        else:
            st.warning("⚠️ Could not determine a precise location or it's outside India.")
    except GeocoderTimedOut:
        st.error("Geocoder timed out. Please try again.")

# Define India's geographical bounds
INDIA_LAT_RANGE = (6.5, 37.0)
INDIA_LON_RANGE = (68.0, 97.0)

if lat is not None and lon is not None:
    if INDIA_LAT_RANGE[0] <= lat <= INDIA_LAT_RANGE[1] and INDIA_LON_RANGE[0] <= lon <= INDIA_LON_RANGE[1]:
        key = f"{lat:.4f}_{lon:.4f}"
        if "inputs" not in st.session_state:
            st.session_state.inputs = {}

        if key not in st.session_state.inputs:
            mobility = np.random.choice([0, 30, 60, 90])
            rssi = np.clip(np.random.normal(-85, 5), -120, -70)
            sinr = np.clip(np.random.normal(15, 3), 0, 30)
            rsrp = np.clip(np.random.normal(-95, 4), -120, -75)
            rsrq = np.clip(np.random.normal(-10, 2), -20, 0)
            st.session_state.inputs[key] = [mobility, rssi, sinr, rsrp, rsrq]

        mobility, rssi, sinr, rsrp, rsrq = st.session_state.inputs[key]
        feature_names = ["latitude", "longitude", "mobility", "RSSI", "SINR", "RSRP", "RSRQ", "bandwidth"]
        input_row = np.array([[lat, lon, mobility, rssi, sinr, rsrp, rsrq]])
        dummy_values = [*input_row[0], 0]  # Add dummy bandwidth
        dummy_df = pd.DataFrame([dummy_values], columns=feature_names)
        input_scaled_full = scaler.transform(dummy_df)
        input_scaled = input_scaled_full[0][:-1]
        sequence = np.array([input_scaled] * 10)
        input_tensor = torch.tensor(sequence).unsqueeze(0).float()

        with torch.no_grad():
            normalized_pred = model(input_tensor).item()
            full_scaled = input_scaled_full.copy()
            full_scaled[0, -1] = normalized_pred 

            predicted_bandwidth = scaler.inverse_transform(full_scaled)[0, -1]

            st.markdown("### 📶 Estimated 5G Bandwidth:")
            st.success(f"💡 {predicted_bandwidth:.2f} Mbps")
    else:
        st.error("❌ Invalid coordinates: Please enter a location within India's geographical bounds.")
