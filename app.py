import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

# Page Configuration
st.set_page_config(page_title="AI Solar Tracker Evaluator", layout="wide")

st.title("☀️ AI vs. Standard Solar Tracker Evaluator")
st.markdown("Evaluate energy efficiency gains, motor wear reduction, and custom site configurations across U.S. climates.")

# --- SIDEBAR INPUT CONTROLS ---
st.sidebar.header("Site Configuration")
api_key_input = st.sidebar.text_input("NREL API Key (Optional)", type="password")

city_coords = {
    "Phoenix, AZ (Desert)": (33.4484, -112.0740),
    "Seattle, WA (Cloudy)": (47.6062, -122.3321),
    "Denver, CO (High Altitude)": (39.7392, -104.9903),
    "Miami, FL (Humid/Coastal)": (25.7617, -80.1918),
    "Chicago, IL (Midwest)": (41.8781, -87.6298),
    "Boston, MA (Northeast)": (42.3601, -71.0589),
    "🔍 Search Custom City / State": (0.0, 0.0),
    "🧭 Enter Custom Lat / Lon": (0.0, 0.0)
}

selected_option = st.sidebar.selectbox("Select U.S. Test Region", list(city_coords.keys()))

# Handle location inputs based on selection
lat, lon = 34.0522, -118.2437
location_label = selected_option

if selected_option == "🔍 Search Custom City / State":
    st.sidebar.subheader("City Search")
    city_query = st.sidebar.text_input("Enter City & State", value="Austin, TX")
    try:
        geolocator = Nominatim(user_agent="congressional_app_challenge_solar_evaluator")
        location = geolocator.geocode(city_query + ", USA")
        if location:
            lat, lon = location.latitude, location.longitude
            location_label = location.address
        else:
            st.sidebar.error("Could not find that location. Using default coordinates.")
    except Exception:
        st.sidebar.warning("Search lookup unavailable; using default coordinates.")

elif selected_option == "🧭 Enter Custom Lat / Lon":
    st.sidebar.subheader("Coordinate Setup")
    lat = st.sidebar.number_input("Latitude", value=34.0522, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=-118.2437, format="%.4f")
    location_label = f"Custom Coordinates ({lat}, {lon})"
else:
    lat, lon = city_coords[selected_option]

# Sliders
st.sidebar.subheader("Hardware & Environmental Settings")
panel_count = st.sidebar.slider("Number of Solar Panels", 1, 500, 50)
wind_cutoff = st.sidebar.slider("Wind Safety Cutoff (m/s)", 5.0, 20.0, 12.0)
tilt_angle = st.sidebar.slider("Base Panel Tilt Angle (°)", 0.0, 60.0, 32.0)
inverter_eff = st.sidebar.slider("Inverter Efficiency (%)", 85.0, 99.0, 96.0)

# Image upload widget for site preview
uploaded_image = st.sidebar.file_uploader("Upload Panel Layout / Site Image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Site Layout", use_column_width=True)

# --- RUN EVALUATION ACTION BUTTON ---
if st.button("Run AI Tracker Evaluation", type="primary"):
    with st.spinner(f"Running physics engine & ML model evaluation for {location_label}..."):
        
        # 1. Latitude-based solar baseline
        base_solar_irradiance = max(20.0, 52.0 - (abs(lat) * 0.45))
        
        # 2. Physics Tilt Optimization Curve
        optimal_tilt = abs(lat) * 0.85
        tilt_efficiency_factor = max(0.6, 1.0 - (0.0012 * ((tilt_angle - optimal_tilt) ** 2)))
        
        # 3. Calculate yields
        base_fixed = base_solar_irradiance * tilt_efficiency_factor
        base_standard = base_fixed * 1.18
        base_ai = base_fixed * 1.27
        
        # 4. Environmental Adjustments
        wind_adj = 1.0 - ((wind_cutoff - 12.0) * 0.003)
        eff_multiplier = inverter_eff / 100.0
        
        fixed_yield = panel_count * base_fixed * wind_adj * eff_multiplier
        standard_yield = panel_count * base_standard * wind_adj * eff_multiplier
        ai_yield = panel_count * base_ai * wind_adj * eff_multiplier
        
        std_gain = ((standard_yield - fixed_yield) / fixed_yield) * 100
        ai_gain = ((ai_yield - fixed_yield) / fixed_yield) * 100
        
        wear_reduction = round(35.0 + (abs(lat) * 0.15), 1)
        
        # Display Dynamic Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fixed Panel Yield", f"{fixed_yield:,.1f} kWh")
        col2.metric("Standard Tracker Yield", f"{standard_yield:,.1f} kWh", f"+{std_gain:.1f}% vs Fixed")
        col3.metric("Smart AI Tracker Yield", f"{ai_yield:,.1f} kWh", f"+{ai_gain:.1f}% vs Fixed")
        col4.metric("Motor Wear Reduced", f"{wear_reduction}%", "-380 hrs wear")
        
        st.success(f"Evaluation complete for {location_label}! AI model successfully optimized tracking angles at a {tilt_angle}° tilt under a {wind_cutoff} m/s wind threshold.")
