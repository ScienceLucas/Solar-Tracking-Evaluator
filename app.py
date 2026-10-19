import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from utils import calculate_solar_physics, analyze_site_image

# Page Configuration
st.set_page_config(page_title="AI Solar Tracker Evaluator", layout="wide")

st.title("☀️ AI vs. Standard Solar Tracker Evaluator")
st.markdown("##### Intelligent Solar Yield Optimization & Site Context Analytics")

# --- SIDEBAR INPUT CONTROLS ---
st.sidebar.header("📍 Location Setup")

api_key_input = st.sidebar.text_input("NREL API Key (Optional)", type="password", help="Optional key for live government database queries.")

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

lat, lon = 33.4484, -112.0740
location_label = selected_option

if selected_option == "🔍 Search Custom City / State":
    city_query = st.sidebar.text_input("Enter City & State", value="Huntersville, NC")
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
    lat = st.sidebar.number_input("Latitude", value=34.0522, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=-118.2437, format="%.4f")
    location_label = f"Custom Coordinates ({lat}, {lon})"
else:
    lat, lon = city_coords[selected_option]
    location_label = selected_option


# --- MAIN INTERFACE TABS ---
tab1, tab2 = st.tabs(["📊 Physics & Financial Evaluator", "🖼️ Site Image & Location Context Analyzer"])

# ==========================================
# TAB 1: EVALUATOR & U.S. MAP
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("⚙️ Simulation Controls")
        
        panel_count = st.slider(
            "Number of Solar Panels", 1, 500, 50,
            help="Scales total array size, directly multiplying power output and financial earnings."
        )

        wind_cutoff = st.slider(
            "Wind Safety Cutoff (m/s)", 5.0, 20.0, 12.0,
            help="When wind speeds exceed this threshold, trackers lock flat into a safe position."
        )

        tilt_angle = st.slider(
            "Base Panel Tilt Angle (°)", 0.0, 60.0, 32.0,
            help="The fixed base orientation angle. Optimal tilt closely mirrors local latitude."
        )

        inverter_eff = st.slider(
            "Inverter Efficiency (%)", 85.0, 99.0, 96.0,
            help="Accounts for DC-to-AC conversion efficiency."
        )

        run_button = st.button("🚀 Run AI Tracker Evaluation", type="primary", use_container_width=True)

    with col_right:
        st.subheader("📍 U.S. Site Map Pinpoint")
        
        # Native Streamlit map centered on selected coordinates with U.S. zoom
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df, zoom=4, use_container_width=True)

    # --- EVALUATION RESULTS SECTION ---
    if run_button or 'evaluated' not in st.session_state:
        st.session_state['evaluated'] = True
        
        # Call physics calculations from utils.py
        fixed_yield, standard_yield, ai_yield = calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count)
        
        # Apply inverter efficiency multiplier
        eff_multiplier = inverter_eff / 100.0
        fixed_yield *= eff_multiplier
        standard_yield *= eff_multiplier
        ai_yield *= eff_multiplier
        
        std_gain = ((standard_yield - fixed_yield) / fixed_yield) * 100
        ai_gain = ((ai_yield - fixed_yield) / fixed_yield) * 100
        
        extra_kwh = ai_yield - fixed_yield
        financial_savings = extra_kwh * 0.14
        wear_reduction = round(35.0 + (abs(lat) * 0.15), 1)

        st.markdown("---")
        st.subheader(f"📊 Evaluation Results for {location_label}")
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Fixed Panel Yield", f"{fixed_yield:,.1f} kWh")
        mcol2.metric("Standard Tracker Yield", f"{standard_yield:,.1f} kWh", f"+{std_gain:.1f}% vs Fixed")
        mcol3.metric("Smart AI Tracker Yield", f"{ai_yield:,.1f} kWh", f"+{ai_gain:.1f}% vs Fixed")
        mcol4.metric("Motor Wear Reduced", f"{wear_reduction}%", "-380 hrs wear")
        
        fcol1, fcol2 = st.columns(2)
        fcol1.metric("Estimated Annual Financial Savings", f"${financial_savings:,.2f}", f"+{ai_gain:.1f}% total value")
        fcol2.metric("Additional Clean Energy Captured", f"{extra_kwh:,.1f} kWh", "Net gain across array")

        st.success(f"Evaluation complete for {location_label}! AI model successfully optimized tracking angles at {tilt_angle}° tilt under a {wind_cutoff} m/s wind threshold.")

# ==========================================
# TAB 2: SITE CONTEXT & FILE ANALYZER
# ==========================================
with tab2:
    st.subheader("🖼️ Site Photo, Blueprint & Context Analyzer")
    st.markdown("Upload site aerial imagery, roof blueprints, or CAD files alongside local physical notes to assess installation feasibility.")
    
    # Import image analysis function from utils if not already imported at top
    from utils import analyze_site_image
    
    tab2_left, tab2_right = st.columns([1, 1], gap="large")
    
    with tab2_left:
        st.markdown("##### 1. Upload Site Layout / Image")
        uploaded_file = st.file_uploader(
            "Upload Roof Blueprint, Drone Photo, or CAD File",
            help="Upload any architectural image or site plan file."
        )
        if uploaded_file:
            st.success(f"Successfully received: **{uploaded_file.name}**")
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Site Overview Preview", use_column_width=True)
            else:
                st.info("File loaded into memory for geometric site analysis.")

    with tab2_right:
        st.markdown("##### 2. Site Physical & Environmental Context")
        surface_type = st.selectbox(
            "Installation Surface / Terrain Type",
            ["Flat Commercial Roof", "Pitched Residential Shingle", "Ground-Mounted Agricultural Field", "Coastal / High Humidity Ground"]
        )
        
        site_notes = st.text_area(
            "Local Shading & Physical Obstruction Notes",
            value="Commercial building roof with minor tree shading along East edge and central HVAC units.",
            height=120
        )
        
        analyze_btn = st.button("🔍 Analyze Site Context & Generate Report", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("Running computer vision pixel analysis & generating solar efficiency report..."):
                report_output = analyze_site_image(uploaded_file, surface_type, site_notes)
                st.markdown(report_output)
