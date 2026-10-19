import streamlit as st
import pandas as pd
from utils import calculate_solar_physics, analyze_site_image
from geopy.geocoders import Nominatim

st.set_page_config(page_title="AI Solar Tracker & Site Evaluator", layout="wide", initial_sidebar_state="expanded")

st.title("☀️ AI Solar Tracker & Site Intelligence Auditor")
st.markdown("*Congressional App Challenge Project | Advanced U.S. GIS, Computer Vision & Financial ROI Simulator*")

# --- SIDEBAR CONTROLS (DEDICATED EXCLUSIVELY TO TAB 1 SIMULATOR) ---
st.sidebar.header("🌍 U.S. Simulation Controls")
city_query_input = st.sidebar.text_input("Search U.S. City / Location", "Cornelius, NC", help="Enter any U.S. city, state, or zip code.")

latitude_val, longitude_val = 35.4858, -80.9038
active_location_label = "Cornelius, North Carolina"
try:
    geolocator_sidebar = Nominatim(user_agent="solar_app_sidebar_responsive")
    geocoded_location_obj = geolocator_sidebar.geocode(city_query_input + ", USA")
    if geocoded_location_obj:
        latitude_val = geocoded_location_obj.latitude
        longitude_val = geocoded_location_obj.longitude
        active_location_label = geocoded_location_obj.address
        st.sidebar.success(f"U.S. Pinned: {active_location_label[:38]}...")
    else:
        st.sidebar.warning("Location not found in U.S. Using default (Cornelius, NC).")
except Exception:
    pass

st.sidebar.markdown("---")
array_panel_count = st.sidebar.number_input("Number of Panels in Array", min_value=1, max_value=500, value=12, help="Scales total array wattage and financial output.")
configured_tilt_angle = st.sidebar.slider("Panel Tilt Angle (Degrees)", 0, 90, int(abs(latitude_val)*0.85), help="Fixed orientation angle relative to horizontal ground.")
configured_wind_cutoff = st.sidebar.slider("Wind Speed Cutoff Threshold (mph)", 5, 30, 15, help="Wind speed where tracking motors lock flat into a safe stowing position.")
configured_inverter_eff = st.sidebar.slider("Inverter Efficiency (%)", 85.0, 99.0, 96.0, help="Accounts for direct current to alternating current power conversion losses.")

# --- TABS ---
tab_simulation, tab_audit = st.tabs(["⚡ Tab 1: U.S. Tracking Physics & Simulation", "👁️ Tab 2: Site Image & Panel ROI Audit"])

with tab_simulation:
    st.header("U.S. Location & AI Tracking Physics Simulator")
    st.markdown("Adjust any slider in the sidebar to instantly observe how U.S. regional latitude, panel tilt angles, wind cutoff safety thresholds, and array sizing impact kilowatt-hour yields and mechanical motor wear.")
    
    col_map_view, col_results_view = st.columns([1, 1.2], gap="large")
    
    with col_map_view:
        st.subheader("🗺️ U.S. Region Map Pinpoint")
        interactive_map_df = pd.DataFrame({"lat": [latitude_val], "lon": [longitude_val]})
        st.map(interactive_map_df, zoom=5, use_container_width=True)
        st.info(f"Active Simulation Zone: **{active_location_label}** (Lat: {latitude_val:.2f}°)")

    with col_results_view:
        st.subheader("📊 Live Simulation Results & Performance Audit")
        
        # Execute physics simulation reactively with decimal precision
        fixed_yield_val, standard_yield_val, ai_yield_val = calculate_solar_physics(
            latitude_val, configured_tilt_angle, configured_wind_cutoff, array_panel_count, configured_inverter_eff
        )
        
        standard_gain_pct = ((standard_yield_val - fixed_yield_val) / fixed_yield_val) * 100 if fixed_yield_val > 0 else 0
        ai_gain_pct = ((ai_yield_val - fixed_yield_val) / fixed_yield_val) * 100 if fixed_yield_val > 0 else 0
        motor_wear_reduction_pct = round(32.0 + (abs(latitude_val) * 0.1) + max(0, (20 - configured_wind_cutoff) * 0.3), 1)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Fixed Tilt Yield", f"{fixed_yield_val:,.1f} kWh")
        metric_col2.metric("Standard Tracker", f"{standard_yield_val:,.1f} kWh", f"{standard_gain_pct:+.1f}% vs Fixed")
        metric_col3.metric("AI Smart Tracker", f"{ai_yield_val:,.1f} kWh", f"{ai_gain_pct:+.1f}% vs Fixed")

        sub_metric_col1, sub_metric_col2 = st.columns(2)
        sub_metric_col1.metric("Estimated Motor Wear Reduced", f"{motor_wear_reduction_pct}%", "Gear fatigue saved")
        sub_metric_col2.metric("Additional Clean Energy", f"{(ai_yield_val - fixed_yield_val):,.1f} kWh", "Net gain across array")

        st.markdown("---")
        chart_dataframe = pd.DataFrame({
            "System Architecture": ["Fixed Tilt", "Standard Tracker", "AI Smart Tracker"],
            "Annual Energy Output (kWh)": [fixed_yield_val, standard_yield_val, ai_yield_val]
        })
        st.bar_chart(chart_dataframe.set_index("System Architecture"))

with tab_audit:
    st.header("Automated Computer Vision & Panel ROI Audit")
    st.markdown("Upload site aerials, drone photos, or roof blueprints. The computer vision engine evaluates pixel luminosity and foliage density to rank the **Big 3 Panel Types** for energy gain and structural wear resilience, followed by specific engineering parameter checks.")
    
    audit_col_left, audit_col_right = st.columns(2)
    with audit_col_left:
        uploaded_site_image_file = st.file_uploader("1. Upload Roof Blueprint, Drone Photo, or CAD File", type=["png", "jpg", "jpeg", "webp"])
        manual_image_location_input = st.text_input("Specify U.S. Image Location (City, State)", "Cornelius, NC", help="If your photo lacks EXIF GPS metadata, enter the U.S. city where it was captured.")
        
        if uploaded_site_image_file:
            st.success(f"Successfully received: **{uploaded_site_image_file.name}**")
            st.image(uploaded_site_image_file, caption="Site Overview Preview", use_container_width=True)

    with audit_col_right:
        st.markdown("##### 2. Site Physical & Environmental Context")
        surface_terrain_type = st.selectbox(
            "Installation Surface / Terrain Type",
            ["Flat Commercial Roof", "Pitched Residential Shingle", "Ground-Mounted Agricultural Field", "Coastal / High Humidity Ground"]
        )
        
        user_site_notes = st.text_area(
            "Local Shading & Physical Obstruction Notes",
            value="Commercial roof with minor tree shading along East edge. 15 mph average wind exposure.",
            height=120
        )
        
        execute_audit_button = st.button("🔍 Analyze Site Context & Generate Report", use_container_width=True)
        
        if execute_audit_button:
            with st.spinner("Running computer vision pixel scan & Big 3 financial ROI modeling..."):
                generated_audit_report = analyze_site_image(uploaded_site_image_file, surface_terrain_type, user_site_notes, manual_image_location_input)
                st.markdown(generated_audit_report)
