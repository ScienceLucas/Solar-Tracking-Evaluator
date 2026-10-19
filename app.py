import streamlit as st
import pandas as pd
from utils import calculate_solar_physics, analyze_site_image
from geopy.geocoders import Nominatim

st.set_page_config(page_title="AI Solar Tracker & Site Evaluator", layout="wide")

st.title("☀️ AI Solar Tracker & Site Intelligence Auditor")
st.markdown("*Congressional App Challenge Project | Advanced GIS, Computer Vision & Financial ROI Simulator*")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🌍 Global GIS & Location Settings")
city_query = st.sidebar.text_input("Search U.S. City / Location", "Cornelius, NC")

lat, lon = 35.4858, -80.9038
location_label = "Cornelius, North Carolina"
try:
    geolocator = Nominatim(user_agent="solar_app_sidebar_v4")
    loc = geolocator.geocode(city_query + ", USA")
    if loc:
        lat = loc.latitude
        lon = loc.longitude
        location_label = loc.address
        st.sidebar.success(f"Pinned: {location_label[:35]}...")
except Exception:
    pass

st.sidebar.markdown("---")
panel_count = st.sidebar.number_input("Number of Panels in Array", min_value=1, max_value=500, value=12, help="Scales total array size and financial outputs.")
tilt_angle = st.sidebar.slider("Panel Tilt Angle (Degrees)", 0, 90, int(abs(lat)*0.85))
wind_cutoff = st.sidebar.slider("Wind Speed Cutoff Threshold (mph)", 5, 30, 15, help="Wind speed where trackers lock flat for safety.")
inverter_eff = st.sidebar.slider("Inverter Efficiency (%)", 85.0, 99.0, 96.0)

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["⚡ Tab 1: Physics & AI Tracking Simulation", "👁️ Tab 2: Site Image & Financial ROI Audit"])

with tab1:
    st.header("U.S. Location & Tracking Physics Simulator")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Simulation Parameters")
        st.info(f"Active Location: **{location_label}** (Lat: {lat:.2f}°)")
        st.markdown(f"- **Selected Tilt:** {tilt_angle}°")
        st.markdown(f"- **Wind Safety Cutoff:** {wind_cutoff} mph")
        st.markdown(f"- **Array Scale:** {panel_count} Panels")
        
        # Native Streamlit map preview
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df, zoom=5, use_container_width=True)

    with col2:
        st.subheader("Simulated Annual Output & Performance Gains")
        
        fixed_y, standard_y, ai_y = calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count)
        
        # Apply inverter efficiency multiplier
        eff_mult = inverter_eff / 100.0
        fixed_y *= eff_mult
        standard_y *= eff_mult
        ai_y *= eff_mult
        
        std_gain = ((standard_y - fixed_y) / fixed_y) * 100
        ai_gain = ((ai_y - fixed_y) / fixed_y) * 100
        wear_reduction = round(35.0 + (abs(lat) * 0.1), 1)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Fixed Tilt Yield", f"{fixed_y:,.0f} kWh")
        mcol2.metric("Standard Tracker", f"{standard_y:,.0f} kWh", f"+{std_gain:.1f}% vs Fixed")
        mcol3.metric("AI Smart Tracker", f"{ai_y:,.0f} kWh", f"+{ai_gain:.1f}% vs Fixed")

        wear_col1, wear_col2 = st.columns(2)
        wear_col1.metric("Estimated Motor Wear Reduced", f"{wear_reduction}%", "-380 hrs wear")
        wear_col2.metric("Additional Clean Energy Captured", f"{(ai_y - fixed_y):,.1f} kWh", "Net gain across array")

        chart_data = pd.DataFrame({
            "System Type": ["Fixed Tilt", "Standard Tracker", "AI Smart Tracker"],
            "Annual kWh Output": [fixed_y, standard_y, ai_y]
        })
        st.bar_chart(chart_data.set_index("System Type"))

with tab2:
    st.header("Automated Computer Vision & Financial ROI Audit")
    
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("1. Upload Roof Blueprint, Drone Photo, or CAD File", type=["png", "jpg", "jpeg", "webp"])
        manual_image_location = st.text_input("Specify Image Location (City, State or Address)", "Cornelius, NC", help="If your photo doesn't have GPS metadata, type the location here.")
        
        if uploaded_file:
            st.success(f"Successfully received: **{uploaded_file.name}**")
            st.image(uploaded_file, caption="Site Overview Preview", use_container_width=True)

    with col_b:
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
            with st.spinner("Running computer vision pixel analysis & financial modeling..."):
                report_output = analyze_site_image(uploaded_file, surface_type, site_notes, manual_image_location)
                st.markdown(report_output)
