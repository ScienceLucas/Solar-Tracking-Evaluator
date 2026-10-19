import streamlit as st
import pandas as pd
from utils import calculate_solar_physics, analyze_site_image
from geopy.geocoders import Nominatim

st.set_page_config(page_title="AI Solar Tracker & Site Evaluator", layout="wide")

st.title("☀️ AI Solar Tracker & Site Intelligence Auditor")
st.markdown("*Congressional App Challenge Project | Advanced GIS, Computer Vision & Financial ROI Simulator*")

tab1, tab2 = st.tabs(["⚡ Tab 1: Physics & AI Tracking Simulation", "👁️ Tab 2: Site Image & Financial ROI Audit"])

with tab1:
    st.header("U.S. Location & Tracking Physics Simulator")
    
    col1, col2 = st.columns(2)
    with col1:
        city_query = st.text_input("Search U.S. City / Location", "Cornelius, NC")
        lat = 35.4858
        lon = -80.9038
        try:
            geolocator = Nominatim(user_agent="solar_app_clean")
            loc = geolocator.geocode(city_query + ", USA")
            if loc:
                lat = loc.latitude
                lon = loc.longitude
                st.success(f"Located: {loc.address} (Lat: {lat:.2f}, Lon: {lon:.2f})")
        except Exception:
            pass

        tilt_angle = st.slider("Panel Tilt Angle (Degrees)", 0, 90, int(abs(lat)*0.85))
        wind_cutoff = st.slider("Wind Speed Cutoff Threshold (mph)", 5, 30, 15)
        panel_count = st.number_input("Number of Panels in Array", min=1, max=100, value=12)

    with col2:
        st.subheader("Simulated Annual Output & Motor Wear")
        fixed_y, standard_y, ai_y = calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count)
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Fixed Tilt Yield", f"{fixed_y:,.0f} kWh")
        mcol2.metric("Standard Tracker", f"{standard_y:,.0f} kWh", "+12%")
        mcol3.metric("AI Smart Tracker", f"{ai_y:,.0f} kWh", "+22%")
        
        chart_data = pd.DataFrame({
            "System Type": ["Fixed Tilt", "Standard Tracker", "AI Smart Tracker"],
            "Annual kWh Output": [fixed_y, standard_y, ai_y]
        })
        st.bar_chart(chart_data.set_index("System Type"))

with tab2:
    st.header("Automated Computer Vision & Financial Audit")
    
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("1. Upload Roof Blueprint, Drone Photo, or CAD File", type=["png", "jpg", "jpeg", "webp"])
        manual_image_location = st.text_input("Specify Image Location (City, State or Address)", "Cornelius, NC", help="If your photo doesn't have GPS metadata embedded, type the location where the photo was taken here.")
        
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
