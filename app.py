import streamlit as st
import pandas as pd
from utils import calculate_solar_physics, analyze_site_image
from geopy.geocoders import Nominatim

st.set_page_config(page_title="AI Solar Tracker & Site Evaluator", layout="wide")

st.title("☀️ AI Solar Tracker & Site Intelligence Auditor")
st.markdown("*Congressional App Challenge Project | Advanced U.S. GIS, Computer Vision & Financial ROI Simulator*")

# --- CLEAN TABBED INTERFACE (NO SIDEBAR CLUTTER ON TAB 2) ---
tab1, tab2 = st.tabs(["⚡ Tab 1: U.S. Tracking Physics & Simulation", "👁️ Tab 2: Site Image & Panel ROI Audit"])

with tab1:
    st.header("U.S. Location & AI Tracking Physics Simulator")
    st.markdown("Test how U.S. regional latitude, panel tilt angles, wind cutoff safety thresholds, and array sizing impact annual kilowatt-hour yields and mechanical motor wear.")
    
    col_ctrl, col_res = st.columns([1, 1.2], gap="large")
    
    with col_ctrl:
        st.subheader("🎛️ U.S. Simulation Controls")
        city_query = st.text_input("Search U.S. City / Location", "Cornelius, NC", help="Enter any U.S. city, state, or zip code.")
        
        lat, lon = 35.4858, -80.9038
        location_label = "Cornelius, North Carolina"
        try:
            geolocator = Nominatim(user_agent="solar_app_us_tab1_v5")
            loc = geolocator.geocode(city_query + ", USA")
            if loc:
                lat = loc.latitude
                lon = loc.longitude
                location_label = loc.address
                st.success(f"U.S. Pinned: {location_label[:45]}...")
            else:
                st.warning("Location not found in U.S. Using default (Cornelius, NC).")
        except Exception:
            pass

        panel_count = st.number_input("Number of Panels in Array", min_value=1, max_value=500, value=12, help="Scales total array size and financial output.")
        tilt_angle = st.slider("Panel Tilt Angle (Degrees)", 0, 90, int(abs(lat)*0.85), help="Fixed orientation angle. Optimal tilt closely mirrors local U.S. latitude.")
        wind_cutoff = st.slider("Wind Speed Cutoff Threshold (mph)", 5, 30, 15, help="Wind speed where tracking motors lock flat into a safe position to prevent structural fatigue.")
        inverter_eff = st.slider("Inverter Efficiency (%)", 85.0, 99.0, 96.0, help="Accounts for DC-to-AC conversion efficiency.")
        
        st.markdown("---")
        st.subheader("🗺️ U.S. Region Map Pinpoint")
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df, zoom=5, use_container_width=True)

    with col_res:
        st.subheader("📊 Live Simulation Results & Performance Audit")
        
        # Execute rigorous physics calculations responding directly to all controls
        fixed_y, standard_y, ai_y = calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count, inverter_eff)
        
        std_gain = ((standard_y - fixed_y) / fixed_y) * 100 if fixed_y > 0 else 0
        ai_gain = ((ai_y - fixed_y) / fixed_y) * 100 if fixed_y > 0 else 0
        wear_reduction = round(34.0 + (abs(lat) * 0.12) + max(0, (20 - wind_cutoff) * 0.4), 1)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Fixed Tilt Yield", f"{fixed_y:,.0f} kWh")
        mcol2.metric("Standard Tracker", f"{standard_y:,.0f} kWh", f"+{std_gain:.1f}% vs Fixed")
        mcol3.metric("AI Smart Tracker", f"{ai_y:,.0f} kWh", f"+{ai_gain:.1f}% vs Fixed")

        wear_col1, wear_col2 = st.columns(2)
        wear_col1.metric("Estimated Motor Wear Reduced", f"{wear_reduction}%", "Gear fatigue saved")
        wear_col2.metric("Additional Clean Energy", f"{(ai_y - fixed_y):,.1f} kWh", "Net gain across array")

        st.markdown("---")
        chart_data = pd.DataFrame({
            "System Type": ["Fixed Tilt", "Standard Tracker", "AI Smart Tracker"],
            "Annual kWh Output": [fixed_y, standard_y, ai_y]
        })
        st.bar_chart(chart_data.set_index("System Type"))
        
        st.info(f"💡 **Physics Verification:** Results are dynamically computed for **{location_label}** (Lat: {lat:.2f}°) with a {tilt_angle}° tilt and {wind_cutoff} mph wind safety cutoff.")

with tab2:
    st.header("Automated Computer Vision & Panel ROI Audit")
    st.markdown("Upload site aerials, drone photos, or roof blueprints. The computer vision engine evaluates pixel luminosity and foliage density to rank the **Big 3 Panel Types** for energy gain and structural wear resilience.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("1. Upload Roof Blueprint, Drone Photo, or CAD File", type=["png", "jpg", "jpeg", "webp"])
        manual_image_location = st.text_input("Specify U.S. Image Location (City, State)", "Cornelius, NC", help="If your photo lacks EXIF GPS metadata, enter the U.S. city where it was captured.")
        
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
            value="Commercial roof with minor tree shading along East edge. 15 mph average wind exposure.",
            height=120
        )
        
        analyze_btn = st.button("🔍 Analyze Site Context & Generate Report", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("Running computer vision pixel scan & Big 3 financial ROI modeling..."):
                report_output = analyze_site_image(uploaded_file, surface_type, site_notes, manual_image_location)
                st.markdown(report_output)
