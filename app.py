"""
===============================================================================
PROJECT: AI Solar Tracker & Site Intelligence Auditor
MODULE: app.py (Streamlit Frontend UI & Interactive Execution Workspace)
DESCRIPTION: Main frontend application for the Congressional App Challenge submission.
             Houses U.S. simulation controls in sidebar batch forms, explicit 
             execution controls, live U.S. maps, temporal mode selectors, low-wind 
             warnings, and Tab 2 AI computer vision image classification.
===============================================================================
"""

import streamlit as st
import pandas as pd
from utils import calculate_solar_physics, analyze_site_image
from geopy.geocoders import Nominatim

# =============================================================================
# 1. PAGE CONFIGURATION & GLOBAL STYLING
# =============================================================================
st.set_page_config(
    page_title="AI Solar Tracker & Site Evaluator", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("☀️ AI Solar Tracker & Site Intelligence Auditor")
st.markdown("*Congressional App Challenge Project | Advanced U.S. GIS, Computer Vision & Financial ROI Simulator*")

# =============================================================================
# 2. NAVIGATION LAYOUT CONTROL
# =============================================================================
app_tab_selection = st.radio(
    "Navigation Mode", 
    ["⚡ Tab 1: U.S. Tracking Physics & Simulation", "👁️ Tab 2: AI Image Site & Panel Audit"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# =============================================================================
# 3. TAB 1: U.S. TRACKING PHYSICS & SIMULATION
# =============================================================================
if app_tab_selection == "⚡ Tab 1: U.S. Tracking Physics & Simulation":
    
    # --- Sidebar Input Controls ---
    st.sidebar.header("🌍 U.S. Simulation Controls")
    city_query_input = st.sidebar.text_input(
        "Search U.S. City / Location", 
        "Cornelius, NC", 
        help="Enter any U.S. city, state, or zip code for regional solar radiation mapping."
    )
    
    nrel_api_key_input = st.sidebar.text_input(
        "NREL API Key (Optional)", 
        value="DEMO_KEY", 
        type="password", 
        help="Enter your free API key from developer.nrel.gov."
    )

    simulation_mode = st.sidebar.selectbox(
        "Simulation Temporal Mode",
        ["Full Year Aggregate", "Seasonal / Quarterly Breakdown"],
        help="Switch between annual totals and detailed seasonal performance analysis across Spring, Summer, Autumn, and Winter."
    )

    # Default U.S. coordinates (Cornelius, NC)
    latitude_val, longitude_val = 35.4858, -80.9038
    active_location_label = "Cornelius, North Carolina"
    
    try:
        geolocator_sidebar = Nominatim(user_agent="solar_app_sidebar_enterprise_v100")
        geocoded_location_obj = geolocator_sidebar.geocode(city_query_input + ", USA")
        if geocoded_location_obj:
            latitude_val = geocoded_location_obj.latitude
            longitude_val = geocoded_location_obj.longitude
            active_location_label = geocoded_location_obj.address
            st.sidebar.success(f"U.S. Pinned: {active_location_label[:38]}...")
        else:
            st.sidebar.warning("Location not found in U.S. Using default (Cornelius, NC).")
    except Exception as geo_error:
        st.sidebar.debug(f"Geocoding error: {str(geo_error)}")

    st.sidebar.markdown("---")

    # Explicit Form Submission to prevent auto-rerunning on slider drag
    with st.sidebar.form(key="simulation_input_form"):
        st.markdown("##### 🎛️ Parameters Configuration")
        array_panel_count = st.number_input(
            "Number of Panels in Array", 
            min_value=1, 
            max_value=500, 
            value=60,
            help="Total count of photovoltaic modules in the simulated array."
        )
        configured_tilt_angle = st.slider(
            "Panel Tilt Angle / Fixed Tilt Surface (Degrees)", 
            0, 90, 31,
            help="Optimal tilt angle relative to the horizontal surface."
        )
        configured_wind_cutoff = st.slider(
            "Wind Speed Cutoff Threshold (mph)", 
            5, 30, 15,
            help="Wind speed at which trackers lock flat for mechanical safety."
        )
        configured_inverter_eff = st.slider(
            "Inverter Efficiency (%)", 
            85.0, 99.0, 96.0,
            help="DC-to-AC conversion efficiency rating."
        )

        st.markdown("---")
        run_simulation_button = st.form_submit_button(
            "🚀 Run Simulation Model", 
            type="primary", 
            use_container_width=True
        )

    if 'simulation_executed' not in st.session_state:
        st.session_state['simulation_executed'] = True

    if run_simulation_button:
        st.session_state['simulation_executed'] = True

    # --- Main Tab 1 Display Area ---
    st.header("U.S. Location & AI Tracking Physics Simulator")
    st.markdown(f"Active Mode: **{simulation_mode}**. Adjust your parameters in the sidebar form and click **'Run Simulation Model'** to update outputs.")
    
    col_map_view, col_results_view = st.columns([1, 1.2], gap="large")
    
    with col_map_view:
        st.subheader("🗺️ U.S. Region Map Pinpoint")
        interactive_map_df = pd.DataFrame({"lat": [latitude_val], "lon": [longitude_val]})
        st.map(interactive_map_df, zoom=5, use_container_width=True)
        st.info(f"Active Simulation Zone: **{active_location_label}** (Lat: {latitude_val:.2f}°)")

    with col_results_view:
        st.subheader("📊 Live Simulation Results & Performance Audit")
        
        if st.session_state['simulation_executed']:
            sim_results = calculate_solar_physics(
                lat=latitude_val, 
                lon=longitude_val, 
                tilt_angle=configured_tilt_angle, 
                wind_cutoff=configured_wind_cutoff, 
                panel_count=array_panel_count, 
                inverter_eff=configured_inverter_eff,
                nrel_api_key=nrel_api_key_input,
                simulation_mode=simulation_mode
            )
            
            if simulation_mode == "Full Year Aggregate":
                fixed_yield_val, standard_yield_val, ai_yield_val, motor_wear_reduction_pct, seasonal_dict, wind_cutoff_comparison_message, unoptimal_wind_warning = sim_results
                
                if unoptimal_wind_warning:
                    st.warning(f"⚠️ **Low Wind Speed Cutoff Warning ({configured_wind_cutoff} mph):**\n\n{unoptimal_wind_warning}")

                standard_gain_pct = ((standard_yield_val - fixed_yield_val) / fixed_yield_val) * 100 if fixed_yield_val > 0 else 0
                ai_gain_pct = ((ai_yield_val - fixed_yield_val) / fixed_yield_val) * 100 if fixed_yield_val > 0 else 0

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Fixed Tilt Yield", f"{fixed_yield_val:,.1f} kWh")
                metric_col2.metric("Standard Tracker", f"{standard_yield_val:,.1f} kWh", f"{standard_gain_pct:+.1f}% vs Fixed")
                metric_col3.metric("AI Smart Tracker", f"{ai_yield_val:,.1f} kWh", f"{ai_gain_pct:+.1f}% vs Fixed")

                sub_metric_col1, sub_metric_col2 = st.columns(2)
                sub_metric_col1.metric("Estimated Motor Wear Reduced", f"{motor_wear_reduction_pct}%", "Gear fatigue saved via AI")
                sub_metric_col2.metric("Additional Clean Energy", f"{(ai_yield_val - fixed_yield_val):,.1f} kWh", "Net gain across array")

                st.markdown("---")
                st.info(f"🌬️ **Wind Speed Cutoff Impact Analysis (Full Year Mode):**\n\n{wind_cutoff_comparison_message}")

                chart_dataframe = pd.DataFrame({
                    "System Architecture": ["1. Fixed Tilt", "2. Standard Tracker", "3. AI Smart Tracker"],
                    "Annual Energy Output (kWh)": [fixed_yield_val, standard_yield_val, ai_yield_val]
                }).set_index("System Architecture")
                
                st.bar_chart(chart_dataframe)
                
            else:
                fixed_yield_val, standard_yield_val, ai_yield_val, motor_wear_reduction_pct, seasonal_dict, wind_cutoff_comparison_message, unoptimal_wind_warning = sim_results
                
                if unoptimal_wind_warning:
                    st.warning(f"⚠️ **Low Wind Speed Cutoff Warning ({configured_wind_cutoff} mph):**\n\n{unoptimal_wind_warning}")

                st.markdown("##### 🍂 Quarterly Seasonal Energy Breakdown")
                
                seasonal_rows = []
                for season_name, yields in seasonal_dict.items():
                    seasonal_rows.append({
                        "Season": season_name,
                        "Fixed Tilt (kWh)": round(yields["Fixed"], 1),
                        "Standard Tracker (kWh)": round(yields["Standard"], 1),
                        "AI Smart Tracker (kWh)": round(yields["AI Tracker"], 1)
                    })
                
                seasonal_df = pd.DataFrame(seasonal_rows).set_index("Season")
                st.dataframe(seasonal_df, use_container_width=True)

                st.markdown("---")
                st.info(f"🍂 **Wind Speed Cutoff Impact Analysis (Seasonal Mode):**\n\n{wind_cutoff_comparison_message}")
                
                st.markdown("---")
                st.bar_chart(seasonal_df)
        else:
            st.info("👈 Modify your parameters in the sidebar form and click **'Run Simulation Model'** to generate results.")

# =============================================================================
# 4. TAB 2: AI IMAGE SITE & PANEL CLASSIFICATION AUDITOR
# =============================================================================
else:
    st.header("👁️ AI Image Site & Panel Classification Auditor")
    st.markdown("Upload aerial photos, drone captures, roof blueprints, or ground images. Verify panel presence and count below to ensure flawless mathematical energy modeling.")
    
    audit_col_left, audit_col_right = st.columns(2)
    
    with audit_col_left:
        uploaded_site_image_file = st.file_uploader(
            "1. Upload Site Image / Roof Blueprint", 
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload the visual asset you want the computer vision engine to scan."
        )
        
        manual_image_location_input = st.text_input(
            "Specify U.S. Image Location (City, State)", 
            "Cornelius, NC", 
            help="Enter the U.S. city where the image was captured for PSH and solar radiation calculations."
        )
        
        if uploaded_site_image_file:
            st.success(f"Successfully received: **{uploaded_site_image_file.name}**")
            st.image(uploaded_site_image_file, caption="Uploaded Site Asset Preview", use_container_width=True)

    with audit_col_right:
        st.markdown("##### 2. Essential Site & Terrain Information")
        
        surface_terrain_type = st.selectbox(
            "Installation Surface / Terrain Type",
            [
                "Pitched Residential Shingle", 
                "Flat Commercial Roof", 
                "Ground-Mounted Agricultural Field", 
                "Coastal / High Humidity Ground"
            ],
            help="Select the structural mounting surface type."
        )
        
        # Yes/No User Question with Clarifying Helper Text
        has_panels_choice = st.radio(
            "Are there existing solar panels on this site?",
            ["No", "Yes"],
            horizontal=True,
            help="(Helps ensure the AI is correct and prevents false classifications)"
        )
        
        # Mandatory Panel Count Input
        user_panel_count = st.number_input(
            "Total Number of Panels (Existing Count or Desired Target)", 
            min_value=1, 
            max_value=2000, 
            value=36,
            help="Enter the exact count for precise NREL energy output and financial ROI math."
        )
        
        user_site_notes = st.text_area(
            "General Optional Info (Shading, Obstructions, Notes)",
            value="Roof structure with minimal shading. Ready for high-efficiency deployment.",
            height=90,
            help="Optional notes on local shading or environmental conditions."
        )
        
        execute_audit_button = st.button(
            "🔍 Run AI Image Classification & Tailored Recommendation Audit", 
            use_container_width=True, 
            type="primary"
        )
        
        if execute_audit_button:
            if uploaded_site_image_file is None:
                st.warning("⚠️ Please upload an image in Step 1 before running the AI image audit.")
            elif user_panel_count <= 0:
                st.warning("⚠️ Please enter a valid panel count greater than 0 to proceed with mathematical modeling.")
            else:
                with st.spinner("AI model analyzing image pixels and calculating financial savings..."):
                    generated_audit_report = analyze_site_image(
                        uploaded_file=uploaded_site_image_file, 
                        surface_type=surface_terrain_type, 
                        has_panels_choice=has_panels_choice,
                        user_panel_count=user_panel_count,
                        site_notes=user_site_notes, 
                        manual_image_location=manual_image_location_input,
                        nrel_api_key="DEMO_KEY"
                    )
                    st.markdown(generated_audit_report)
                    st.markdown(generated_audit_report)
