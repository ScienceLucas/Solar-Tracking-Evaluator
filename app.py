"""
===============================================================================
PROJECT: AI Solar Tracker & Site Intelligence Auditor
MODULE: app.py (Streamlit Frontend UI & Interactive Execution Workspace)
DESCRIPTION: Houses U.S. simulation controls in sidebar batch forms, explicit 
             execution controls, live maps, temporal mode selectors, low-wind 
             warnings, and Tab 2 AI image classification with dynamic sidebar hiding.
===============================================================================
"""

import streamlit as st
import pandas as pd
from utils import calculate_solar_physics, analyze_site_image
from geopy.geocoders import Nominatim

# Page Configuration & Layout Setup
st.set_page_config(
    page_title="AI Solar Tracker & Site Evaluator", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("☀️ AI Solar Tracker & Site Intelligence Auditor")
st.markdown("*Congressional App Challenge Project | Advanced U.S. GIS, Computer Vision & Financial ROI Simulator*")

# --- PRIMARY APP TABS ---
tab_simulation, tab_audit = st.tabs(["⚡ Tab 1: U.S. Tracking Physics & Simulation", "👁️ Tab 2: AI Image Site & Panel Audit"])

# --- DYNAMIC SIDEBAR VISIBILITY CONTROL ---
# When Tab 2 is active, we inject CSS to hide the sidebar completely. When Tab 1 is active, we ensure it's visible.
# Note: Since st.tabs doesn't return an active index directly without state, 
# we control the sidebar rendering or inject CSS based on container context, 
# or use st.radio/st.selectbox if absolute tab-switch CSS binding is needed. 
# Alternatively, we can use a clean CSS state toggle or put the sidebar code strictly inside Tab 1's block!

# BEST STREAMLIT APPROACH: Render sidebar contents ONLY when on Tab 1, or hide via CSS if Tab 2 is selected.
# Let's use a session state or check tab selection by rendering the tabs and controlling sidebar inputs dynamically:

with tab_simulation:
    # --- SIDEBAR CONTROLS ENCLOSED IN A BATCH-SUBMISSION FORM (ONLY FOR TAB 1) ---
    st.sidebar.header("🌍 U.S. Simulation Controls")
    city_query_input = st.sidebar.text_input("Search U.S. City / Location", "Cornelius, NC", help="Enter any U.S. city, state, or zip code.")
    nrel_api_key_input = st.sidebar.text_input("NREL API Key (Optional)", value="DEMO_KEY", type="password", help="Enter your free API key from developer.nrel.gov.")

    simulation_mode = st.sidebar.selectbox(
        "Simulation Temporal Mode",
        ["Full Year Aggregate", "Seasonal / Quarterly Breakdown"],
        help="Switch between annual totals and detailed seasonal performance analysis across Spring, Summer, Autumn, and Winter."
    )

    latitude_val, longitude_val = 35.4858, -80.9038
    active_location_label = "Cornelius, North Carolina"
    try:
        geolocator_sidebar = Nominatim(user_agent="solar_app_sidebar_run_v18")
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

    with st.sidebar.form(key="simulation_input_form"):
        st.markdown("##### 🎛️ Parameters Configuration")
        array_panel_count = st.number_input("Number of Panels in Array", min_value=1, max_value=500, value=60)
        configured_tilt_angle = st.slider("Panel Tilt Angle / Fixed Tilt Surface (Degrees)", 0, 90, 31)
        configured_wind_cutoff = st.slider("Wind Speed Cutoff Threshold (mph)", 5, 30, 15)
        configured_inverter_eff = st.slider("Inverter Efficiency (%)", 85.0, 99.0, 96.0)

        st.markdown("---")
        run_simulation_button = st.form_submit_button("🚀 Run Simulation Model", type="primary", use_container_width=True)

    if 'simulation_executed' not in st.session_state:
        st.session_state['simulation_executed'] = True

    if run_simulation_button:
        st.session_state['simulation_executed'] = True

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

with tab_audit:
    # Hide sidebar via CSS injection when Tab 2 is active
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("👁️ AI Image Site & Panel Classification Auditor")
    st.markdown("Upload aerial photos, drone captures, roof blueprints, or ground images. Specify the **total** number of panels in the full system array (assuming identical modules if only partially pictured), log your local shading notes, and run the AI audit.")
    
    audit_col_left, audit_col_right = st.columns(2)
    with audit_col_left:
        uploaded_site_image_file = st.file_uploader("1. Upload Site Image / Roof Blueprint", type=["png", "jpg", "jpeg", "webp"])
        manual_image_location_input = st.text_input("Specify U.S. Image Location (City, State)", "Cornelius, NC", help="If your photo lacks EXIF GPS metadata, enter the U.S. city where it was captured.")
        
        total_array_panel_count_input = st.number_input(
            "Total Number of Panels in Full Array (Assumes Identical Scaling)", 
            min_value=1, max_value=500, value=60,
            help="Enter the total panel count for the complete installation. If the image only captures a partial view, the model samples the image and scales the output across this total count."
        )
        
        if uploaded_site_image_file:
            st.success(f"Successfully received: **{uploaded_site_image_file.name}**")
            st.image(uploaded_site_image_file, caption="Uploaded Site Asset Preview", use_container_width=True)

    with audit_col_right:
        st.markdown("##### 2. User Architecture & Environmental Input")
        user_declared_system_type = st.selectbox(
            "Current or Intended System Setup Type",
            ["Fixed Tilt Array", "Standard Single-Axis Tracker", "AI Smart Tracker", "Unknown / Evaluating New Site"]
        )
        
        surface_terrain_type = st.selectbox(
            "Installation Surface / Terrain Type",
            ["Flat Commercial Roof", "Pitched Residential Shingle", "Ground-Mounted Agricultural Field", "Coastal / High Humidity Ground"]
        )
        
        user_site_notes = st.text_area(
            "Local Shading & Physical Obstruction Notes",
            value="Commercial roof with minor tree shading along East edge. 15 mph average wind exposure.",
            height=120,
            help="Describe any obstructions, nearby trees, or shading. The AI model explicitly factors these notes into the audit."
        )
        
        execute_audit_button = st.button("🔍 Run AI Image Classification & Comparative Audit", use_container_width=True, type="primary")
        
        if execute_audit_button:
            if uploaded_site_image_file is None:
                st.warning("⚠️ Please upload an image in Step 1 before running the AI image audit.")
            else:
                with st.spinner("AI model analyzing pixel geometry, total array scaling, obstruction notes, and comparative architectures..."):
                    generated_audit_report = analyze_site_image(
                        uploaded_file=uploaded_site_image_file, 
                        surface_type=surface_terrain_type, 
                        site_notes=user_site_notes, 
                        user_system_type=user_declared_system_type,
                        total_panel_count=total_array_panel_count_input,
                        manual_image_location=manual_image_location_input,
                        nrel_api_key=nrel_api_key_input
                    )
                    st.markdown(generated_audit_report)
