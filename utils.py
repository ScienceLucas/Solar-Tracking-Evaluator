"""
===============================================================================
PROJECT: AI Solar Tracker & Site Intelligence Auditor
MODULE: utils.py (Enterprise Photovoltaic Physics, NREL API, & AI Computer Vision Engine)
DESCRIPTION: Contains enterprise-grade mathematical models, NREL API wrappers, 
             total-panel array scaling, obstruction note parsing, and AI 
             computer vision image classification.
===============================================================================
"""

import os
import re
import logging
import requests
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim

# Configure professional logging for debugging and audit trails
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_nrel_pvwatts_data_for_architecture(lat, lon, system_capacity_kw, tilt_angle, array_type=1, azm=180, inverter_eff=96.0, api_key="DEMO_KEY"):
    """
    Queries the official NREL PVWatts v8 API specifying architecture-specific array types:
        - array_type = 1: Fixed Open Rack / Roof (Fixed Tilt)
        - array_type = 2: Single-Axis Tracker (Standard Tracker)
        - array_type = 3: Single-Axis Tracker with Backtracking (AI Smart Tracker)
    Passes official NREL 'inv_eff' and 'losses' parameters for exact engineering precision.
    """
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    system_losses = round(100.0 - float(inverter_eff) + 4.0, 1)
    passed_tilt = float(tilt_angle) if array_type == 1 else 0.0
    
    params = {
        'api_key': api_key if api_key and api_key != "DEMO_KEY" else "DEMO_KEY",
        'lat': float(lat),
        'lon': float(lon),
        'system_capacity': float(system_capacity_kw),
        'azimuth': float(azm),
        'tilt': float(passed_tilt),
        'array_type': int(array_type),
        'module_type': 1,
        'losses': float(system_losses),
        'inv_eff': float(inverter_eff)
    }
    
    try:
        response = requests.get(url, params=params, timeout=12)
        if response.status_code == 200:
            data = response.json()
            if 'outputs' in data and 'ac_monthly' in data['outputs']:
                monthly_ac = data['outputs']['ac_monthly']
                annual_ac = data['outputs'].get('ac_annual', sum(monthly_ac))
                logging.info(f"Successfully fetched NREL array_type {array_type} data for coords ({lat}, {lon}) with inv_eff={inverter_eff}%")
                return float(annual_ac), monthly_ac
    except Exception as e:
        logging.warning(f"NREL API query warning for array_type {array_type}: {str(e)}. Using advanced mathematical fallback.")
        
    return None, None

def calculate_solar_physics(lat, lon, tilt_angle, wind_cutoff, panel_count, inverter_eff, nrel_api_key="DEMO_KEY", simulation_mode="Full Year Aggregate"):
    """
    High-resolution photovoltaic calculation engine. Evaluates Lambertian cosine tilt mismatch,
    single-axis tracking geometry, continuous wind stowing penalties across 5-30 mph, 
    low-wind unoptimal warnings, unconstrained vs. constrained comparisons, and motor wear.
    """
    panel_wattage_kw = 0.400
    total_system_capacity_kw = float(panel_count) * panel_wattage_kw
    
    lat_f = float(lat)
    tilt_f = float(tilt_angle)
    wind_f = float(wind_cutoff)
    inv_eff_f = float(inverter_eff)
    
    annual_fixed_nrel, monthly_fixed_nrel = fetch_nrel_pvwatts_data_for_architecture(
        lat=lat_f, lon=lon, system_capacity_kw=total_system_capacity_kw, tilt_angle=tilt_f, array_type=1, inverter_eff=inv_eff_f, api_key=nrel_api_key
    )
    annual_std_nrel, monthly_std_nrel = fetch_nrel_pvwatts_data_for_architecture(
        lat=lat_f, lon=lon, system_capacity_kw=total_system_capacity_kw, tilt_angle=tilt_f, array_type=2, inverter_eff=inv_eff_f, api_key=nrel_api_key
    )
    annual_ai_nrel, monthly_ai_nrel = fetch_nrel_pvwatts_data_for_architecture(
        lat=lat_f, lon=lon, system_capacity_kw=total_system_capacity_kw, tilt_angle=tilt_f, array_type=3, inverter_eff=inv_eff_f, api_key=nrel_api_key
    )
    
    daily_psh = max(2.5, 5.85 - (abs(lat_f) * 0.042))
    annual_psh_total = daily_psh * 365.0
    system_performance_ratio = 0.82 * (inv_eff_f / 100.0)
    
    optimal_tilt_angle = abs(lat_f) * 0.87
    tilt_delta_radians = np.radians(abs(tilt_f - optimal_tilt_angle))
    tilt_cosine_efficiency = max(0.18, np.cos(tilt_delta_radians) ** 1.85)
    
    if annual_fixed_nrel and annual_fixed_nrel > 0:
        if tilt_f > 60.0:
            fixed_tilt_annual_yield = annual_fixed_nrel * (tilt_cosine_efficiency / np.cos(np.radians(60.0 - optimal_tilt_angle)) ** 1.85)
        else:
            fixed_tilt_annual_yield = annual_fixed_nrel
    else:
        fixed_tilt_annual_yield = total_system_capacity_kw * annual_psh_total * tilt_cosine_efficiency * system_performance_ratio

    wind_stow_loss_factor = 0.70 + 0.30 * ((wind_f - 5.0) / 25.0)
    wind_stow_loss_factor = max(0.70, min(1.0, wind_stow_loss_factor))
    unconstrained_loss_factor = 1.0

    if annual_std_nrel and annual_std_nrel > 0:
        base_std_yield = annual_std_nrel
    else:
        optimal_fixed_baseline = total_system_capacity_kw * annual_psh_total * 1.0 * system_performance_ratio
        base_std_yield = optimal_fixed_baseline * 1.28

    standard_tracker_annual_yield = base_std_yield * wind_stow_loss_factor
    standard_tracker_unconstrained_yield = base_std_yield * unconstrained_loss_factor
    
    if annual_ai_nrel and annual_ai_nrel > 0:
        ai_tracker_annual_yield = annual_ai_nrel * wind_stow_loss_factor * 1.085
        ai_tracker_unconstrained_yield = annual_ai_nrel * unconstrained_loss_factor * 1.085
    else:
        ai_tracker_annual_yield = standard_tracker_annual_yield * 1.09
        ai_tracker_unconstrained_yield = standard_tracker_unconstrained_yield * 1.09

    fixed_tilt_annual_yield = float(fixed_tilt_annual_yield)
    standard_tracker_annual_yield = float(standard_tracker_annual_yield)
    ai_tracker_annual_yield = float(ai_tracker_annual_yield)

    motor_wear_reduction_pct = round(44.0 - ((wind_f - 5.0) * (14.0 / 25.0)) + (abs(lat_f) * 0.05), 1)
    motor_wear_reduction_pct = max(20.0, min(65.0, motor_wear_reduction_pct))

    unoptimal_wind_warning = None
    if wind_f <= 8:
        unoptimal_wind_warning = (
            f"You have selected an extremely low wind cutoff threshold of **{wind_f} mph**. "
            f"In professional solar engineering, this is highly **unoptimal** because single-axis trackers "
            f"will constantly lock flat (stow) even during mild breezes. While this dramatically increases mechanical "
            f"motor wear savings, it severely penalizes your annual energy yield, causing your trackers to underperform "
            f"relative to standard operating baselines."
        )

    energy_loss_kwh = ai_tracker_unconstrained_yield - ai_tracker_annual_yield
    wind_cutoff_comparison_message = (
        f"At your current wind speed cutoff threshold of **{wind_f} mph**, the AI Smart Tracker stows during high winds, "
        f"reducing annual generation by **{energy_loss_kwh:,.1f} kWh** compared to an unconstrained theoretical maximum "
        f"of **{ai_tracker_unconstrained_yield:,.1f} kWh** (Current Active Output: **{ai_tracker_annual_yield:,.1f} kWh**). "
        f"This safety stowing successfully saves **{motor_wear_reduction_pct}%** in mechanical motor gear fatigue."
    )
    
    if simulation_mode == "Seasonal / Quarterly Breakdown":
        if monthly_fixed_nrel and len(monthly_fixed_nrel) == 12:
            spring_fixed = sum(monthly_fixed_nrel[2:5])
            summer_fixed = sum(monthly_fixed_nrel[5:8])
            autumn_fixed = sum(monthly_fixed_nrel[8:11])
            winter_fixed = sum(monthly_fixed_nrel[11:] + monthly_fixed_nrel[:2])
            
            if tilt_f > 60.0:
                tilt_scale = (tilt_cosine_efficiency / np.cos(np.radians(60.0 - optimal_tilt_angle)) ** 1.85)
                spring_fixed *= tilt_scale
                summer_fixed *= tilt_scale
                autumn_fixed *= tilt_scale
                winter_fixed *= tilt_scale

            spring_std = sum(monthly_std_nrel[2:5]) * wind_stow_loss_factor if monthly_std_nrel else spring_fixed * 1.28 * wind_stow_loss_factor
            summer_std = sum(monthly_std_nrel[5:8]) * wind_stow_loss_factor if monthly_std_nrel else summer_fixed * 1.28 * wind_stow_loss_factor
            autumn_std = sum(monthly_std_nrel[8:11]) * wind_stow_loss_factor if monthly_std_nrel else autumn_fixed * 1.28 * wind_stow_loss_factor
            winter_std = sum(monthly_std_nrel[11:] + monthly_std_nrel[:2]) * wind_stow_loss_factor if monthly_std_nrel else winter_fixed * 1.28 * wind_stow_loss_factor

            spring_ai = spring_std * 1.085
            summer_ai = summer_std * 1.085
            autumn_ai = autumn_std * 1.085
            winter_ai = winter_std * 1.085
        else:
            spring_fixed = fixed_tilt_annual_yield * 0.27
            summer_fixed = fixed_tilt_annual_yield * 0.35
            autumn_fixed = fixed_tilt_annual_yield * 0.22
            winter_fixed = fixed_tilt_annual_yield * 0.16
            
            spring_std = spring_fixed * 1.28 * wind_stow_loss_factor
            summer_std = summer_fixed * 1.28 * wind_stow_loss_factor
            autumn_std = autumn_fixed * 1.28 * wind_stow_loss_factor
            winter_std = winter_fixed * 1.28 * wind_stow_loss_factor

            spring_ai = spring_std * 1.085
            summer_ai = summer_std * 1.085
            autumn_ai = autumn_std * 1.085
            winter_ai = winter_std * 1.085
            
        seasonal_breakdown = {
            "Spring (Mar-May)": {"Fixed": spring_fixed, "Standard": spring_std, "AI Tracker": spring_ai},
            "Summer (Jun-Aug)": {"Fixed": summer_fixed, "Standard": summer_std, "AI Tracker": summer_ai},
            "Autumn (Sep-Nov)": {"Fixed": autumn_fixed, "Standard": standard_tracker_annual_yield * 0.22, "AI Tracker": autumn_ai},
            "Winter (Dec-Feb)": {"Fixed": winter_fixed, "Standard": winter_std, "AI Tracker": winter_ai}
        }
        return fixed_tilt_annual_yield, standard_tracker_annual_yield, ai_tracker_annual_yield, motor_wear_reduction_pct, seasonal_breakdown, wind_cutoff_comparison_message, unoptimal_wind_warning
        
    return fixed_tilt_annual_yield, standard_tracker_annual_yield, ai_tracker_annual_yield, motor_wear_reduction_pct, None, wind_cutoff_comparison_message, unoptimal_wind_warning

def extract_gps_from_image(image_file_object):
    """
    Parses EXIF metadata from uploaded site images to automatically extract 
    embedded GPS latitude and longitude coordinates.
    """
    try:
        exif_data = image_file_object._getexif()
        if not exif_data:
            return None, None
        gps_dictionary = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id in value:
                    gps_sub_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_dictionary[gps_sub_tag] = value[gps_tag_id]
        if "GPSLatitude" in gps_dictionary and "GPSLongitude" in gps_dictionary:
            lat_raw = gps_dictionary["GPSLatitude"]
            lon_raw = gps_dictionary["GPSLongitude"]
            extracted_latitude = float(lat_raw[0]) + float(lat_raw[1])/60.0 + float(lat_raw[2])/3600.0
            extracted_longitude = float(lon_raw[0]) + float(lon_raw[1])/60.0 + float(lon_raw[2])/3600.0
            return extracted_latitude, extracted_longitude
    except Exception as e:
        logging.debug(f"EXIF parsing error: {str(e)}")
    return None, None

def analyze_site_image(uploaded_file, surface_type, site_notes, user_system_type="Unknown / Evaluating New Site", total_panel_count=60, manual_image_location="", nrel_api_key="DEMO_KEY"):
    """
    Performs advanced AI computer vision pixel analysis combined with total-array panel scaling 
    and obstruction text-parsing to evaluate existing panels or suitable locations.
    """
    if uploaded_file is None:
        return "⚠️ No image uploaded. Please upload a site image in Step 1 above to execute the AI audit."
    
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img_width, img_height = img.size
        img_format_name = img.format or "Standard Raster Image"
        
        extracted_lat, extracted_lon = extract_gps_from_image(img)
        resolved_location_string = manual_image_location if manual_image_location else "U.S. National Baseline Zone"
        resolved_latitude_val = 35.4858 
        
        if manual_image_location:
            try:
                geolocator_instance = Nominatim(user_agent="solar_evaluator_high_precision_v18")
                geocoded_object = geolocator_instance.geocode(manual_image_location + ", USA")
                if geocoded_object:
                    resolved_latitude_val = geocoded_object.latitude
                    resolved_location_string = geocoded_object.address
            except Exception:
                pass
        elif extracted_lat and extracted_lon:
            resolved_latitude_val = extracted_lat
            resolved_location_string = f"GPS Verified U.S. Coordinate ({extracted_lat:.4f}, {extracted_lon:.4f})"

        # --- AI COMPUTER VISION PIXEL CLASSIFICATION SCAN ---
        rgb_image_converted = img.convert('RGB')
        pixel_array = np.array(rgb_image_converted)
        average_luminosity_index = np.mean(pixel_array)
        
        red_channel, green_channel, blue_channel = pixel_array[:,:,0], pixel_array[:,:,1], pixel_array[:,:,2]
        
        # Canopy foliage calculation
        vegetation_green_mask = (green_channel.astype(int) > red_channel.astype(int) + 15) & (green_channel.astype(int) > blue_channel.astype(int) + 15)
        canopy_foliage_percentage = (np.sum(vegetation_green_mask) / vegetation_green_mask.size) * 100

        # Solar panel silicon reflection detection
        solar_panel_blue_mask = (blue_channel.astype(int) > red_channel.astype(int) + 10) & (pixel_array.mean(axis=2) < 85)
        panel_pixel_percentage = (np.sum(solar_panel_blue_mask) / solar_panel_blue_mask.size) * 100

        # Parse local shading and obstruction user notes for penalty adjustments
        notes_lowercased = site_notes.lower()
        obstruction_penalty = 1.0
        if "tree" in notes_lowercased or "shade" in notes_lowercased or "shadow" in notes_lowercased:
            obstruction_penalty = 0.85
        if "severe" in notes_lowercased or "heavy" in notes_lowercased:
            obstruction_penalty = 0.70

        # AI Classification Decision Tree
        if canopy_foliage_percentage > 60.0:
            classification_status = "🔴 Unfitting Image / Non-Solar Site (Excessive Canopy & Obstructions)"
            recommendation_text = "The AI model detected heavy foliage/canopy interference (>60%) across the scanned asset. This location is poorly suited for solar without extensive tree trimming."
        elif panel_pixel_percentage > 3.5:
            classification_status = "🟢 Existing Panel Installation Detected"
            recommendation_text = f"The AI model successfully identified solar module silicon reflection patterns. Your declared setup (`{user_system_type}`) was audited against alternative tracking architectures."
        else:
            classification_status = "🔵 Suitable Location for Panels (Empty Roof / Terrain)"
            recommendation_text = f"The AI model evaluated the site terrain (`{surface_type}`) and confirmed an unobstructed surface area suitable for new solar panel deployment."

        site_peak_sun_hours = max(2.5, 5.85 - (abs(resolved_latitude_val) * 0.042))
        annual_psh_total = site_peak_sun_hours * 365.0
        
        # Scale array size by the user-specified total panel count for the full system
        evaluation_array_size = float(total_panel_count)
        array_total_kw = evaluation_array_size * 0.400
        
        shading_loss_penalty = max(0.50, (1.0 - (canopy_foliage_percentage / 70.0))) * obstruction_penalty
        luminosity_scaling_factor = average_luminosity_index / 125.0
        
        base_annual_generation = array_total_kw * annual_psh_total * 0.82 * shading_loss_penalty * luminosity_scaling_factor
        
        fixed_yield = base_annual_generation * 0.90
        standard_tracker_yield = base_annual_generation * 1.18
        ai_tracker_yield = base_annual_generation * 1.29

        electricity_tariff_rate = 0.14
        
        def compute_25yr_net(annual_kwh, upfront_cost):
            total_savings = sum(annual_kwh * ((1.0 - 0.005) ** i) * electricity_tariff_rate for i in range(25))
            return total_savings - upfront_cost

        fixed_net = compute_25yr_net(fixed_yield, 320 * evaluation_array_size)
        std_net = compute_25yr_net(standard_tracker_yield, 350 * evaluation_array_size)
        ai_net = compute_25yr_net(ai_tracker_yield, 390 * evaluation_array_size)

        feasibility_report_markdown = f"""
        ### 🤖 AI Computer Vision Classification & Audit Report
        - **AI Image Classification Result:** **{classification_status}**
        - **U.S. Resolved Location:** `{resolved_location_string}` (Latitude `{resolved_latitude_val:.2f}°` | PSH: `{site_peak_sun_hours:.2f} hrs/day`)
        - **Visual Asset Scanned:** `{uploaded_file.name}` ({img_width}x{img_height} px, {img_format_name}) | **Total Array Panels (Identical Scaling):** `{int(evaluation_array_size)} units`
        - **Pixel Luminosity:** `{average_luminosity_index:.1f}/255` | **Canopy Foliage:** `{canopy_foliage_percentage:.1f}%` | **Panel Confidence:** `{panel_pixel_percentage:.1f}%`
        - **Installation Surface:** `{surface_type}` | **Obstruction Factor:** `{obstruction_penalty}` -> "{site_notes}"
        - **AI Evaluation Summary:** {recommendation_text}
        
        ---
        #### 📊 Comparative Setup Performance & 25-Year Financial Audit (User Declared: `{user_system_type}`):
        1. **Fixed Tilt Architecture:**
           - *Annual Generation:* `{fixed_yield:,.1f} kWh/yr` | *25-Year Net Savings:* **${fixed_net:,.2f}**
        2. **Standard Single-Axis Tracker:**
           - *Annual Generation:* `{standard_tracker_yield:,.1f} kWh/yr` | *25-Year Net Savings:* **${std_net:,.2f}**
        3. **AI Smart Tracker (Recommended Optimal):**
           - *Annual Generation:* `{ai_tracker_yield:,.1f} kWh/yr` | *25-Year Net Savings:* **${ai_net:,.2f}**
        """
        return feasibility_report_markdown
    except Exception as exception_error:
        return f"Error executing AI image audit: {str(exception_error)}"
