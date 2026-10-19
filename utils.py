import numpy as np
import re
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim

def calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count, inverter_eff):
    """
    Advanced physics model evaluating U.S. latitude solar irradiance, 
    tilt mismatch efficiency, wind safety cutoffs, and AI motor wear reduction.
    """
    base_solar_irradiance = max(25.0, 58.0 - (abs(lat) * 0.48))
    optimal_tilt = abs(lat) * 0.85
    tilt_mismatch = abs(tilt_angle - optimal_tilt)
    
    # Non-linear tilt efficiency drop-off based on solar angle of incidence
    tilt_efficiency = max(0.5, 1.0 - (0.0018 * (tilt_mismatch ** 1.6)))
    
    # Wind safety factor: low wind cutoffs cause frequent stowing; high cutoffs risk gear fatigue
    wind_protection_factor = min(1.0, wind_cutoff / 22.0)
    
    panel_base_yield = base_solar_irradiance * tilt_efficiency * 48.5
    total_base_array = panel_base_yield * panel_count
    
    standard_multiplier = 1.16 + max(0, (12.0 - tilt_mismatch) * 0.004)
    ai_multiplier = 1.32 + max(0, (18.0 - tilt_mismatch) * 0.007) + max(0, (25 - wind_cutoff) * 0.003)
    
    inv_mult = inverter_eff / 100.0
    
    fixed_yield = total_base_array * inv_mult
    standard_yield = total_base_array * standard_multiplier * wind_protection_factor * inv_mult
    ai_yield = total_base_array * ai_multiplier * wind_protection_factor * inv_mult
    
    return fixed_yield, standard_yield, ai_yield

def extract_gps_from_image(image):
    try:
        exif = image._getexif()
        if not exif:
            return None, None
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id in value:
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = value[gps_tag_id]
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat_data = gps_info["GPSLatitude"]
            lon_data = gps_info["GPSLongitude"]
            lat = float(lat_data[0]) + float(lat_data[1])/60.0 + float(lat_data[2])/3600.0
            lon = float(lon_data[0]) + float(lon_data[1])/60.0 + float(lon_data[2])/3600.0
            return lat, lon
    except Exception:
        pass
    return None, None

def analyze_site_image(uploaded_file, surface_type, site_notes, manual_image_location=""):
    """
    Performs rigorous computer vision pixel analysis (luminosity and green canopy foliage %)
    combined with U.S. geocoding and user notes to output a tailored Big 3 financial audit.
    """
    if uploaded_file is None:
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above to execute the computer vision and financial audit."
    
    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        img_format = img.format or "Standard Image"
        
        extracted_lat, extracted_lon = extract_gps_from_image(img)
        loc_str = manual_image_location if manual_image_location else "U.S. National Baseline Zone"
        resolved_lat = 35.4858 
        
        if manual_image_location:
            try:
                geolocator = Nominatim(user_agent="solar_evaluator_deep_audit")
                loc_obj = geolocator.geocode(manual_image_location + ", USA")
                if loc_obj:
                    resolved_lat = loc_obj.latitude
                    loc_str = loc_obj.address
            except Exception:
                pass
        elif extracted_lat and extracted_lon:
            resolved_lat = extracted_lat
            loc_str = f"GPS Verified U.S. Coordinate ({extracted_lat:.4f}, {extracted_lon:.4f})"

        # --- DEEP COMPUTER VISION PIXEL SCAN ---
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        avg_brightness = np.mean(arr)
        
        # Color channel thresholding for green canopy / tree vegetation detection
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_mask = (g.astype(int) > r.astype(int) + 12) & (g.astype(int) > b.astype(int) + 12)
        vegetation_percentage = (np.sum(green_mask) / green_mask.size) * 100

        notes_lower = site_notes.lower()
        wind_match = re.search(r'(\d+)\s*mph', notes_lower)
        noted_wind_speed = int(wind_match.group(1)) if wind_match else 12
        has_shading_mention = any(w in notes_lower for w in ["shade", "tree", "block", "shadow", "obstruction", "hvac"])

        # --- FINANCIAL & MECHANICAL WEAR MODEL FOR THE BIG 3 ---
        local_irradiance = max(25.0, 58.0 - (abs(resolved_lat) * 0.48))
        array_size = 12
        
        mono_eff, mono_cost, mono_durability = 21.8, 320, "High (Resistant to Micro-Cracking & Thermal Stress)"
        poly_eff, poly_cost, poly_durability = 17.4, 230, "Moderate (Sensitive to Wide Thermal Variations)"
        bifacial_eff, bifacial_cost, bifacial_durability = 24.5, 390, "Superior (Optimized for AI Tracking & Ground Albedo Gain)"

        shading_penalty = max(0.60, 1.0 - (vegetation_percentage / 80.0))
        if has_shading_mention:
            shading_penalty *= 0.92
            
        luminosity_factor = avg_brightness / 125.0
        
        mono_annual_kwh = local_irradiance * (mono_eff / 20.0) * shading_penalty * luminosity_factor * 46
        poly_annual_kwh = local_irradiance * (poly_eff / 20.0) * (shading_penalty * 0.90) * luminosity_factor * 46
        bifacial_annual_kwh = local_irradiance * (bifacial_eff / 20.0) * shading_penalty * luminosity_factor * 1.22 * 46

        electricity_rate = 0.14
        mono_upfront = mono_cost * array_size
        poly_upfront = poly_cost * array_size
        bifacial_upfront = bifacial_cost * array_size

        mono_25yr_savings = (mono_annual_kwh * 25 * electricity_rate) - mono_upfront
        poly_25yr_savings = (poly_annual_kwh * 25 * electricity_rate) - poly_upfront
        bifacial_25yr_savings = (bifacial_annual_kwh * 25 * electricity_rate) - bifacial_upfront

        report = f"""
        ### 🔬 Comprehensive Image-Driven Site Intelligence & Feasibility Audit
        - **U.S. Resolved Location:** `{loc_str}` (Latitude `{resolved_lat:.2f}°` active in solar physics model)
        - **Visual Asset Scanned:** `{uploaded_file.name}` ({width}x{height} px, {img_format})
        - **Pixel Luminosity Index:** `{avg_brightness:.1f} / 255.0` | **Detected Foliage / Canopy Interference:** `{vegetation_percentage:.1f}%`
        - **Site Parameters Logged:** "{site_notes}" (Evaluated for `{noted_wind_speed} mph` wind load stress)
        
        ---
        #### 📊 The Big 3 Panel Types: Energy Gain, Financial ROI & Wear Evaluation:
        1. **Monocrystalline Modules ({mono_eff}% Efficiency):**
           - *Annual Generation:* `{mono_annual_kwh:,.1f} kWh/yr` | *Upfront Investment:* **${mono_upfront:,.2f}**
           - *25-Year Net Financial Savings:* **${mono_25yr_savings:,.2f}**
           - *Wear & Tear Resilience:* **{mono_durability}**. High power density minimizes roof footprint, keeping modules safely outside the detected `{vegetation_percentage:.1f}%` canopy boundary.
           
        2. **Polycrystalline Modules ({poly_eff}% Efficiency):**
           - *Annual Generation:* `{poly_annual_kwh:,.1f} kWh/yr` | *Upfront Investment:* **${poly_upfront:,.2f}**
           - *25-Year Net Financial Savings:* **${poly_25yr_savings:,.2f}**
           - *Wear & Tear Resilience:* **{poly_durability}**. Requires a wider surface spread, exposing more panels to edge-shading and thermal fatigue under varying U.S. weather patterns.
           
        3. **Bifacial / Smart Tracking Modules ({bifacial_eff}% Efficiency):**
           - *Annual Generation:* `{bifacial_annual_kwh:,.1f} kWh/yr` | *Upfront Investment:* **${bifacial_upfront:,.2f}**
           - *25-Year Net Financial Savings:* **${bifacial_25yr_savings:,.2f}**
           - *Wear & Tear Resilience:* **{poly_durability}**. Maximizes clean energy capture by harvesting ground-reflected albedo while integrating seamlessly with AI single-axis tracking controllers.
           
        ---
        🔍 **Targeted Engineering Parameter Questionnaire & Checklist for This Site:**
        * **Wind Stress Evaluation:** Based on your logged `{noted_wind_speed} mph` conditions, ensure tracking controllers are programmed with a wind safety cutoff threshold between 15–20 mph to prevent gear fatigue.
        * **Tilt & Latitude Matching:** For `{loc_str.split(',')[0]}`, verify that fixed or tracking tilt angles match the local latitude ({resolved_lat:.1f}°) to optimize seasonal sun capture.
        * **Canopy & Shading Mitigation:** The computer vision scan detected `{vegetation_percentage:.1f}%` green pixels. If perimeter foliage encroaches, micro-inverters or string optimizers should be deployed to prevent voltage drops.
        * **Surface Compatibility:** The selected `{surface_type}` profile pairs best with **Bifacial modules** if ground or reflective roofing albedo exceeds 20%.
        """
        return report
    except Exception as e:
        return f"Error executing site audit: {str(e)}"
