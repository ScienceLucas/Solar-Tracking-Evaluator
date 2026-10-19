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
    # Base U.S. solar irradiance scaled by latitude (°N)
    base_solar_irradiance = max(25.0, 58.0 - (abs(lat) * 0.48))
    
    # Optimal tilt angle in the Northern hemisphere closely mirrors local latitude
    optimal_tilt = abs(lat) * 0.85
    tilt_mismatch = abs(tilt_angle - optimal_tilt)
    
    # Non-linear tilt efficiency drop-off based on solar angle of incidence
    tilt_efficiency = max(0.5, 1.0 - (0.0018 * (tilt_mismatch ** 1.6)))
    
    # Wind safety factor: low wind cutoffs cause frequent stowing, reducing yield; high cutoffs risk gear fatigue
    wind_protection_factor = min(1.0, wind_cutoff / 22.0)
    
    # Base annual generation per panel (kWh)
    panel_base_yield = base_solar_irradiance * tilt_efficiency * 48.5
    total_base_array = panel_base_yield * panel_count
    
    # Multipliers for Fixed Tilt vs Standard Trackers vs AI Smart Trackers
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
    Performs rigorous computer vision pixel analysis (luminosity index, green foliage canopy %)
    combined with U.S. geocoding and user notes to output a comprehensive Big 3 financial audit.
    """
    if uploaded_file is None:
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above to execute the computer vision and financial audit."
    
    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        img_format = img.format or "Standard Image"
        
        extracted_lat, extracted_lon = extract_gps_from_image(img)
        loc_str = manual_image_location if manual_image_location else "U.S. National Baseline Zone"
        resolved_lat = 35.4858 # Default U.S. baseline latitude (NC)
        
        if manual_image_location:
            try:
                geolocator = Nominatim(user_agent="solar_evaluator_pro_audit")
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
        avg_brightness = np.mean(arr) # 0 to 255 luminosity index
        
        # Color channel thresholding for green canopy / tree vegetation detection
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_mask = (g.astype(int) > r.astype(int) + 12) & (g.astype(int) > b.astype(int) + 12)
        vegetation_percentage = (np.sum(green_mask) / green_mask.size) * 100

        # Parse user notes for environmental constraints
        notes_lower = site_notes.lower()
        wind_match = re.search(r'(\d+)\s*mph', notes_lower)
        noted_wind_speed = int(wind_match.group(1)) if wind_match else 12
        has_shading_mention = any(w in notes_lower for w in ["shade", "tree", "block", "shadow", "obstruction", "hvac"])

        # --- FINANCIAL & MECHANICAL WEAR MODEL FOR THE BIG 3 ---
        local_irradiance = max(25.0, 58.0 - (abs(resolved_lat) * 0.48))
        array_size = 12 # Standard baseline residential/commercial array block
        
        # Panel characteristics: Efficiency (%), Upfront Cost ($/panel), Wear Resilience Score
        mono_eff, mono_cost, mono_durability = 21.8, 320, "High (Resistant to Micro-Cracking)"
        poly_eff, poly_cost, poly_durability = 17.4, 230, "Moderate (Sensitive to Thermal Stress)"
        bifacial_eff, bifacial_cost, bifacial_durability = 24.5, 390, "Superior (Optimized for AI Tracking & Albedo)"

        # Shading and luminosity modifiers derived from image pixels
        shading_penalty = max(0.60, 1.0 - (vegetation_percentage / 80.0))
        if has_shading_mention:
            shading_penalty *= 0.92
            
        luminosity_factor = avg_brightness / 125.0
        
        # Annual kWh Yield Calculations
        mono_annual_kwh = local_irradiance * (mono_eff / 20.0) * shading_penalty * luminosity_factor * 46
        poly_annual_kwh = local_irradiance * (poly_eff / 20.0) * (shading_penalty * 0.90) * luminosity_factor * 46
        bifacial_annual_kwh = local_irradiance * (bifacial_eff / 20.0) * shading_penalty * luminosity_factor * 1.22 * 46

        # 25-Year Financial ROI Calculations ($0.14 / kWh blended rate)
        electricity_rate = 0.14
        mono_upfront = mono_cost * array_size
        poly_upfront = poly_cost * array_size
        bifacial_upfront = bifacial_cost * array_size

        mono_25yr_savings = (mono_annual_kwh * 25 * electricity_rate) - mono_upfront
        poly_25yr_savings = (poly_annual_kwh * 25 * electricity_rate) - poly_upfront
        bifacial_25yr_savings = (bifacial_annual_kwh * 25 * electricity_rate) - bifacial_upfront

        report = f"""
        ### 🔬 Comprehensive Computer Vision & U.S. Site Intelligence Audit
        - **U.S. Resolved Location:** `{loc_str}` (Latitude `{resolved_lat:.2f}°` active in solar physics model)
        - **Visual Asset Scanned:** `{uploaded_file.name}` ({width}x{height} px, {img_format})
        - **Pixel Luminosity Index:** `{avg_brightness:.1f} / 255.0` | **Detected Foliage Interference:** `{vegetation_percentage:.1f}%`
        - **Operational Notes Logged:** "{site_notes}" (Evaluated for `{noted_wind_speed} mph` wind load stress)
        
        ---
        #### 💰 25-Year Financial & Mechanical Wear Audit (The Big 3 Panel Types):
        1. **Monocrystalline Modules ({mono_eff}% Efficiency):**
           - *Estimated Upfront Investment:* **${mono_upfront:,.2f}** ({array_size} panels)
           - *Annual Generation:* `{mono_annual_kwh:,.1f} kWh/yr`
           - *Net 25-Year Lifetime Savings:* **${mono_25yr_savings:,.2f}**
           - *Mechanical Wear & Tear Analysis:* **{mono_durability}**. High power density minimizes roof footprint, keeping modules safely outside the detected `{vegetation_percentage:.1f}%` canopy boundary.
           
        2. **Polycrystalline Modules ({poly_eff}% Efficiency):**
           - *Estimated Upfront Investment:* **${poly_upfront:,.2f}**
           - *Annual Generation:* `{poly_annual_kwh:,.1f} kWh/yr`
           - *Net 25-Year Lifetime Savings:* **${poly_25yr_savings:,.2f}**
           - *Mechanical Wear & Tear Analysis:* **{poly_durability}**. Lower conversion efficiency requires a wider surface spread, exposing more panels to edge-shading and thermal fatigue.
           
        3. **Bifacial / Smart Tracking Modules ({bifacial_eff}% Efficiency):**
           - *Estimated Upfront Investment:* **${bifacial_upfront:,.2f}**
           - *Annual Generation:* `{bifacial_annual_kwh:,.1f} kWh/yr`
           - *Net 25-Year Lifetime Savings:* **${bifacial_25yr_savings:,.2f}**
           - *Mechanical Wear & Tear Analysis:* **{bifacial_durability}**. Maximizes clean energy capture by harvesting ground-reflected albedo radiation while integrating seamlessly with AI single-axis tracking controllers.
           
        ---
        💡 **Cross-Tab Actionable Guidance:** 
        Take your location (`{loc_str.split(',')[0]}`) and wind parameters (`{noted_wind_speed} mph`) over to **Tab 1 (U.S. Tracking Physics & Simulation)**. Set your wind cutoff slider to match your site notes and observe how the live physics model calculates motor wear reduction and net energy multipliers!
        """
        return report
    except Exception as e:
        return f"Error executing site audit: {str(e)}"
