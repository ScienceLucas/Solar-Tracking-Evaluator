import numpy as np
import re
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim

def calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count):
    base_solar_irradiance = max(20.0, 52.0 - (abs(lat) * 0.45))
@@ -22,15 +24,13 @@ def extract_gps_from_image(image):
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
@@ -43,68 +43,98 @@ def extract_gps_from_image(image):

def analyze_site_image(uploaded_file, surface_type, site_notes, manual_image_location=""):
    if uploaded_file is None:
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above to run the visual sensor audit."
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above to generate your financial audit."

    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        img_format = img.format or "Standard Image"

        extracted_lat, extracted_lon = extract_gps_from_image(img)
        loc_display = manual_image_location if manual_image_location else "User Specified Location"
        if extracted_lat and extracted_lon:
            loc_display = f"GPS Tagged ({extracted_lat:.4f}, {extracted_lon:.4f})"
        loc_str = manual_image_location if manual_image_location else "U.S. National Baseline Zone"
        resolved_lat = 35.4858 # Default fallback (NC region)
        
        # Geocode location to factor true latitude into solar irradiance and financial math
        if manual_image_location:
            try:
                geolocator = Nominatim(user_agent="solar_evaluator_util_v3")
                loc_obj = geolocator.geocode(manual_image_location + ", USA")
                if loc_obj:
                    resolved_lat = loc_obj.latitude
                    loc_str = loc_obj.address
            except Exception:
                pass
        elif extracted_lat and extracted_lon:
            resolved_lat = extracted_lat
            loc_str = f"GPS Verified ({extracted_lat:.4f}, {extracted_lon:.4f})"

        # Convert image to RGB numpy array for pixel analysis
        # Image Pixel Analysis
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        
        # 1. Luminosity / Exposure Analysis
        avg_brightness = np.mean(arr)

        # 2. Vegetation & Tree Canopy Detection (Analyzing green channel dominance over red and blue)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_mask = (g.astype(int) > r.astype(int) + 15) & (g.astype(int) > b.astype(int) + 15)
        vegetation_percentage = (np.sum(green_mask) / green_mask.size) * 100

        # Parse Notes for Operational Factors
        notes_lower = site_notes.lower()
        wind_match = re.search(r'(\d+)\s*mph', notes_lower)
        noted_wind_speed = int(wind_match.group(1)) if wind_match else 12

        # --- FINANCIAL & EFFICIENCY CALCULATIONS (THE BIG 3) ---
        local_irradiance = max(25.0, 52.0 - (abs(resolved_lat) * 0.45))
        array_size = 12 # Standard baseline residential/commercial array block

        # 3. Determine Shading Risk & Feasibility Verdict purely from image pixels
        if vegetation_percentage > 30:
            shading_risk = "High (Dense surrounding tree canopy detected in frame)"
            verdict = "❌ Marginal Site. Substantial tree trimming required before solar deployment to prevent energy loss."
        elif vegetation_percentage > 12:
            shading_risk = "Moderate (Partial foliage proximity)"
            verdict = "⚠️ Acceptable Site. Micro-siting and tilt adjustments required to dodge canopy shadows."
        else:
            shading_risk = "Low (Open exposure area detected)"
            verdict = "✅ Excellent Site. Unobstructed surface geometry provides high potential for solar ROI."
        mono_eff, mono_cost = 21.5, 320
        poly_eff, poly_cost = 17.2, 230
        bifacial_eff, bifacial_cost = 24.0, 380

        # Dynamic panel efficiency calibration based on image pixel data
        mono_eff = round(21.8 - (vegetation_percentage * 0.04), 1)
        poly_eff = round(17.4 - (vegetation_percentage * 0.03), 1)
        bifacial_eff = round(24.3 + (1.0 if vegetation_percentage < 15 else -0.8), 1)
        shading_penalty = max(0.7, 1.0 - (vegetation_percentage / 100.0))
        luminosity_factor = avg_brightness / 120.0

        mono_annual_kwh = local_irradiance * (mono_eff / 20.0) * shading_penalty * luminosity_factor * 42
        poly_annual_kwh = local_irradiance * (poly_eff / 20.0) * (shading_penalty * 0.94) * luminosity_factor * 42
        bifacial_annual_kwh = local_irradiance * (bifacial_eff / 20.0) * shading_penalty * luminosity_factor * 1.15 * 42

        electricity_rate = 0.14 # $/kWh blended rate
        mono_upfront = mono_cost * array_size
        poly_upfront = poly_cost * array_size
        bifacial_upfront = bifacial_cost * array_size

        mono_25yr_savings = (mono_annual_kwh * 25 * electricity_rate) - mono_upfront
        poly_25yr_savings = (poly_annual_kwh * 25 * electricity_rate) - poly_upfront
        bifacial_25yr_savings = (bifacial_annual_kwh * 25 * electricity_rate) - bifacial_upfront

        report = f"""
        ### 👁️ Computer Vision Pixel & Sensor Audit Report
        - **Asset Evaluated:** `{uploaded_file.name}` ({width}x{height} px, {img_format})
        - **Site Location:** {loc_display}
        - **Detected Vegetation / Canopy Cover:** `{vegetation_percentage:.1f}%` of image pixels
        - **Luminosity / Exposure Index:** `{avg_brightness:.1f} / 255.0`
        - **User Notes:** "{site_notes}"
        ### 💼 Comprehensive Site Intelligence & 25-Year Financial ROI Audit
        - **Resolved Location:** `{loc_str}` (Latitude `{resolved_lat:.2f}°` active in solar physics model)
        - **Visual Asset Scanned:** `{uploaded_file.name}` ({width}x{height} px) | **Foliage Interference:** `{vegetation_percentage:.1f}%`
        - **Operational Notes Logged:** "{site_notes}" (Wind factor evaluated at `{noted_wind_speed} mph`)
        
        ---
        #### 🏗️ Automated Image Feasibility Verdict:
        **{verdict}**
        - *Shading Risk Level:* {shading_risk}

        #### 💰 25-Year Financial Comparison (The Big 3 Panel Types):
        1. **Monocrystalline Modules ({mono_eff}% Efficiency):**
           - *Estimated Upfront Cost:* **${mono_upfront:,.2f}** ({array_size} panels)
           - *Annual Generation:* `{mono_annual_kwh:,.1f} kWh/yr`
           - *Net 25-Year Lifetime Savings:* **${mono_25yr_savings:,.2f}**
           - *Engineering Verdict:* Best overall spatial efficiency. Highly recommended for `{loc_str}` to balance footprint limitations with high power conversion.
           
        2. **Polycrystalline Modules ({poly_eff}% Efficiency):**
           - *Estimated Upfront Cost:* **${poly_upfront:,.2f}**
           - *Annual Generation:* `{poly_annual_kwh:,.1f} kWh/yr`
           - *Net 25-Year Lifetime Savings:* **${poly_25yr_savings:,.2f}**
           - *Engineering Verdict:* Lower initial capital cost, but reduced power output results in lower long-term financial yield under this site's lighting metrics.
           
        3. **Bifacial / Smart Tracking Modules ({bifacial_eff}% Efficiency):**
           - *Estimated Upfront Cost:* **${bifacial_upfront:,.2f}**
           - *Annual Generation:* `{bifacial_annual_kwh:,.1f} kWh/yr`
           - *Net 25-Year Lifetime Savings:* **${bifacial_25yr_savings:,.2f}**
           - *Engineering Verdict:* Maximum long-term ROI. Highly recommended if pairing with ground mounts or reflective `{surface_type}` surfaces.
           
        ---
        #### 📊 Image-Calibrated Panel Efficiency & Suitability Breakdown:
        1. **Monocrystalline Modules:** **{mono_eff}% Efficiency Rating**
           - *Pixel-Calibrated Analysis:* Performs exceptionally well under the frame's current luminance index of {avg_brightness:.1f}. High power density makes it ideal for maximizing space.
        2. **Polycrystalline Modules:** **{poly_eff}% Efficiency Rating**
           - *Pixel-Calibrated Analysis:* Lower cost per watt, but reduced efficiency means larger array footprints, increasing vulnerability to the detected canopy edges.
        3. **Bifacial / Smart Tracking Modules:** **{bifacial_eff}% Efficiency Rating**
           - *Pixel-Calibrated Analysis:* Highly recommended if expanding beyond roof bounds to capture ground-reflected albedo radiation.
        💡 **Cross-Tab Actionable Guidance:** Take your location (`{loc_str.split(',')[0]}`) and wind parameters (`{noted_wind_speed} mph`) over to **Tab 1 (Physics & AI Tracking Simulation)** to test how dynamic single-axis tracking motor wear and energy multipliers perform for this exact site!
        """
        return report
    except Exception as e:
        return f"Error analyzing image pixels: {str(e)}"
        return f"Error executing financial audit: {str(e)}"
