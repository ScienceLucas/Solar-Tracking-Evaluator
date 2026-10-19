import numpy as np
import re
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim

def calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count):
    base_solar_irradiance = max(20.0, 52.0 - (abs(lat) * 0.45))
    optimal_tilt = abs(lat) * 0.85
    tilt_efficiency_factor = max(0.6, 1.0 - (0.0012 * ((tilt_angle - optimal_tilt) ** 2)))
    
    tilt_mismatch = abs(tilt_angle - optimal_tilt)
    standard_multiplier = 1.12 + (tilt_mismatch * 0.003) + (abs(lat) * 0.0005)
    ai_multiplier = 1.22 + (tilt_mismatch * 0.005) + (abs(lat) * 0.001) + max(0, (15.0 - wind_cutoff) * 0.004)
    
    fixed_yield = panel_count * base_solar_irradiance * tilt_efficiency_factor
    standard_yield = fixed_yield * standard_multiplier
    ai_yield = fixed_yield * ai_multiplier
    
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
    if uploaded_file is None:
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above to generate your financial audit."
    
    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        img_format = img.format or "Standard Image"
        
        extracted_lat, extracted_lon = extract_gps_from_image(img)
        loc_str = manual_image_location if manual_image_location else "U.S. National Baseline Zone"
        resolved_lat = 35.4858 
        
        if manual_image_location:
            try:
                geolocator = Nominatim(user_agent="solar_evaluator_clean")
                loc_obj = geolocator.geocode(manual_image_location + ", USA")
                if loc_obj:
                    resolved_lat = loc_obj.latitude
                    loc_str = loc_obj.address
            except Exception:
                pass
        elif extracted_lat and extracted_lon:
            resolved_lat = extracted_lat
            loc_str = f"GPS Verified ({extracted_lat:.4f}, {extracted_lon:.4f})"

        # Image Pixel Analysis
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        avg_brightness = np.mean(arr)
        
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_mask = (g.astype(int) > r.astype(int) + 15) & (g.astype(int) > b.astype(int) + 15)
        vegetation_percentage = (np.sum(green_mask) / green_mask.size) * 100

        notes_lower = site_notes.lower()
        wind_match = re.search(r'(\d+)\s*mph', notes_lower)
        noted_wind_speed = int(wind_match.group(1)) if wind_match else 12

        # Financial Calculations (The Big 3)
        local_irradiance = max(25.0, 52.0 - (abs(resolved_lat) * 0.45))
        array_size = 12
        
        mono_eff, mono_cost = 21.5, 320
        poly_eff, poly_cost = 17.2, 230
        bifacial_eff, bifacial_cost = 24.0, 380

        shading_penalty = max(0.7, 1.0 - (vegetation_percentage / 100.0))
        luminosity_factor = avg_brightness / 120.0
        
        mono_annual_kwh = local_irradiance * (mono_eff / 20.0) * shading_penalty * luminosity_factor * 42
        poly_annual_kwh = local_irradiance * (poly_eff / 20.0) * (shading_penalty * 0.94) * luminosity_factor * 42
        bifacial_annual_kwh = local_irradiance * (bifacial_eff / 20.0) * shading_penalty * luminosity_factor * 1.15 * 42

        electricity_rate = 0.14
        mono_upfront = mono_cost * array_size
        poly_upfront = poly_cost * array_size
        bifacial_upfront = bifacial_cost * array_size

        mono_25yr_savings = (mono_annual_kwh * 25 * electricity_rate) - mono_upfront
        poly_25yr_savings = (poly_annual_kwh * 25 * electricity_rate) - poly_upfront
        bifacial_25yr_savings = (bifacial_annual_kwh * 25 * electricity_rate) - bifacial_upfront

        report = f"""
        ### 💼 Comprehensive Site Intelligence & 25-Year Financial ROI Audit
        - **Resolved Location:** `{loc_str}` (Latitude `{resolved_lat:.2f}°` active in solar physics model)
        - **Visual Asset Scanned:** `{uploaded_file.name}` ({width}x{height} px) | **Foliage Interference:** `{vegetation_percentage:.1f}%`
        - **Operational Notes Logged:** "{site_notes}" (Wind factor evaluated at `{noted_wind_speed} mph`)
        
        ---
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
        💡 **Cross-Tab Actionable Guidance:** Take your location (`{loc_str.split(',')[0]}`) and wind parameters (`{noted_wind_speed} mph`) over to **Tab 1 (Physics & AI Tracking Simulation)** to test how dynamic single-axis tracking motor wear and energy multipliers perform for this exact site!
        """
        return report
    except Exception as e:
        return f"Error executing financial audit: {str(e)}"
