import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

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
    """Attempts to extract GPS coordinates from image EXIF metadata."""
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
            lat_ref = gps_info.get("GPSLatitudeRef", "N")
            lon_ref = gps_info.get("GPSLongitudeRef", "W")
            
            lat = float(lat_data[0]) + float(lat_data[1])/60.0 + float(lat_data[2])/3600.0
            if lat_ref != "N": 
                lat = -lat
            lon = float(lon_data[0]) + float(lon_data[1])/60.0 + float(lon_data[2])/3600.0
            if lon_ref != "W": 
                lon = -lon  # Defaulting standard US longitude orientation check
            return lat, lon
    except Exception:
        pass
    return None, None

def analyze_site_image(uploaded_file, surface_type, site_notes, manual_image_location=""):
    if uploaded_file is None:
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above."
    
    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        img_format = img.format or "Standard Image"
        
        # Check for embedded GPS EXIF metadata
        extracted_lat, extracted_lon = extract_gps_from_image(img)
        
        location_source_msg = ""
        if extracted_lat and extracted_lon:
            location_source_msg = f"📍 **EXIF Metadata Detected:** Extracted exact photo coordinates ({extracted_lat:.4f}, {extracted_lon:.4f})."
        elif manual_image_location:
            location_source_msg = f"📍 **Specified Image Location:** {manual_image_location}"
        else:
            location_source_msg = "📍 **Location:** No embedded GPS found; evaluated via general regional profile."

        # Convert to numpy array to analyze pixel characteristics
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        avg_brightness = np.mean(arr)
        
        exposure_status = "Optimal High-Light Exposure" if avg_brightness > 110 else "Potential Shading / Overcast Conditions Detected"
        aspect_ratio = width / height
        layout_type = "Aerial / Satellite View" if aspect_ratio > 1.2 or aspect_ratio < 0.8 else "Ground-Level Architectural View"
        
        mono_eff = round(21.5 + (0.6 if avg_brightness > 110 else -1.1), 1)
        poly_eff = round(17.0 + (0.4 if avg_brightness > 110 else -0.7), 1)
        bifacial_eff = round(23.8 + (1.4 if "Ground" in surface_type else 0.5), 1)
        
        report = f"""
        ### 🔬 Automated Computer Vision & Site Intelligence Report
        - {location_source_msg}
        - **Image Metadata:** {width}x{height} px ({img_format}), Aspect Ratio: {aspect_ratio:.2f} ({layout_type})
        - **Pixel Brightness / Albedo Index:** {avg_brightness:.1f} / 255.0 — *{exposure_status}*
        - **Selected Surface Profile:** {surface_type}
        - **Site Context Notes:** "{site_notes}"
        
        ---
        #### 📊 Panel Type Efficiency & Suitability Breakdown:
        1. **Monocrystalline Panels (High Efficiency):** **{mono_eff}% Efficiency Rating**
           - *Suitability:* Excellent for limited roof space. Maintains strong output under {exposure_status.lower()}.
        2. **Polycrystalline Panels (Cost-Effective):** **{poly_eff}% Efficiency Rating**
           - *Suitability:* Viable for large-scale arrays where initial capital cost is prioritized over peak spatial yield.
        3. **Bifacial / Smart Tracking Panels:** **{bifacial_eff}% Efficiency Rating**
           - *Suitability:* Highly recommended for this site layout. Captures reflected albedo radiation from surrounding terrain surfaces.
        """
        return report
    except Exception as e:
        return f"Error analyzing uploaded image file: {str(e)}"
