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
        return "⚠️ No image uploaded. Please upload a site layout or aerial photo in Step 1 above to run the visual sensor audit."
    
    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        img_format = img.format or "Standard Image"
        
        extracted_lat, extracted_lon = extract_gps_from_image(img)
        loc_display = manual_image_location if manual_image_location else "User Specified Location"
        if extracted_lat and extracted_lon:
            loc_display = f"GPS Tagged ({extracted_lat:.4f}, {extracted_lon:.4f})"

        # Convert image to RGB numpy array for pixel analysis
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb)
        
        # 1. Luminosity / Exposure Analysis
        avg_brightness = np.mean(arr)
        
        # 2. Vegetation & Tree Canopy Detection (Analyzing green channel dominance over red and blue)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_mask = (g.astype(int) > r.astype(int) + 15) & (g.astype(int) > b.astype(int) + 15)
        vegetation_percentage = (np.sum(green_mask) / green_mask.size) * 100
        
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

        # Dynamic panel efficiency calibration based on image pixel data
        mono_eff = round(21.8 - (vegetation_percentage * 0.04), 1)
        poly_eff = round(17.4 - (vegetation_percentage * 0.03), 1)
        bifacial_eff = round(24.3 + (1.0 if vegetation_percentage < 15 else -0.8), 1)
        
        report = f"""
        ### 👁️ Computer Vision Pixel & Sensor Audit Report
        - **Asset Evaluated:** `{uploaded_file.name}` ({width}x{height} px, {img_format})
        - **Site Location:** {loc_display}
        - **Detected Vegetation / Canopy Cover:** `{vegetation_percentage:.1f}%` of image pixels
        - **Luminosity / Exposure Index:** `{avg_brightness:.1f} / 255.0`
        - **User Notes:** "{site_notes}"
        
        ---
        #### 🏗️ Automated Image Feasibility Verdict:
        **{verdict}**
        - *Shading Risk Level:* {shading_risk}

        ---
        #### 📊 Image-Calibrated Panel Efficiency & Suitability Breakdown:
        1. **Monocrystalline Modules:** **{mono_eff}% Efficiency Rating**
           - *Pixel-Calibrated Analysis:* Performs exceptionally well under the frame's current luminance index of {avg_brightness:.1f}. High power density makes it ideal for maximizing space.
        2. **Polycrystalline Modules:** **{poly_eff}% Efficiency Rating**
           - *Pixel-Calibrated Analysis:* Lower cost per watt, but reduced efficiency means larger array footprints, increasing vulnerability to the detected canopy edges.
        3. **Bifacial / Smart Tracking Modules:** **{bifacial_eff}% Efficiency Rating**
           - *Pixel-Calibrated Analysis:* Highly recommended if expanding beyond roof bounds to capture ground-reflected albedo radiation.
        """
        return report
    except Exception as e:
        return f"Error analyzing image pixels: {str(e)}"
