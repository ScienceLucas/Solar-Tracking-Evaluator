# Paste your exact Colab math and physics functions here

def calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count):
    base_solar_irradiance = max(20.0, 52.0 - (abs(lat) * 0.45))
    optimal_tilt = abs(lat) * 0.85
    tilt_efficiency_factor = max(0.6, 1.0 - (0.0012 * ((tilt_angle - optimal_tilt) ** 2)))
    
    fixed_yield = panel_count * base_solar_irradiance * tilt_efficiency_factor
    standard_yield = fixed_yield * (1.12 + (abs(lat) * 0.0008))
    ai_yield = fixed_yield * (1.22 + (abs(lat) * 0.0012) + max(0, (15.0 - wind_cutoff) * 0.004))
    
    return fixed_yield, standard_yield, ai_yield
