def calculate_solar_physics(lat, tilt_angle, wind_cutoff, panel_count):
    # 1. Base solar energy based on latitude
    base_solar_irradiance = max(20.0, 52.0 - (abs(lat) * 0.45))
    
    # 2. Optimal tilt calculation
    optimal_tilt = abs(lat) * 0.85
    tilt_efficiency_factor = max(0.6, 1.0 - (0.0012 * ((tilt_angle - optimal_tilt) ** 2)))
    
    # 3. Real-world physics: Tracker gains INCREASE when the fixed panel tilt is poorly optimized!
    tilt_mismatch = abs(tilt_angle - optimal_tilt)
    
    # Standard tracker multiplier gets a bigger boost if the fixed tilt is far from optimal
    standard_multiplier = 1.12 + (tilt_mismatch * 0.003) + (abs(lat) * 0.0005)
    
    # AI tracker multiplier gets an even stronger boost from weather/tilt optimization
    ai_multiplier = 1.22 + (tilt_mismatch * 0.005) + (abs(lat) * 0.001) + max(0, (15.0 - wind_cutoff) * 0.004)
    
    # 4. Calculate final yields
    fixed_yield = panel_count * base_solar_irradiance * tilt_efficiency_factor
    standard_yield = fixed_yield * standard_multiplier
    ai_yield = fixed_yield * ai_multiplier
    
    return fixed_yield, standard_yield, ai_yield
