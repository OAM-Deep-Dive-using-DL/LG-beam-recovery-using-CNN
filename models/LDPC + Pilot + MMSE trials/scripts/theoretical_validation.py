import numpy as np

def calculate_turbulence_metrics(wavelength, distance, cn2):
    k = 2 * np.pi / wavelength
    
    # 1. Rytov Variance (Plane Wave approximation)
    # sigma_R^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6)
    # < 1: Weak fluctuations
    # > 1: Strong fluctuations
    rytov_variance = 1.23 * cn2 * (k**(7.0/6.0)) * (distance**(11.0/6.0))
    
    # 2. Fried Parameter (r0) - Plane wave
    # r0 = [0.423 * k^2 * Cn^2 * L]^(-3/5)
    # Represents the coherence diameter of the field
    r0 = (0.423 * (k**2) * cn2 * distance)**(-3.0/5.0)
    
    # 3. Scintillation Index (approximate for weak turbulence)
    # sigma_I^2 approx sigma_R^2 for weak turbulence
    scintillation_index = rytov_variance
    
    return {
        "Rytov Variance": rytov_variance,
        "Fried Parameter (r0) [m]": r0,
        "Fried Parameter (r0) [mm]": r0 * 1000,
        "Regime": "Weak" if rytov_variance < 1 else "Strong"
    }

def analyze_scenario(name, cn2):
    wavelength = 1550e-9
    distance = 1000
    aperture_diameter = 0.5 # 50cm
    
    print(f"\n--- Scenario: {name} ---")
    print(f"Cn2: {cn2:.1e} m^(-2/3)")
    
    metrics = calculate_turbulence_metrics(wavelength, distance, cn2)
    
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    # Analysis of Aperture vs r0
    r0 = metrics["Fried Parameter (r0) [m]"]
    d_over_r0 = aperture_diameter / r0
    print(f"D/r0 ratio: {d_over_r0:.2f}")
    
    if d_over_r0 > 1:
        print("  -> Aperture is larger than coherence length (D > r0).")
        print("  -> Significant wavefront distortion and phase errors expected across the aperture.")
        print("  -> OAM modes will suffer heavy crosstalk.")
    else:
        print("  -> Aperture is within coherence length (D < r0).")
        print("  -> Phase is relatively uniform; lower crosstalk expected.")

if __name__ == "__main__":
    print("Theoretical Turbulence Analysis for FSO Link")
    print("Parameters: L=1km, lambda=1550nm, D=0.5m")
    
    analyze_scenario("Weak Turbulence (Success Case)", 1e-16)
    analyze_scenario("Stronger Turbulence (High BER Case)", 1e-15)
