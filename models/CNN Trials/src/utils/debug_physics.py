import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add physics modules to path
sys.path.insert(0, 'physics')
from lgBeam import LaguerreGaussianBeam

def debug_physics():
    print("Debugging Physics: Checking Phase Rotation Visibility...")
    
    # Parameters
    wavelength = 1.55e-6
    w0 = 0.025
    N = 64
    
    # Dataset Generator Logic approximation:
    # z = 1000m
    # w(z) approx 0.032m
    # physical_beam_radius (l=4) approx 0.1m
    # D_sim = 2 * 6 * 0.1 = 1.2m
    D = 1.2 
    
    print(f"Simulating Grid: D={D}m, N={N}x{N}")
    print(f"Pixel Size: {D/N*1000:.2f} mm")
    print(f"Beam Waist (w0): {w0*1000:.2f} mm")
    print(f"Beam Diameter (at z): ~64 mm")
    print(f"Pixels per Beam Diameter: {0.064 / (D/N):.2f}")
    
    # Grid
    x = np.linspace(-D/2, D/2, N)
    y = np.linspace(-D/2, D/2, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    # 1. Create Pilot (l=0)
    pilot_beam = LaguerreGaussianBeam(p=0, l=0, wavelength=wavelength, w0=w0)
    E_pilot = pilot_beam.generate_beam_field(R, PHI, 0)
    # Normalize pilot
    E_pilot = E_pilot / np.sqrt(np.sum(np.abs(E_pilot)**2))
    
    # 2. Create Signal Mode (l=1)
    signal_beam = LaguerreGaussianBeam(p=0, l=1, wavelength=wavelength, w0=w0)
    E_signal_base = signal_beam.generate_beam_field(R, PHI, 0)
    # Normalize signal
    E_signal_base = E_signal_base / np.sqrt(np.sum(np.abs(E_signal_base)**2))
    
    # 3. Combine with different phases
    phases = [0, np.pi/2, np.pi, 3*np.pi/2]
    phase_names = ["0", "90", "180", "270"]
    
    plt.figure(figsize=(12, 3))
    
    for i, phase in enumerate(phases):
        # Rotate signal phase
        symbol = np.exp(1j * phase)
        E_signal = E_signal_base * symbol
        
        # Combine (50/50 power)
        E_total = (E_pilot + E_signal) / np.sqrt(2)
        
        # Intensity
        I = np.abs(E_total)**2
        
        plt.subplot(1, 4, i+1)
        plt.imshow(I, cmap='hot', origin='lower')
        plt.title(f"Phase {phase_names[i]}°")
        plt.axis('off')
        
        # Check center pixel (should be constant if l=1 has null, but pilot fills it)
        # Check max location
        max_loc = np.unravel_index(np.argmax(I), I.shape)
        # print(f"Phase {phase_names[i]}: Max at {max_loc}")

    plt.tight_layout()
    plt.savefig("debug_physics_rotation.png")
    print("Saved 'debug_physics_rotation.png'")
    
    # Check if images are identical
    # We'll compare Phase 0 and Phase 90
    symbol0 = np.exp(1j * phases[0])
    I0 = np.abs((E_pilot + E_signal_base * symbol0)/np.sqrt(2))**2
    
    symbol90 = np.exp(1j * phases[1])
    I90 = np.abs((E_pilot + E_signal_base * symbol90)/np.sqrt(2))**2
    
    diff = np.mean(np.abs(I0 - I90))
    print(f"\nDifference between 0° and 90°: {diff:.6f}")
    
    if diff < 1e-6:
        print(">> CRITICAL FAILURE: Images are IDENTICAL. Phase is invisible!")
    else:
        print(">> SUCCESS: Images are distinct. Phase information IS present.")

if __name__ == "__main__":
    debug_physics()
