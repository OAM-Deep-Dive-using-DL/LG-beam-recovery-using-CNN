import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
import warnings

# --- Setup Python Path ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

# --- Import lgBeam module ---
try:
    from lgBeam import LaguerreGaussianBeam 
except ImportError:
    print(f"Error: Could not find 'lgBeam.py' in the script directory: {SCRIPT_DIR}")
    print("Please make sure the lgBeam.py file from the previous step is saved in the same folder.")
    sys.exit(1)

warnings.filterwarnings('ignore')
np.random.seed(42)


# --- ANGULAR SPECTRUM PROPAGATION (Correct) ---
def angular_spectrum_propagation(field, delta, wavelength, distance):
    """
    Angular spectrum propagation for free-space wave propagation.
    
    This implements the full (non-paraxial) angular spectrum method using
    the exact transfer function:
    
    H(fx, fy) = exp[i*k*z*sqrt(1 - (λfx)² - (λfy)²)]
    
    where fx, fy are spatial frequencies, k = 2π/λ is the wavenumber,
    and z is the propagation distance.
    
    Evanescent waves (where fx² + fy² > 1/λ²) are set to zero to prevent
    numerical errors.
    
    References:
    - Goodman, J.W., "Introduction to Fourier Optics" (2005), Chapter 3
    - Schmidt, J.D., "Numerical Simulation of Optical Wave Propagation" (2010), Ch. 3
    
    Parameters:
    -----------
    field : ndarray
        Input complex field at z=0 (N×N array)
    delta : float
        Grid spacing [meters]
    wavelength : float
        Optical wavelength [meters]
    distance : float
        Propagation distance [meters]
    
    Returns:
    --------
    propagated_field : ndarray
        Output complex field at z=distance (N×N array)
    """
    N = field.shape[0]
    k = 2 * np.pi / wavelength
    
    # Create frequency grid
    fx = fftfreq(N, delta)
    fy = fftfreq(N, delta)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    f2 = FX**2 + FY**2
    
    # Transfer function H(fx, fy)
    with np.errstate(invalid='ignore'):  # Ignore warnings for sqrt of neg numbers
        H = np.exp(1j * k * distance * np.sqrt(1 - (wavelength**2) * f2))
    
    # Set evanescent waves (where f2 > 1/lambda^2) to 0
    # This is the standard approach to prevent NaN errors
    H[f2 > (1.0 / wavelength**2)] = 0
    
    # Perform the propagation
    field_ft = fft2(field)
    propagated_ft = field_ft * H
    propagated_field = ifft2(propagated_ft)
    
    return propagated_field


# --- PHASE SCREEN GENERATOR (FULLY RECTIFIED) ---
def generate_phase_screen(r0, N, delta, L0=10.0, l0=0.005):
    """
    Generates a single random phase screen using the von Kármán PSD.
    
    This is a robust, literature-based implementation.
    [Ref: Schmidt, "Numerical Simulation of Optical Wave Propagation" (2010), Ch. 5]
    [Ref: Andrews & Phillips, (2005), Ch. 12, Eq. 12.75]
    """
    df = 1.0 / (N * delta)  # Frequency grid spacing [m⁻¹]
    
    fx = fftfreq(N, delta)
    fy = fftfreq(N, delta)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    f = np.sqrt(FX**2 + FY**2)
    f[0, 0] = 1e-15  # Avoid singularity at DC

    f0 = 1.0 / L0  # Outer scale frequency [m⁻¹]
    fm = 5.92 / (2.0 * np.pi * l0)  # Inner scale frequency [m⁻¹]

    ### --- FIX 1: Correct Von Karman Phase PSD Constant --- ###
    # The constant for the von Karman *Phase* PSD, Phi_phi(f), is 
    # approximately 0.49, not 0.023.
    # Ref: Schmidt (2010), Eq. 5.17, combining 5.6, 5.14, 5.17
    # Phi_phi(f) = (2pi*k)^2 * dz * Phi_n(f)
    # Phi_n(f) = 0.033 * Cn2 * ...
    # r0 = (0.423 * k^2 * Cn2 * dz)^(-3/5)
    # --> Phi_phi(f) approx 0.49 * r0**(-5/3) * ...
    PSD_CONSTANT = 0.49
    psd_phi = PSD_CONSTANT * r0**(-5/3) * np.exp(-(f / fm)**2) / ((f**2 + f0**2)**(11/6))
    psd_phi[0, 0] = 0 # Set DC component to zero
    
    # Create unit-variance complex Gaussian noise
    noise = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
    
    # Apply the PSD and the correct FFT scaling for scipy.fft.ifft2
    # The (N / delta) factor is (1 / df)
    phi_ft = noise * (N / delta) * np.sqrt(psd_phi)

    # Inverse FFT to get the phase screen
    phi = np.real(ifft2(phi_ft))
    
    return phi


# --- TURBULENCE PARAMETER CLASS (Correct) ---
class AtmosphericTurbulence:
    """Helper class for atmospheric turbulence parameter calculations."""
    def __init__(self, Cn2=1e-14, L0=10.0, l0=0.005, wavelength=1550e-9):
        self.Cn2 = Cn2
        self.L0 = L0
        self.l0 = l0
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength

    def fried_parameter(self, L_path):
        """Calculates the path-integrated Fried parameter r0."""
        return (0.423 * self.k**2 * self.Cn2 * L_path)**(-3/5)

    def turbulence_strength(self):
        """Returns qualitative turbulence strength based on Cn2."""
        # This classification is based on Cn2 at sea level
        if self.Cn2 <= 5e-17:
            return "Weak"
        elif self.Cn2 <= 5e-15:
            return "Weak-to-Moderate"
        elif self.Cn2 <= 5e-13:
            return "Moderate-to-Strong"
        else:
            return "Strong"

    def rytov_variance(self, distance):
        """Calculates the Rytov variance for a plane wave."""
        return 1.23 * self.Cn2 * self.k**(7/6) * distance**(11/6)


# --- MULTI-LAYER SCREEN CREATION (Correct) ---
def create_multi_layer_screens(total_distance, num_screens, wavelength, ground_Cn2=5e-13, L0=10.0, l0=0.005, verbose=True):
    """
    Generates the positions and strengths (r0) for each phase screen.
    """
    turb = AtmosphericTurbulence(ground_Cn2, L0, l0, wavelength)
    k = turb.k
    
    if num_screens == 1:
        positions = [total_distance / 2.0]
        delta_Ls = [total_distance]
    else:
        # Use standard uniform spacing
        positions = np.linspace(0, total_distance, num_screens + 1)
        delta_Ls = np.diff(positions)
        # Place screen at the *center* of its slab
        positions = positions[:-1] + delta_Ls / 2.0
        
    layers = []

    if verbose:
        print(f"\nLayer Configuration ({num_screens} screens for Cn²={ground_Cn2:.1e}):")
        print(f"{'Layer':<6} {'Pos [m]':<10} {'ΔL [m]':<10} {'r0_layer [mm]':<15}")
        print("-" * 50)

    for i in range(num_screens):
        pos = positions[i]
        delta_L = delta_Ls[i]
        
        if delta_L <= 0: continue 

        r0_layer = (0.423 * k**2 * ground_Cn2 * delta_L)**(-3/5)
        
        if verbose:
            print(f"{i+1:<6} {pos:<10.1f} {delta_L:<10.1f} {r0_layer*1000:<15.2f}")

        layers.append({
            'position': pos,
            'r0_layer': r0_layer,
            'delta_L': delta_L
        })

    return layers


# --- SPLIT-STEP FOURIER PROPAGATION (RECTIFIED FOR PIPELINE) ---
def apply_multi_layer_turbulence(
    initial_field,  
    base_beam,      
    layers, 
    total_distance, 
    *,
    N, 
    oversampling, 
    L0, 
    l0,
    phase_screens=None # <--- ADD THIS ARGUMENT
):
    """
    Applies the full Split-Step Fourier (SSF) propagation method
    to a given initial_field.
    
    This version is rectified to accept a list of pre-generated
    'phase_screens'. If provided, it uses them. If 'phase_screens'
    is None, it generates them on the fly (old behavior).
    
    This ensures that the same physical turbulence can be applied
    to multiple fields, which is essential for dataset generation.
    """
    
    # Calculate grid parameters
    D = oversampling * 6 * base_beam.beam_waist(total_distance)
    delta = D / N
    
    field_turb = initial_field.copy()
    field_pristine = initial_field.copy()
    
    layers_sorted = sorted(layers, key=lambda x: x['position'])
    current_position = 0

    # This list will store the screens we *actually* use
    used_phase_screens = [] 
    
    # --- RECTIFIED LOGIC ---
    # We must iterate with an index 'i' to access phase_screens[i]
    for i, layer in enumerate(layers_sorted):
        prop_distance = layer['position'] - current_position
            
        # 1. Propagate BOTH fields
        if prop_distance > 1e-9:  
            field_turb = angular_spectrum_propagation(field_turb, delta, base_beam.wavelength, prop_distance)
            field_pristine = angular_spectrum_propagation(field_pristine, delta, base_beam.wavelength, prop_distance)

        # 2. Generate or apply phase screen to TURBULENT field only
        if phase_screens is not None:
            # We were given a list of screens. Use the correct one.
            phi = phase_screens[i]
        else:
            # No list was given. Generate a new one (fallback behavior).
            phi = generate_phase_screen(layer['r0_layer'], N, delta, L0, l0)
        
        used_phase_screens.append(phi) # Store the screen we used
        
        field_turb = field_turb * np.exp(1j * phi)
        # (field_pristine is not modified)

        current_position = layer['position']
    # --- END OF RECTIFIED LOGIC ---

    # 3. Propagate remaining distance for BOTH fields
    remaining_distance = total_distance - current_position
    if remaining_distance > 1e-9:
        field_turb = angular_spectrum_propagation(field_turb, delta, base_beam.wavelength, remaining_distance)
        field_pristine = angular_spectrum_propagation(field_pristine, delta, base_beam.wavelength, remaining_distance)

    return {
        'final_field': field_turb,
        'pristine_field': field_pristine,
        'phase_screens': used_phase_screens, # Return the screens that were used
        'grid_info': {'D': D, 'delta': delta, 'N': N}
    }


# --- PHASE SCREEN DIAGNOSTICS (Corrected) ---
def diagnose_phase_screen(r0, N, delta, L0=10.0, l0=0.005):
    """
    Diagnose phase screen statistics and validate against theory.
    """
    print(f"\n--- Phase Screen Diagnostics ---")
    
    phi = generate_phase_screen(r0, N, delta, L0, l0)
    phase_var = np.var(phi)
    
    D = N * delta
    expected_var_kolmogorov = 1.03 * (D / r0)**(5/3)
    
    print(f"  Screen size D: {D:.2f} m, r0: {r0*1000:.2f} mm, D/r0 = {D/r0:.1f}")
    
    print(f"\nPhase Variance Statistics:")
    print(f"  Actual (von Kármán) variance:   {phase_var:.3e} rad²")
    print(f"  Expected (Kolmogorov) variance: {expected_var_kolmogorov:.3e} rad²")
    
    ratio = phase_var / (expected_var_kolmogorov + 1e-12)
    print(f"  Ratio (actual/expected):        {ratio:.3f}")
    
    ### --- FIX 3: Corrected Validation Logic --- ###
    # The old check (var < expected * 1.5) was wrong, as it passed
    # when the variance was far too low (like 0.040).
    # A correct check ensures the ratio is reasonably close to 1.0.
    if 0.5 < ratio < 1.5: # Check if ratio is within +/- 50%
        print(f"  Validation: ✓ PASSED (Actual variance is close to expected)")
    else:
        print(f"  Validation: ✗ FAILED (Ratio {ratio:.3f} is far from 1.0. Check PSD constant.)")
    
    return phi


# --- TURBULENCE EFFECTS ANALYSIS (Corrected) ---
def analyze_turbulence_effects(beam, layers, total_distance, N=256, oversampling=1, L0=10.0, l0=0.005):
    """
    Analyzes turbulence effects by computing Strehl ratio and scintillation index.
    """
    
    # Generate the initial field at z=0
    D = oversampling * 6 * beam.beam_waist(total_distance) # Size for receiver
    delta = D / N
    x = np.linspace(-D/2, D/2, N)
    y = np.linspace(-D/2, D/2, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    initial_field = beam.generate_beam_field(R, PHI, 0)

    result = apply_multi_layer_turbulence(
        initial_field, beam, layers, total_distance, 
        N=N, oversampling=oversampling, L0=L0, l0=l0
    )
    
    final_intensity = np.abs(result['final_field'])**2
    # --- BUG FIX: Use the NUMERICAL pristine field from the result ---
    pristine_intensity = np.abs(result['pristine_field'])**2
    
    if np.isnan(final_intensity).any():
        print("\n--- TURBULENCE EFFECTS SUMMARY: FAILED (NaN) ---")
        return result

    pristine_max = np.max(pristine_intensity)
    final_max = np.max(final_intensity)
    
    # Strehl ratio: ratio of peak intensities
    strehl = final_max / (pristine_max + 1e-12) # Add epsilon
    
    
    ### --- FIX 2: Correct Scintillation Index Calculation --- ###
    # Scintillation index must be calculated only over the beam's
    # region of interest (ROI), not the entire empty grid.
    # We define the ROI as anywhere the *pristine* beam has
    # intensity > 1% of its peak.
    
    # Create a boolean mask for the ROI
    roi_mask = pristine_intensity > (pristine_max * 0.01)
    
    if np.sum(roi_mask) > 1: # Ensure the mask is not empty (use > 1 for var)
        # Calculate mean and variance *only* within the ROI
        mean_intensity_roi = np.mean(final_intensity[roi_mask])
        var_intensity_roi = np.var(final_intensity[roi_mask])
        
        if mean_intensity_roi > 1e-12:
            scintillation = var_intensity_roi / (mean_intensity_roi**2)
        else:
            scintillation = 0.0 # Avoid division by zero
    else:
        scintillation = 0.0 # Should not happen, but safe to check
    
    # --- End of Fix 2 ---

    print(f"\n--- Turbulence Effects Summary ---")
    print(f"  Peak intensity ratio (Strehl): {strehl:.3f}")
    print(f"  Scintillation index (σ_I²):    {scintillation:.3f}")

    return result


# --- PLOTTING FUNCTION (Corrected) ---
def plot_multi_layer_effects(beam, num_screens_list, distance, 
                            ground_Cn2, L0, l0, 
                            N, oversampling, save_path=None):
    """
    Plots a 4xN grid comparing pristine/distorted intensity/phase.
    """
    n_cases = len(num_screens_list)
    fig, axes = plt.subplots(4, n_cases, figsize=(6*n_cases, 16), squeeze=False)

    for i, num_screens in enumerate(num_screens_list):
        print(f"\nPlotting case for {num_screens} screen(s)...")
        layers = create_multi_layer_screens(distance, num_screens, beam.wavelength, 
                                           ground_Cn2, L0, l0, verbose=False)
        
        # Generate the initial field at z=0
        D = oversampling * 6 * beam.beam_waist(distance) # Size for receiver
        delta = D / N
        x = np.linspace(-D/2, D/2, N)
        y = np.linspace(-D/2, D/2, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        initial_field = beam.generate_beam_field(R, PHI, 0)

        result = apply_multi_layer_turbulence(
            initial_field, beam, layers, distance, N=N, 
            oversampling=oversampling, L0=L0, l0=l0
        )
        
        # --- BUG FIX: Use the NUMERICAL pristine field from the result ---
        pristine_field = result['pristine_field']
        distorted_field = result['final_field']
        
        if np.isnan(distorted_field).any():
            print(f"Plotting failed for {num_screens} screens (NaN field).")
            axes[2, i].set_title("PROPAGATION FAILED (NaN)")
            axes[3, i].set_title("PROPAGATION FAILED (NaN)")
            continue

        pristine_intensity = np.abs(pristine_field)**2
        distorted_intensity = np.abs(distorted_field)**2
        pristine_phase = np.angle(pristine_field)
        distorted_phase = np.angle(distorted_field)
        
        grid = result['grid_info']
        extent_mm = grid['D'] * 1e3 / 2

        # Plotting
        im1 = axes[0, i].imshow(pristine_intensity.T, 
                                extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                                cmap='hot', origin='lower')
        axes[0, i].set_title(f'Pristine Intensity LG_{beam.p}^{beam.l}')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046).set_label('Intensity', fontsize=9)

        im2 = axes[1, i].imshow(pristine_phase.T, 
                                extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                                cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
        axes[1, i].set_title(f'Pristine Phase (z={distance}m)')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046).set_label('Phase [rad]', fontsize=9)

        im3 = axes[2, i].imshow(distorted_intensity.T, 
                                extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                                cmap='hot', origin='lower')
        axes[2, i].set_title(f'Distorted Intensity ({num_screens} Screens)')
        plt.colorbar(im3, ax=axes[2, i], fraction=0.046).set_label('Intensity', fontsize=9)

        im4 = axes[3, i].imshow(distorted_phase.T, 
                                extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                                cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
        axes[3, i].set_title(f'Distorted Phase (Cn²={ground_Cn2:.0e})')
        plt.colorbar(im4, ax=axes[3, i], fraction=0.046).set_label('Phase [rad]', fontsize=9)
        
        for r in range(4):
            axes[r, i].set_xlabel('x [mm]')
            axes[r, i].set_ylabel('y [mm]')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


# --- MAIN EXECUTION (FOR TESTING THIS FILE ONLY) ---
if __name__ == "__main__":
    
    # --- Global Simulation Parameters ---
    WAVELENGTH = 1550e-9  
    W0 = 25e-3           
    DISTANCE = 1000      # 1 km
    
    CN2 = 1e-12          # VERY WEAK turbulence
    L0 = 10.0           # 10 m
    L0_INNER = 0.005    # 5 mm

    N_GRID = 512        
    OVERSAMPLING = 2    
    NUM_SCREENS_OPTIONS = [1, 5, 10]

    # --- Setup and Display Configuration ---
    print("ATMOSPHERIC TURBULENCE SIMULATION (DIAGNOSTIC RUN)")
    print("\nCONFIGURATION:")
    print(f"  Wavelength:       {WAVELENGTH*1e9:.0f} nm")
    print(f"  Beam waist:       {W0*1e3:.1f} mm")
    print(f"  Link distance:    {DISTANCE} m")
    print(f"  Cn²:              {CN2:.2e} m^(-2/3)")
    print(f"  Outer scale (L0): {L0:.1f} m")
    print(f"  Inner scale (l0): {L0_INNER*1000:.1f} mm")

    # Create a test beam (e.g., LG_1^1)
    beam = LaguerreGaussianBeam(1, 1, WAVELENGTH, W0) # M^2 = 4

    # Create turbulence object
    turb = AtmosphericTurbulence(CN2, L0, L0_INNER, WAVELENGTH)
    total_r0 = turb.fried_parameter(DISTANCE) # This is the r0 for the *entire path*
    
    print(f"\nBeam: LG_{beam.p}^{beam.l} (M²={beam.M_squared:.1f})")
    print(f"Total r0 (path-integrated): {total_r0*1000:.2f} mm ({turb.turbulence_strength()} turbulence)")
    print(f"Total Rytov variance: {turb.rytov_variance(DISTANCE):.3f}")

    # --- Grid and Sampling Diagnostics ---
    beam_size_at_rx = beam.beam_waist(DISTANCE)
    D = OVERSAMPLING * 6 * beam_size_at_rx
    delta = D / N_GRID
    
    print(f"\n--- Simulation Grid ---")
    print(f"  Grid size:        D = {D:.2f} m")
    print(f"  Grid points:      N = {N_GRID}")
    print(f"  Pixel spacing:    Δx = {delta*1000:.2f} mm")
    print(f"  Beam size at Rx:  w(L) = {beam_size_at_rx*1000:.2f} mm")
    
    # Critical sampling check
    print(f"\n--- Sampling Validation ---")
    sampling_ratio = delta / L0_INNER
    print(f"  Δx / l0 = {sampling_ratio:.3f} (MUST be < 1.0 for accurate sampling)")
    
    if sampling_ratio >= 1.0:
        print(f"    WARNING: UNDERSAMPLING! Increase N_GRID or increase l0.")
    else:
        print(f"   Sampling is adequate")
    
    # --- Phase Screen Diagnostics ---
    diagnose_phase_screen(total_r0, N_GRID, delta, L0, L0_INNER)
    
    # --- Run Full Multi-Screen Simulation ---
    for num_screens in NUM_SCREENS_OPTIONS:
        print(f"\n--- CASE: {num_screens} SCREEN(S) ---")
        
        layers = create_multi_layer_screens(DISTANCE, num_screens, WAVELENGTH, CN2, L0, L0_INNER)
        result = analyze_turbulence_effects(beam, layers, DISTANCE, N=N_GRID, 
                                           oversampling=OVERSAMPLING, L0=L0, l0=L0_INNER)

    # --- Generate and Save Comparison Plots ---
    print("\n--- Generating Comparison Plots ---")
    
    output_dir = os.path.join(SCRIPT_DIR, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plot_multi_layer_effects(beam, NUM_SCREENS_OPTIONS, DISTANCE, CN2, L0, L0_INNER,
                                   N=N_GRID, oversampling=OVERSAMPLING,
                                   save_path=os.path.join(output_dir, 'turbulence_diagnostic.png'))

    plt.show()
    print("Diagnostic run complete.")