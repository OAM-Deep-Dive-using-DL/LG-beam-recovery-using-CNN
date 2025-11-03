import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, factorial
import os

# --- Simulation Parameters ---
# These are now defined in the __main__ block (see Improvement 4)
# to make the script reusable as a module.


class LaguerreGaussianBeam:
    def __init__(self, p, l, wavelength, w0):
        """
        Initializes the Laguerre-Gaussian beam parameters based on
        canonical formulations.
        
        References:
        - Andrews & Phillips, "Laser Beam Propagation through Random Media" (2005)
        - Siegman, "Lasers" (1986)
        - Beijersbergen et al., "Astigmatic laser mode converters and transfer of orbital angular momentum" (1993)
        
        Parameters:
        -----------
        p : int
            Radial index (number of radial nodes)
        l : int
            Azimuthal index (OAM quantum number, can be positive or negative)
        wavelength : float
            Optical wavelength [m]
        w0 : float
            Beam waist radius at z=0 [m] (1/e^2 intensity radius)
        """
        if p < 0:
            raise ValueError(f"Radial index p must be non-negative, got p={p}")
        
        self.p = p
        self.l = l
        self.wavelength = wavelength
        self.w0 = w0  # Fundamental Gaussian waist parameter
        self.k = 2 * np.pi / wavelength

        # M-squared factor [Ref: Siegman, 1986, Eq. 13.4-1]
        # For LG modes: M² = (2p + |l| + 1)
        self.M_squared = 2 * p + abs(l) + 1
        
        # Effective Rayleigh range for the M² beam [Ref: Andrews & Phillips, 2005, Eq. 4.40]
        # CRITICAL: z_R = z₀/M² where z₀ = πw₀²/λ is the fundamental Rayleigh range
        # Since divergence scales with M² (θ_eff = M²·θ₀), the collimation distance
        # scales inversely: z_R = z₀/M². This ensures w(z) = w₀√(1+(z/z_R)²) holds.
        # See Siegman (1986) M² formalism: all Gaussian formulas work when z_R is scaled.
        z0_fundamental = (np.pi * w0**2) / wavelength
        self.z_R = z0_fundamental / self.M_squared
        
        # Normalization constant for orthonormal LG modes [Ref: A&P2005, Eq. 4.35]
        # This ensures ∫∫ |E|² dA = 1 for unit power
        # Formula: C_norm = sqrt(2 * factorial(p) / (π * factorial(p + |l|)))
        # Note: factorial(0) = 1, so this formula works correctly for all cases including p=0, l=0
        self.C_norm = np.sqrt(2.0 * factorial(p) / (np.pi * factorial(p + abs(l))))

    def beam_waist(self, z):
        """
        Calculates the z-dependent Gaussian spot size parameter w(z).
        
        For LG modes, the beam waist evolution follows the universal formula:
        w(z) = w₀ * sqrt(1 + (z/z_R)²)
        
        CRITICAL: z_R here is the EFFECTIVE Rayleigh range: z_R = z₀/M²
        where z₀ = πw₀²/λ is the fundamental Rayleigh range.
        This ensures the formula correctly predicts divergence for M² > 1 beams.
        
        References:
        - Siegman (1986), Eq. 16.7-1 (M² formalism)
        - Andrews & Phillips (2005), Eq. 4.37
        """
        return self.w0 * np.sqrt(1 + (z / self.z_R) ** 2)

    def radius_of_curvature(self, z):
        """
        Calculates the z-dependent radius of curvature R(z) of the wavefront.
        
        For Gaussian/LG beams: R(z) = z * (1 + (z_R/z)²)
        At z=0, R→∞ (plane wavefront)
        At z=z_R, R=2z_R (minimum curvature)
        
        References:
        - Siegman (1986), Eq. 16.7-2
        - Andrews & Phillips (2005), Eq. 4.38
        
        Parameters:
        -----------
        z : float
            Propagation distance [m]
        
        Returns:
        --------
        R_z : float
            Radius of curvature [m] (positive for diverging beam)
        """
        if abs(z) < 1e-12:
            return np.inf  # Plane wave at z=0
        return z * (1 + (self.z_R / z) ** 2)

    def gouy_phase(self, z):
        """
        Calculates the z-dependent Gouy phase Psi(z).
        
        For LG modes, the Gouy phase is: ψ(z) = (2p + |l| + 1) * arctan(z/z_R)
        This is equivalent to M² * arctan(z/z_R) since M² = 2p + |l| + 1
        
        References:
        - Andrews & Phillips (2005), Eq. 4.36
        - Siegman (1986), Sec. 16.3
        """
        return self.M_squared * np.arctan(z / self.z_R)

    # IMPROVEMENT 3: Use @property for derived attributes
    # This is now accessed as 'beam.effective_divergence_angle'
    @property
    def effective_divergence_angle(self):
        """
        Returns the fundamental (theta_0) and effective (Theta) far-field divergence angles.
        
        The divergence angle scales with M²:
        - θ₀ = λ/(πw₀): Fundamental Gaussian divergence
        - θ_eff = M² * θ₀: Effective divergence for LG mode
        
        References:
        - Siegman (1986), Eq. 16.7-3
        - Andrews & Phillips (2005), Eq. 4.39
        
        Returns:
        --------
        theta_0 : float
            Fundamental divergence angle [rad]
        theta_eff : float
            Effective divergence angle [rad] (scaled by M²)
        """
        theta_0 = self.wavelength / (np.pi * self.w0)
        theta_eff = self.M_squared * theta_0
        return theta_0, theta_eff

    # IMPROVEMENT 3: Use @property for derived attributes
    # This is now accessed as 'beam.group_velocity_ratio'
    @property
    def group_velocity_ratio(self):
        """
        Returns the group velocity ratio (v_g / c) for the mode.
        """
        return 1.0 / (1.0 + self.M_squared / (2.0 * (self.k * self.w0)**2))

    def generate_beam_field(self, r, phi, z):
        """
        Generates the complex electric field E(r, phi, z) based on the full
        analytical solution for Laguerre-Gaussian beams.
        
        The field expression follows the standard form:
        E(r,φ,z) = C_norm * (w₀/w(z)) * (√2 r/w(z))^|l| * L_p^|l|(2r²/w²) 
                   * exp(-r²/w²) * exp(-ilφ) * exp(-ikr²/(2R)) 
                   * exp(-i(2p+|l|+1)ψ) * exp(ikz)
        
        References:
        - Andrews & Phillips (2005), Eq. 4.34
        - Beijersbergen et al. (1993)
        - Siegman (1986), Sec. 16.4
        
        Parameters:
        -----------
        r : array_like
            Radial coordinate [m]
        phi : array_like
            Azimuthal angle [rad]
        z : float
            Propagation distance [m]
        
        Returns:
        --------
        field : array_like
            Complex electric field [V/m] (normalized for unit power)
        """
        # Get z-dependent parameters
        w_z = self.beam_waist(z)
        R_z = self.radius_of_curvature(z)
        psi_z = self.gouy_phase(z)

        # Generalized Laguerre polynomial L_p^|l|(2r²/w²)
        # Note: genlaguerre(p, |l|) gives the correct polynomial
        L_p_l = genlaguerre(self.p, abs(self.l))(2 * r**2 / w_z**2)

        # --- Assemble Field Components [Ref: A&P2005, Eq. 4.34] ---

        # Amplitude decay and normalization
        # CORRECTED: Standard form uses (w₀/w(z)) factor for amplitude scaling
        # The normalization constant C_norm ensures orthonormality
        amplitude_factor = self.C_norm * (self.w0 / w_z)
        
        # Radial profile (Gaussian * Laguerre * r^|l|)
        # Standard form: (√2 r/w(z))^|l| * L_p^|l|(2r²/w²) * exp(-r²/w²)
        radial_factor = (
            (np.sqrt(2) * r / w_z) ** abs(self.l) * L_p_l * np.exp(-(r**2) / w_z**2)
        )
        
        # Azimuthal (OAM) helical phase: exp(-ilφ)
        # This gives the beam its orbital angular momentum
        azimuthal_factor = np.exp(-1j * self.l * phi)

        # Phase from wavefront radius of curvature: exp(-ikr²/(2R))
        # At z=0, R→∞, so this term is zero
        if np.isinf(R_z):
            curvature_phase = 0  # Plane wave at z=0
        else:
            curvature_phase = -1j * self.k * r**2 / (2 * R_z)

        # Gouy phase term: exp(-i(2p+|l|+1)ψ) = exp(-i*M²*ψ)
        # This accounts for the phase shift through focus
        gouy_phase_term = np.exp(-1j * psi_z)

        # Carrier propagation phase: exp(ikz)
        # Standard plane wave phase term
        propagation_phase = np.exp(1j * self.k * z)

        # Total complex field (all terms multiplied)
        field = (
            amplitude_factor
            * radial_factor
            * azimuthal_factor
            * np.exp(curvature_phase)
            * gouy_phase_term
            * propagation_phase
        )

        return field

    def calculate_intensity(self, r, phi, z):
        """
        Calculates the intensity (magnitude-squared) of the field.
        """
        field = self.generate_beam_field(r, phi, z)
        return np.abs(field) ** 2

    def get_beam_parameters(self, z):
        """Helper function to return a dictionary of parameters at z."""
        return {
            "z": z,
            "w_z": self.beam_waist(z),
            "R_z": self.radius_of_curvature(z),
            "gouy_phase": self.gouy_phase(z),
            "beam_expansion": self.beam_waist(z) / self.w0,
        }

    def propagation_summary(self, z_distances):
        """Prints a formatted table of beam parameters vs. distance."""
        print(f"\nPropagation Summary for LG_{self.p}^{self.l} beam:")
        print("Distance (m) | w(z) (mm) | R(z) (m) | Gouy Phase (rad) | w(z)/w0")
        print("-" * 65)

        for z in z_distances:
            params = self.get_beam_parameters(z)
            w_z_mm = params["w_z"] * 1e3
            R_z = params["R_z"]
            
            R_str = "∞" if np.isinf(R_z) else f"{R_z:.2f}"

            print(
                f"{z:8.1f}   | {w_z_mm:7.2f}   | {R_str:>8} | {params['gouy_phase']:11.3f}    | {params['beam_expansion']:.2f}"
            )

    def __str__(self):
        """String representation for printing the beam object."""
        return (
            f"LG_{self.p}^{self.l} beam (λ={self.wavelength * 1e9:.0f}nm, "
            f"w0={self.w0 * 1e3:.1f}mm, z_R={self.z_R:.2f}m, M²={self.M_squared})"
        )

    def __repr__(self):
        """Official string representation."""
        return f"LaguerreGaussianBeam(p={self.p}, l={self.l}, λ={self.wavelength}, w0={self.w0})"


# IMPROVEMENT 4: Plotting logic is now in its own function
def plot_beam_analysis(beam, grid_size, max_radius_mm, save_fig=False, plot_dir="plots"):
    """
    Generates and plots the 4-panel analysis for a given LG beam.
    
    Args:
        beam (LaguerreGaussianBeam): The beam object to analyze.
        grid_size (int): The N x N grid size for 2D plots.
        max_radius_mm (float): The extent of the 2D plots in mm.
        save_fig (bool): Whether to save the figure to disk.
        plot_dir (str): Directory to save the figure in.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Laguerre-Gaussian Beam Analysis: $LG_{{{beam.p}}}^{{{beam.l}}}$', 
                 fontsize=20, fontweight='bold')
    
    # --- Panel 1: Lateral Profile at z=0 ---
    ax1 = axes[0, 0]
    r_range = np.linspace(0, max_radius_mm * 1e-3, grid_size * 2) # Higher res for 1D
    intensity_profile = beam.calculate_intensity(r_range, 0, 0)
    
    ax1.plot(r_range * 1e3, intensity_profile, 'b-', linewidth=2)
    ax1.axvline(beam.w0 * 1e3, color='r', linestyle='--', alpha=0.7, 
                label=f'w₀={beam.w0 * 1e3:.1f}mm')
    ax1.set_xlabel('Radial Position r [mm]')
    ax1.set_ylabel('Intensity [a.u.]') 
    ax1.set_title(f'Lateral Profile at z=0')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Setup for 2D Plots (Intensity & Phase) ---
    r_max_m = max_radius_mm * 1e-3
    x = np.linspace(-r_max_m, r_max_m, grid_size)
    y = np.linspace(-r_max_m, r_max_m, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X) # 4-quadrant arctan is crucial

    # IMPROVEMENT 1: Calculate the complex field ONCE
    field_z0 = beam.generate_beam_field(R, PHI, 0)
    
    # --- Panel 2: Intensity at z=0 ---
    ax2 = axes[0, 1]
    intensity_z0 = np.abs(field_z0) ** 2 # Derive intensity
    
    im2 = ax2.imshow(intensity_z0, 
                     extent=[-max_radius_mm, max_radius_mm, -max_radius_mm, max_radius_mm],
                     cmap='hot', origin='lower', interpolation='bilinear')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title(f'Intensity at z=0')
    ax2.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Intensity [a.u.]')

    # --- Panel 3: Phase at z=0 ---
    ax3 = axes[1, 0]
    phase_z0 = np.angle(field_z0) # Derive phase
    
    im3 = ax3.imshow(phase_z0, 
                     extent=[-max_radius_mm, max_radius_mm, -max_radius_mm, max_radius_mm],
                     cmap='twilight', origin='lower', interpolation='bilinear',
                     vmin=-np.pi, vmax=np.pi)
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.set_title(f'Phase at z=0 (OAM={beam.l}ℏ)')
    ax3.set_aspect('equal')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Phase [rad]')
    cbar3.set_ticks([-np.pi, 0, np.pi])
    cbar3.set_ticklabels(['-π', '0', 'π'])

    # --- Panel 4: Longitudinal Propagation ---
    ax_long = axes[1, 1]
    
    z_max = 3 * beam.z_R  
    num_z_steps = 150
    num_r_steps = 200
    
    z_array = np.linspace(0, z_max, num_z_steps)
    r_max_long = 3 * beam.beam_waist(z_max) # Plot 3x the final waist
    r_array = np.linspace(-r_max_long, r_max_long, num_r_steps)
    
    # Create the longitudinal intensity map
    intensity_long = np.zeros((num_r_steps, num_z_steps))
    for i, z in enumerate(z_array):
        # Calculate 1D intensity slice. phi=0 is arbitrary and fine.
        intensity_long[:, i] = beam.calculate_intensity(np.abs(r_array), 0, z)
    
    z_array_km = z_array / 1000
    z_max_km = z_max / 1000
    r_max_mm = r_max_long * 1e3
    
    im_long = ax_long.imshow(intensity_long, 
                             extent=[0, z_max_km, -r_max_mm, r_max_mm],
                             aspect='auto', cmap='hot', origin='lower', 
                             interpolation='bilinear')
    
    # Overlay analytical w(z) as a sanity check
    w_z_array_mm = np.array([beam.beam_waist(z) * 1e3 for z in z_array])
    ax_long.plot(z_array_km, w_z_array_mm, 'c--', linewidth=2.5, label='w(z)')
    ax_long.plot(z_array_km, -w_z_array_mm, 'c--', linewidth=2.5)
    
    # Mark the Rayleigh range
    ax_long.axvline(beam.z_R / 1000, color='lime', linestyle=':', linewidth=2.5,
                   label=f'z_R={beam.z_R / 1000:.3f}km', alpha=0.8)
    
    ax_long.set_xlabel('Propagation Distance z [km]', fontsize=12)
    ax_long.set_ylabel('Radial Position r [mm]', fontsize=12)
    ax_long.set_title(f'Longitudinal Propagation (M²={beam.M_squared})',
                     fontsize=13)
    ax_long.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax_long.grid(True, alpha=0.3)
    plt.colorbar(im_long, ax=ax_long, label='Intensity [a.u.]', 
                 fraction=0.046, pad=0.04)
    
    # --- Finalize and Show/Save Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    if save_fig:
        os.makedirs(plot_dir, exist_ok=True)
        fig_name = f"lg_p{beam.p}_l{beam.l}_beam.png"
        save_path = os.path.join(plot_dir, fig_name)
        print(f"Saving figure to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # dpi=300 is good for screens
    
    plt.show()



def main():
    """
    Main driver script to initialize and analyze an LG beam.
    """
    # --- Define Simulation Parameters ---
    WAVELENGTH = 1550e-9  
    W0 = 25e-3           
    P_MODE = 0         
    L_MODE = 2          
    GRID_SIZE = 200       
    MAX_RADIUS_MM = 75   
    SAVE_FIGURE = True  
    PLOT_DIR = "plots"   

    # --- 1. Initialize Beam ---
    beam = LaguerreGaussianBeam(P_MODE, L_MODE, WAVELENGTH, W0)
    
    # --- 2. Print Console Summary ---
    print(f"Beam Definition: {beam}")

    # Accessing properties (no '()')
    theta_0, theta_eff = beam.effective_divergence_angle
    print(f"Ideal Divergence (θ₀): {theta_0 * 1e6:.2f} µrad")
    print(f"Effective Divergence (Θ): {theta_eff * 1e6:.2f} µrad")

    vg_ratio = beam.group_velocity_ratio
    print(f"Group Velocity Ratio (v_g / c): {vg_ratio:.8f}")

    params_at_zR = beam.get_beam_parameters(beam.z_R)
    print(f"\nParams at z=z_R ({beam.z_R:.1f}m):")
    print(f"  w(z_R) = {params_at_zR['w_z']*1e3:.2f} mm")
    print(f"  R(z_R) = {params_at_zR['R_z']:.2f} m")

    z_list = [0, beam.z_R, 2 * beam.z_R, 3 * beam.z_R]
    beam.propagation_summary(z_list)
    
    # --- 3. Generate and Show Plots ---

    plot_beam_analysis(beam, GRID_SIZE, MAX_RADIUS_MM, SAVE_FIGURE, PLOT_DIR)


if __name__ == "__main__":
    main()