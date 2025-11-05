import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings


try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

# --- Import Required Modules ---
try:
    # 1. From lgBeam.py
    from lgBeam import LaguerreGaussianBeam
    
    # 2. From encoding.py
    #    (Assuming your transmitter file is named 'encoding.py')
    from encoding import encodingRunner 
    
    # 3. From turbulence.py
    from turbulence import (
        AtmosphericTurbulence,
        create_multi_layer_screens,
        apply_multi_layer_turbulence  # The rectified function
    )
    
    # 4. From fsplAtmAttenuation.py
    #    (Assuming your link budget file is 'fsplAtmAttenuation.py')
    from fsplAtmAttenuation import calculate_kim_attenuation, calculate_geometric_loss
    
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("Please ensure lgBeam.py, encoding.py, turbulence.py, and fsplAtmAttenuation.py are in the same directory.")
    sys.exit(1)

warnings.filterwarnings('ignore')
np.random.seed(41)


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """
    Centralized configuration for the FSO-OAM system.
    """
    # Optical Parameters
    WAVELENGTH = 1550e-9  # [m] C-band telecom wavelength
    W0 = 25e-3           # [m] Initial beam waist
    
    # Link Parameters
    DISTANCE = 1000      # [m] Propagation distance (1 km)
    RECEIVER_DIAMETER = 0.3  # [m] 20 cm aperture
    P_TX_TOTAL_W = 1.0     # [W] Total transmit power (30 dBm)
    
    # Spatial Modes (p, l) - Mode Division Multiplexing
    SPATIAL_MODES = [(0, 0), (0, -1), (0, 1), (0, -3), (0, 3)]
    
    # Turbulence Parameters
    CN2 = 1e-13          # [m^(-2/3)] Strong turbulence
    L0 = 10.0           # [m] Outer scale
    L0_INNER = 0.005    # [m] Inner scale (5 mm)
    NUM_SCREENS = 20    # Number of phase screens
    
    # Weather Condition
    WEATHER = 'clear'    # Options: 'clear', 'haze', 'light_fog'
    
    # Encoding Parameters
    FEC_RATE = 0.8      
    PILOT_RATIO = 0.1   
    N_INFO_BITS = 4096  
    
    # Simulation Grid
    N_GRID = 1024        
    OVERSAMPLING = 2    
    
    # Output
    PLOT_DIR = os.path.join(SCRIPT_DIR, "pipeline_results")
    DPI = 600             


# ============================================================================
# MAIN PIPELINE CLASS (RECTIFIED)
# ============================================================================

class FSOPipeline:
    
    def __init__(self, config):
        self.cfg = config
        os.makedirs(self.cfg.PLOT_DIR, exist_ok=True)
        
        print("\n" + "="*70)
        print("FSO-OAM COMMUNICATION SYSTEM PIPELINE")
        print("="*70)
        
        # 1. Initialize encoding system
        print("\n[1] Initializing Transmitter...")
        self.encoder = encodingRunner(
            spatial_modes=self.cfg.SPATIAL_MODES,
            wavelength=self.cfg.WAVELENGTH,
            w0=self.cfg.W0,
            fec_rate=self.cfg.FEC_RATE,
            pilot_ratio=self.cfg.PILOT_RATIO
        )
        
        # 2. Initialize turbulence model (uses Cn2 from config)
        print("\n[2] Initializing Atmospheric Turbulence Model...")
        self.turb = AtmosphericTurbulence(
            self.cfg.CN2, self.cfg.L0, self.cfg.L0_INNER, self.cfg.WAVELENGTH
        )
        self.total_r0 = self.turb.fried_parameter(self.cfg.DISTANCE)
        self.ry_var = self.turb.rytov_variance(self.cfg.DISTANCE) 
        
        print(f"    Turbulence strength: {self.turb.turbulence_strength()}")
        print(f"    Fried parameter r0: {self.total_r0*1000:.2f} mm")
        print(f"    Rytov variance: {self.ry_var:.3f}")
        
        # 3. Create phase screens
        print(f"\n[3] Generating {self.cfg.NUM_SCREENS} Phase Screens...")
        self.layers = create_multi_layer_screens(
            self.cfg.DISTANCE, self.cfg.NUM_SCREENS, 
            self.cfg.WAVELENGTH, self.cfg.CN2, 
            self.cfg.L0, self.cfg.L0_INNER,
            verbose=False  ### --- FIX --- ### Quieten the output for 50 screens
        )
        print(f"    Generated {len(self.layers)} screen layers.")
        
        # Initialize storage
        self.tx_signals = None
        self.grid_info = None
        self.total_tx_field = None
        self.total_rx_field_turbulent = None
        self.total_rx_field_pristine = None # Will be set by apply_turbulence
        self.rx_results = {}

        
    def run_transmitter_and_field_gen(self):
        """
        Stages 1 & 2: Generate data AND the combined z=0 field.
        """
        print("\n" + "="*70)
        print("STAGES 1 & 2: TRANSMITTER & FIELD GENERATION")
        print("="*70)
        
        # --- Stage 1: Generate Data ---
        data_bits = np.random.randint(0, 2, self.cfg.N_INFO_BITS)
        print(f"\nGenerated {self.cfg.N_INFO_BITS} random information bits")
        self.tx_signals = self.encoder.transmit(data_bits, verbose=True)
        
        # --- Stage 2: Generate z=0 Field ---
        
        # Grid sizing
        self.max_m2_beam = max(self.encoder.lg_beams.values(), key=lambda beam: beam.M_squared)
        beam_size_at_rx = self.max_m2_beam.beam_waist(self.cfg.DISTANCE)
        D = self.cfg.OVERSAMPLING * 6 * beam_size_at_rx
        delta = D / self.cfg.N_GRID
        
        print(f"\nGrid sized for max M²={self.max_m2_beam.M_squared:.1f} (LG_{self.max_m2_beam.p}^{self.max_m2_beam.l}), w(z)={beam_size_at_rx*1000:.1f}mm")
        print(f"  Grid size: D = {D:.2f} m, delta = {delta*1000:.2f} mm")
        
        # Check sampling
        sampling_ratio = delta / self.cfg.L0_INNER
        print(f"  Sampling check: Δx/l0 = {sampling_ratio:.3f} ", end="")
        if sampling_ratio < 1.0:
            print("✓ (adequate)")
        else:
            print(f"✗ (WARNING: undersampled! Δx > l0. Need N > {int(D / self.cfg.L0_INNER)})")

        x = np.linspace(-D/2, D/2, self.cfg.N_GRID)
        y = np.linspace(-D/2, D/2, self.cfg.N_GRID)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        self.grid_info = {'x': x, 'y': y, 'X': X, 'Y': Y, 'R': R, 'PHI': PHI, 'D': D, 'delta': delta}
        
        # Generate and SUM the fields
        print(f"\nGenerating and combining {len(self.cfg.SPATIAL_MODES)} modes at z=0...")
        
        self.total_tx_field = np.zeros((self.cfg.N_GRID, self.cfg.N_GRID), dtype=complex)
        
        ### --- FIX --- ###
        # REMOVED: self.total_rx_field_pristine = np.zeros_like(self.total_tx_field)
        # This field will now be generated by the numerical propagation.

        for mode_key in self.cfg.SPATIAL_MODES:
            beam = self.encoder.lg_beams[mode_key]
            
            if self.tx_signals[mode_key]['n_symbols'] > 0:
                symbol = self.tx_signals[mode_key]['symbols'][0]
            else:
                symbol = 1.0 + 0j
            
            # Field at z=0
            field_z0 = beam.generate_beam_field(R, PHI, 0) * symbol
            self.total_tx_field += field_z0
            
            ### --- FIX --- ###
            # REMOVED: Analytical calculation of the pristine field at z=L.
            # field_zL = beam.generate_beam_field(R, PHI, self.cfg.DISTANCE) * symbol
            # self.total_rx_field_pristine += field_zL
        
        # Normalize total transmitted field power
        total_power_z0 = np.sum(np.abs(self.total_tx_field)**2)
        norm_factor = np.sqrt(self.cfg.P_TX_TOTAL_W / total_power_z0)
        
        self.total_tx_field *= norm_factor
        
        ### --- FIX --- ###
        # REMOVED: self.total_rx_field_pristine *= norm_factor
        
        print(f"  Initial field power normalized to {self.cfg.P_TX_TOTAL_W:.2f} W")
            
    
    def apply_turbulence(self):
        """
        Stage 3: Propagate the COMBINED field through turbulence.
        """
        print("\n" + "="*70)
        print("STAGE 3: TURBULENT PROPAGATION")
        print("="*70)
        
        print(f"\nPropagating multiplexed field (all {len(self.cfg.SPATIAL_MODES)} modes combined)...")
        
        ### --- FIX --- ###
        # Pass the beam with the largest M^2 as the base beam.
        # This ensures the grid sizing in apply_multi_layer_turbulence
        # matches the grid we just created.
        base_beam = self.max_m2_beam
        
        result = apply_multi_layer_turbulence(
            self.total_tx_field, # Pass the combined z=0 field
            base_beam, 
            self.layers, 
            self.cfg.DISTANCE,
            N=self.cfg.N_GRID, 
            oversampling=self.cfg.OVERSAMPLING,
            L0=self.cfg.L0, 
            l0=self.cfg.L0_INNER
        )
        
        ### --- FIX --- ###
        # Store BOTH the turbulent and the numerically propagated pristine field
        # for a correct 1-to-1 comparison.
        self.total_rx_field_turbulent = result['final_field']
        self.total_rx_field_pristine = result['pristine_field'] 
        print("  Propagation complete")

    
    def apply_attenuation_and_analyze(self):
        """
        Stages 4 & 5: Apply attenuation and analyze results.
        """
        print("\n" + "="*70)
        print("STAGES 4 & 5: ATTENUATION & RECEIVER ANALYSIS")
        print("="*70)
        
        wavelength_nm = self.cfg.WAVELENGTH * 1e9
        
        # --- Stage 4: Attenuation ---
        print("\n[4] Path Loss Calculation")
        
        visibility_map_km = {
            'clear': 23, 'haze': 4, 'light_fog': 0.5,
            'moderate_fog': 0.2, 'dense_fog': 0.05
        }
        visibility_km = visibility_map_km.get(self.cfg.WEATHER.lower(), 23)

        print(f"  Using worst-case beam for path loss: LG_p={self.max_m2_beam.p}, l={self.max_m2_beam.l}, M²={self.max_m2_beam.M_squared:.1f}")

        # Calculate losses
        w_z_analytical = self.max_m2_beam.beam_waist(self.cfg.DISTANCE)
        L_geo_dB, eta = calculate_geometric_loss(w_z_analytical, self.cfg.RECEIVER_DIAMETER / 2.0)
        
        alpha_dBkm = calculate_kim_attenuation(wavelength_nm, visibility_km)
        L_atm_dB = alpha_dBkm * (self.cfg.DISTANCE / 1000.0)
        
        L_total_dB_analytical = L_geo_dB + L_atm_dB
        
        print(f"  Weather: {self.cfg.WEATHER}, Alpha: {alpha_dBkm:.2f} dB/km")
        print(f"  Geometric loss: {L_geo_dB:.2f} dB")
        print(f"  Atmospheric loss: {L_atm_dB:.2f} dB")
        print(f"  Total path loss (analytical): {L_total_dB_analytical:.2f} dB")
        print(f"  Collection efficiency: {eta*100:.1f}%")
        
        # Apply *only* atmospheric loss to simulated fields
        # (Geometric loss is already included in the propagation)
        amplitude_loss = 10**(-L_atm_dB / 20.0)
        
        ### --- FIX --- ###
        # These variables are now the correct numerical fields.
        final_field = self.total_rx_field_turbulent * amplitude_loss
        pristine_field = self.total_rx_field_pristine * amplitude_loss
        
        # --- Stage 5: Analysis ---
        print("\n[5] Performance Metrics")
        
        pristine_int = np.abs(pristine_field)**2
        final_int = np.abs(final_field)**2
        
        if np.isnan(final_int).any():
            print("  ✗ ERROR: Simulation failed. Field contains NaNs.")
            return

        # Calculate metrics on the *correct* fields
        pristine_max = np.max(pristine_int)
        final_max = np.max(final_int)
        strehl = final_max / (pristine_max + 1e-12)
        
        receiver_radius = self.cfg.RECEIVER_DIAMETER / 2.0
        aperture_mask_bool = self.grid_info['R'] <= receiver_radius
        
        intensity_in_aperture = final_int[aperture_mask_bool]
        
        if intensity_in_aperture.size > 0:
            scintillation = np.var(intensity_in_aperture) / (np.mean(intensity_in_aperture)**2 + 1e-12)
        else:
            scintillation = 0.0
        
        print(f"  Strehl ratio: {strehl:.3f}")
        print(f"  Scintillation index (σ_I²): {scintillation:.3f}")

        # Power calculations
        P_tx_dBm = 10 * np.log10(self.cfg.P_TX_TOTAL_W * 1000.0)
        P_rx_analytical_dBm = P_tx_dBm - L_total_dB_analytical
        
        # Calculate received power from *simulation*
        receiver_radius = self.cfg.RECEIVER_DIAMETER / 2.0
        aperture_mask = (self.grid_info['R'] <= receiver_radius).astype(float)
        
        # We find the *ratio* of power received to power transmitted
        # in the simulation, and apply that ratio to the real P_TX.
        
        power_total_tx_sim = np.sum(np.abs(self.total_tx_field)**2)
        power_in_turbulent_sim = np.sum(final_int * aperture_mask)
        
        ### --- FIX --- ###
        # Also check pristine power for a "turbulence-free" sim loss
        power_in_pristine_sim = np.sum(pristine_int * aperture_mask)
        ptr_pristine = power_in_pristine_sim / power_total_tx_sim
        P_rx_sim_pristine_W = self.cfg.P_TX_TOTAL_W * ptr_pristine
        P_rx_sim_pristine_dBm = 10 * np.log10(P_rx_sim_pristine_W * 1000.0)

        # Power Transfer Ratio (PTR)
        ptr_turbulent = power_in_turbulent_sim / power_total_tx_sim
        
        P_rx_sim_W = self.cfg.P_TX_TOTAL_W * ptr_turbulent
        P_rx_sim_dBm = 10 * np.log10(P_rx_sim_W * 1000.0)
        
        print(f"\n[Power Budget]")
        print(f"  Received power (analytical, no turb): {P_rx_analytical_dBm:.2f} dBm")
        print(f"  Received power (simulated, no turb):  {P_rx_sim_pristine_dBm:.2f} dBm")
        print(f"  Received power (simulated with turb): {P_rx_sim_dBm:.2f} dBm")
        
        ### --- FIX --- ###
        # This loss is now a direct sim-to-sim comparison
        add_loss = P_rx_sim_pristine_dBm - P_rx_sim_dBm
        print(f"  Additional turbulence loss: {add_loss:.2f} dB")
        
        # Store results
        self.rx_results = {
            'strehl': strehl,
            'scintillation': scintillation,
            'pristine_int': pristine_int,
            'final_int': final_int,
            'pristine_phase': np.angle(pristine_field),
            'final_phase': np.angle(final_field),
            'log_data': {
                'L_total_dB_analytical': L_total_dB_analytical,
                'P_rx_analytical_dBm': P_rx_analytical_dBm,
                'P_rx_sim_pristine_dBm': P_rx_sim_pristine_dBm, ### --- FIX --- ###
                'P_rx_sim_dBm': P_rx_sim_dBm,
                'add_loss_dB': add_loss
            }
        }

    def plot_complete_pipeline(self):
        """
        Generate comprehensive visualization of the entire pipeline.
        """
        print("\n" + "="*70)
        print("GENERATING PIPELINE VISUALIZATION")
        print("="*70)
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f'FSO-OAM Multiplexed Pipeline (Cn²={self.cfg.CN2:.1e}, L={self.cfg.DISTANCE}m)', 
                     fontsize=18, fontweight='bold')
        
        extent_mm = self.grid_info['D'] * 1e3 / 2
        res = self.rx_results
        
        # --- Plot Initial (z=0) Combined Field ---
        tx_int = np.abs(self.total_tx_field)**2
        vmax_tx = np.percentile(tx_int, 99.9)
        
        im1 = axes[0].imshow(tx_int.T, 
                        extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                        cmap='hot', origin='lower', vmax=vmax_tx)
        axes[0].set_title(f'TX: Combined Field (z=0)', fontweight='bold')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, label='Intensity')
        
        # --- Plot Pristine RX Field ---
        vmax_pristine = np.percentile(res['pristine_int'], 99.9)
        im2 = axes[1].imshow(res['pristine_int'].T,
                        extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                        cmap='hot', origin='lower', vmax=vmax_pristine)
        axes[1].set_title(f'RX: Pristine Intensity (z=L)', fontweight='bold')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Intensity')

        # --- Plot Distorted RX Intensity ---
        vmax_final = np.percentile(res['final_int'], 99.9)
        im3 = axes[2].imshow(res['final_int'].T,
                        extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                        cmap='hot', origin='lower', vmax=vmax_final)
        axes[2].set_title(f'RX: Distorted Intensity (S={res["strehl"]:.3f})', fontweight='bold')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, label='Intensity')
        
        # --- Plot Distorted RX Phase ---
        im4 = axes[3].imshow(res['final_phase'].T,
                        extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                        cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
        axes[3].set_title(f'RX: Distorted Phase (σ$_I^2$={res["scintillation"]:.3f})', fontweight='bold')
        plt.colorbar(im4, ax=axes[3], fraction=0.046, label='Phase [rad]')
        
        for ax in axes:
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        save_path = os.path.join(self.cfg.PLOT_DIR, 'complete_pipeline.png')
        plt.savefig(save_path, dpi=self.cfg.DPI, bbox_inches='tight')
        print(f"\n✓ Saved pipeline visualization to: {save_path}")
        
        return fig
    
    def print_summary(self):
        """
        Print comprehensive system summary.
        """
        print("\n" + "="*70)
        print("SYSTEM PERFORMANCE SUMMARY")
        print("="*70)
        
        print("\n[LINK PARAMETERS]")
        print(f"  Distance: {self.cfg.DISTANCE} m ({self.cfg.DISTANCE/1000:.1f} km)")
        print(f"  Wavelength: {self.cfg.WAVELENGTH*1e9:.0f} nm")
        print(f"  Weather: {self.cfg.WEATHER}")
        print(f"  Turbulence: Cn²={self.cfg.CN2:.2e} m^(-2/3) ({self.turb.turbulence_strength()})")
        print(f"  Rytov Variance: {self.ry_var:.3f}") 
        
        print("\n[SPATIAL MODES]")
        for mode_key in self.cfg.SPATIAL_MODES:
            beam = self.encoder.lg_beams[mode_key]
            p, l = mode_key
            print(f"  LG_{p}^{l}: M²={beam.M_squared:.1f}, "
                  f"z_R={beam.z_R:.1f}m, "
                  f"w(L)={beam.beam_waist(self.cfg.DISTANCE)*1000:.1f}mm")
        
        print("\n[PERFORMANCE METRICS]")
        print(f"{'Metric':<30} {'Value':<12}")
        print("-" * 42)
        
        if self.rx_results:
            res = self.rx_results
            log_data = res['log_data']
            print(f"{'Strehl Ratio:':<30} {res['strehl']:.4f}")
            print(f"{'Scintillation Index (σ_I²):':<30} {res['scintillation']:.4f}")
            print(f"{'Total Analytical Loss:':<30} {log_data['L_total_dB_analytical']:.2f} dB")
            print(f"{'Analytical Rx Power:':<30} {log_data['P_rx_analytical_dBm']:.2f} dBm")
            ### --- FIX --- ###
            print(f"{'Simulated Rx Power (Pristine):':<30} {log_data['P_rx_sim_pristine_dBm']:.2f} dBm")
            print(f"{'Simulated Rx Power (Turb):':<30} {log_data['P_rx_sim_dBm']:.2f} dBm")
            print(f"{'Turbulence Loss (Sim):':<30} {log_data['add_loss_dB']:.2f} dB")
        else:
            print("  No results to display.")
        
        print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    config = SimulationConfig()
    pipeline = FSOPipeline(config)
    
    try:
        pipeline.run_transmitter_and_field_gen()
        pipeline.apply_turbulence()
        pipeline.apply_attenuation_and_analyze()
        pipeline.plot_complete_pipeline()
        pipeline.print_summary()
        
        print("\n✓ Pipeline execution completed successfully!")
        print(f"✓ Results saved to: {config.PLOT_DIR}")
        
        plt.show()
        
    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




# ==================================================




# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# from dataclasses import dataclass
# from typing import Dict, Optional, Any

# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     SCRIPT_DIR = os.getcwd()
# sys.path.insert(0, SCRIPT_DIR)

# try:
#     from lgBeam import LaguerreGaussianBeam 
# except ImportError:
#     print(f"Error: Could not find 'lgBeam.py' in the script directory: {SCRIPT_DIR}")
#     print("Please make sure the lgBeam.py file is saved in the same folder.")
#     sys.exit(1)

# warnings.filterwarnings('ignore')
# np.random.seed(42)


# @dataclass
# class FSO_MDM_Frame:
#     """
#     Unified data interface for FSO-MDM transmission frames.
    
#     Provides standardized structure for transmitter→channel→receiver flow.
#     Includes metadata for mode identification, pilot positions, and grid information.
#     """
#     # Core data: symbol streams per mode (for backward compatibility)
#     tx_signals: Dict[tuple, Dict[str, Any]]
    
#     # Frame metadata
#     symbols_per_mode: int  # Equal symbols per mode (enforced)
#     n_modes: int
#     spatial_modes: list  # List of (p, l) tuples
#     pilot_positions: np.ndarray  # Global pilot positions in frame
    
#     # Grid information (for spatial field generation)
#     grid_info: Optional[Dict[str, Any]] = None
    
#     # Optional: 3D multiplexed field array (n_symbols, grid_size, grid_size)
#     # This is expensive to compute, so only generated when explicitly requested
#     multiplexed_field: Optional[np.ndarray] = None
    
#     # Additional metadata
#     metadata: Optional[Dict[str, Any]] = None
    
#     def __post_init__(self):
#         """Validate frame structure after initialization."""
#         # Verify equal symbols per mode
#         symbol_lengths = [sig['n_symbols'] for sig in self.tx_signals.values()]
#         if len(set(symbol_lengths)) > 1:
#             raise ValueError(f"Frame validation failed: Unequal symbol lengths {set(symbol_lengths)}")
#         if symbol_lengths[0] != self.symbols_per_mode:
#             raise ValueError(f"Frame validation failed: Expected {self.symbols_per_mode}, got {symbol_lengths[0]}")
    
#     def get_total_symbols(self):
#         """Returns total symbols across all modes."""
#         return self.symbols_per_mode * self.n_modes
    
#     def get_mode_symbols(self, mode_key):
#         """Convenience method to get symbols for a specific mode."""
#         return self.tx_signals[mode_key]['symbols']
    
#     def to_dict(self):
#         """Convert to dictionary for serialization."""
#         return {
#             'tx_signals': self.tx_signals,
#             'symbols_per_mode': self.symbols_per_mode,
#             'n_modes': self.n_modes,
#             'spatial_modes': self.spatial_modes,
#             'pilot_positions': self.pilot_positions.tolist() if isinstance(self.pilot_positions, np.ndarray) else self.pilot_positions,
#             'metadata': self.metadata or {}
#         }


# # --- Parameters (moved to main) ---
# # WAVELENGTH = 1550e-9  
# # W0 = 25e-3  
# # Z_PROPAGATION = 1000
# # # OAM_MODES = [-4, -2, 2, 4]  # <-- RECTIFIED: This will be changed to SPATIAL_MODES
# # FEC_RATE = 0.8  
# # ...etc...


# # --- All classes (QPSKModulator, SimplifiedLDPC, PilotHandler) are IDENTICAL. ---
# # --- No changes are needed to them, so they are omitted for brevity. ---
# # ... (QPSKModulator class code) ...
# # ... (SimplifiedLDPC class code) ...
# # ... (PilotHandler class code) ...

# class QPSKModulator:
#     """
#     QPSK (Quadrature Phase-Shift Keying) Modulator with Gray coding.
    
#     Standard QPSK constellation with Gray coding minimizes bit errors:
#     - 00 -> (1+j)/√2   (I=+, Q=+)
#     - 01 -> (-1+j)/√2  (I=-, Q=+)
#     - 11 -> (-1-j)/√2  (I=-, Q=-)
#     - 10 -> (1-j)/√2   (I=+, Q=-)
    
#     Adjacent constellation points differ by only 1 bit (Gray code property).
    
#     References:
#     - Proakis & Salehi, "Digital Communications" (2008), Ch. 4
#     - Sklar, "Digital Communications" (2001), Ch. 4
#     """
#     def __init__(self, symbol_energy=1.0):
#         """
#         Parameters:
#         -----------
#         symbol_energy : float
#             Average symbol energy Es [Joules]. Default 1.0.
#         """
#         self.Es = symbol_energy
#         self.A = np.sqrt(symbol_energy)

#         # Gray-coded QPSK constellation map
#         # Bit order: (I_bit, Q_bit) where I_bit is MSB, Q_bit is LSB
#         self.constellation_map = {
#             (0, 0): self.A * (1 + 1j) / np.sqrt(2),   # 00 -> I=+, Q=+
#             (0, 1): self.A * (-1 + 1j) / np.sqrt(2),  # 01 -> I=-, Q=+
#             (1, 1): self.A * (-1 - 1j) / np.sqrt(2),  # 11 -> I=-, Q=-
#             (1, 0): self.A * (1 - 1j) / np.sqrt(2)    # 10 -> I=+, Q=-
#         }
#         self.constellation_points = np.array(list(self.constellation_map.values()))

#     def modulate(self, bits):
#         """
#         Maps bits to QPSK symbols using Gray coding.
        
#         Parameters:
#         -----------
#         bits : array_like
#             Input bits (will be padded with 0 if odd length)
        
#         Returns:
#         --------
#         symbols : ndarray
#             QPSK symbols (complex, length = ceil(len(bits)/2))
#         """
#         if len(bits) % 2 != 0:
#             bits = np.append(bits, 0)
#         bit_pairs = bits.reshape(-1, 2)
#         symbols = np.array([self.constellation_map[tuple(pair)] for pair in bit_pairs])
#         return symbols

#     def demodulate_hard(self, rx_symbols):
#         """
#         Hard decision demodulation: maps received symbols to bits using
#         minimum Euclidean distance to constellation points.
        
#         Parameters:
#         -----------
#         rx_symbols : array_like
#             Received QPSK symbols (complex)
        
#         Returns:
#         --------
#         bits : ndarray
#             Demodulated bits (length = 2 * len(rx_symbols))
#         """
#         bits = []
#         for symbol in rx_symbols:
#             distances = np.abs(self.constellation_points - symbol)
#             min_idx = np.argmin(distances)
#             for key, val in self.constellation_map.items():
#                 if np.isclose(val, self.constellation_points[min_idx]):
#                     bits.extend(list(key))
#                     break
#         return np.array(bits)

#     def demodulate_soft(self, rx_symbols, noise_var):
#         """
#         Soft decision demodulation: computes Log-Likelihood Ratios (LLRs)
#         for each bit using maximum a posteriori (MAP) detection.
        
#         Uses log-sum-exp trick for numerical stability.
        
#         Parameters:
#         -----------
#         rx_symbols : array_like
#             Received QPSK symbols (complex)
#         noise_var : float
#             Noise variance σ²
        
#         Returns:
#         --------
#         llrs : ndarray
#             Log-Likelihood Ratios for each bit (length = 2 * len(rx_symbols))
#             Positive LLR favors bit=0, negative LLR favors bit=1
#         """
#         llrs = []
#         for symbol in rx_symbols:
#             for bit_pos in range(2):
#                 # Find constellation points with bit=0 and bit=1 at position bit_pos
#                 points_bit0 = [s for b, s in self.constellation_map.items() if b[bit_pos] == 0]
#                 points_bit1 = [s for b, s in self.constellation_map.items() if b[bit_pos] == 1]
                
#                 # Compute log-likelihoods (negative squared distances / noise_var)
#                 metrics_0 = np.array([-np.abs(symbol - s)**2 / noise_var for s in points_bit0])
#                 metrics_1 = np.array([-np.abs(symbol - s)**2 / noise_var for s in points_bit1])
                
#                 # Log-sum-exp trick for numerical stability
#                 max_0 = np.max(metrics_0)
#                 max_1 = np.max(metrics_1)
#                 llr = (max_0 + np.log(np.sum(np.exp(metrics_0 - max_0))) -
#                        max_1 - np.log(np.sum(np.exp(metrics_1 - max_1))))
#                 llrs.append(llr)
#         return np.array(llrs)

#     def plot_constellation(self, ax=None):
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(6, 6))
#         for bits, symbol in self.constellation_map.items():
#             ax.plot(symbol.real, symbol.imag, 'ro', markersize=15)
#             ax.annotate(f'{bits[0]}{bits[1]}',
#                        xy=(symbol.real, symbol.imag),
#                        xytext=(symbol.real*1.3, symbol.imag*1.3),
#                        fontsize=14, fontweight='bold',
#                        ha='center', va='center')
#         ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
#         ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
#         ax.grid(True, alpha=0.3)
#         ax.set_xlabel('In-Phase (I)', fontsize=12)
#         ax.set_ylabel('Quadrature (Q)', fontsize=12)
#         ax.set_title('QPSK Constellation\n(Gray Coded)', fontsize=14, fontweight='bold')
#         ax.axis('equal')
#         ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
#         return ax

# class SimplifiedLDPC:
#     """
#     Simplified Low-Density Parity-Check (LDPC) code encoder/decoder.
    
#     This implements a systematic LDPC code where codewords have the form:
#     c = [u | p] where u is k information bits and p is m parity bits.
    
#     The parity check matrix has systematic form: H = [P | I]
#     where P is m×k parity part and I is m×m identity matrix.
#     Parity bits are computed as: p = P · u (mod 2)
    
#     Note: This is a simplified implementation for simulation purposes.
#     Real LDPC codes use more sophisticated matrix construction methods
#     and advanced decoding algorithms (e.g., belief propagation).
    
#     References:
#     - Gallager, "Low-Density Parity-Check Codes" (1962)
#     - Richardson & Urbanke, "Modern Coding Theory" (2008)
#     """
#     def __init__(self, n=1024, rate=0.8):
#         """
#         Parameters:
#         -----------
#         n : int
#             Codeword length (total bits per block)
#         rate : float
#             Code rate R = k/n (information bits / codeword length)
#         """
#         if n <= 0 or rate <= 0 or rate >= 1:
#             raise ValueError(f"Invalid LDPC parameters: n={n}, rate={rate}")
        
#         self.n = n
#         self.k = int(n * rate)
#         self.m = n - self.k
#         self.rate = rate
        
#         # Generate parity check matrix H = [P | I] in systematic form
#         self.H, self.H_info_T, self.H_parity_inv = self._generate_systematic_H()

#     def _generate_systematic_H(self):
#         """
#         Generates a systematic parity check matrix H = [P | I]
#         where P is the parity part (m×k) and I is identity (m×m).
        
#         This uses a simple random construction with column weight 3.
#         For real systems, use structured construction (Gallager, PEG, etc.)
#         """
#         col_weight = 3
        
#         # Ensure column weight doesn't exceed number of check nodes
#         if col_weight > self.m:
#             col_weight = self.m
        
#         # Generate parity part P (m×k matrix)
#         # Each column has exactly 'col_weight' ones randomly placed
#         P = np.zeros((self.m, self.k), dtype=int)
#         for col in range(self.k):
#             check_nodes = np.random.choice(self.m, col_weight, replace=False)
#             P[check_nodes, col] = 1
        
#         # Systematic form: H = [P | I] where I is m×m identity
#         H_parity = np.eye(self.m, dtype=int)
#         H = np.concatenate([P, H_parity], axis=1).astype(int)
        
#         return H, P.T, None 

#     def encode(self, info_bits):
#         """
#         Encodes information bits using systematic LDPC encoding.
        
#         For systematic encoding: c = [u | p] where
#         - u: k information bits
#         - p: m parity bits computed as p = P · u (mod 2)
        
#         Input bits are padded to multiple of k and encoded block-by-block.
        
#         Parameters:
#         -----------
#         info_bits : array_like
#             Information bits to encode (can be any length)
        
#         Returns:
#         --------
#         coded_bits : ndarray
#             Encoded codeword bits (length = ceil(len(info_bits)/k) * n)
#         """
#         if len(info_bits) == 0:
#             return np.array([], dtype=int)
        
#         # Pad input to multiple of k bits
#         num_blocks = int(np.ceil(len(info_bits) / self.k))
#         padded_info_len = num_blocks * self.k
#         pad_len = padded_info_len - len(info_bits)
        
#         if pad_len > 0:
#             padded_info_bits = np.concatenate([info_bits, np.zeros(pad_len, dtype=int)])
#         else:
#             padded_info_bits = info_bits

#         # Encode each block
#         coded_bits = []
#         for i in range(num_blocks):
#             block = padded_info_bits[i*self.k:(i+1)*self.k]
#             parity = self._compute_parity(block)
#             # Systematic codeword: [information bits | parity bits]
#             codeword = np.concatenate([block, parity])
#             coded_bits.append(codeword)
            
#         return np.concatenate(coded_bits).astype(int)

#     def _compute_parity(self, info_bits):
#         """
#         Computes parity bits for systematic LDPC encoding.
        
#         For H = [P | I], the parity bits are: p = P · u (mod 2)
#         where P is the parity part (first k columns of H).
        
#         Parameters:
#         -----------
#         info_bits : array_like
#             k information bits
        
#         Returns:
#         --------
#         parity : ndarray
#             m parity bits
#         """
#         # Extract parity part P from H = [P | I]
#         P = self.H[:, :self.k]
#         # Compute parity: p = P · u (mod 2)
#         parity = np.dot(P, info_bits) % 2
#         return parity

#     def decode_simple(self, received_bits):
#         num_blocks = len(received_bits) // self.n
#         decoded = []
#         for i in range(num_blocks):
#             block = received_bits[i*self.n:(i+1)*self.n]
#             info_bits = block[:self.k]
#             syndrome = np.dot(self.H, block) % 2
#             if np.sum(syndrome) == 0:
#                 decoded.append(info_bits)
#             else:
#                 corrected = self._bit_flipping(block, max_iter=10)
#                 decoded.append(corrected[:self.k])
#         return np.concatenate(decoded).astype(int)

#     def _bit_flipping(self, received, max_iter):
#         r = received.copy()
#         for iteration in range(max_iter):
#             syndrome = np.dot(self.H, r) % 2
#             if np.sum(syndrome) == 0:
#                 break
#             unsatisfied_counts = np.dot(self.H.T, syndrome)
#             flip_idx = np.argmax(unsatisfied_counts)
#             r[flip_idx] ^= 1
#         return r

# class PilotHandler:
#     """
#     Pilot symbol insertion and extraction for channel estimation.
    
#     Pilots are known symbols inserted periodically in the data stream
#     to enable channel estimation at the receiver. The pilot ratio
#     defines the fraction of total symbols that are pilots.
    
#     References:
#     - Proakis & Salehi (2008), Ch. 10 (Channel Estimation)
#     - Goldsmith, "Wireless Communications" (2005), Ch. 13
#     """
#     def __init__(self, pilot_ratio=0.1, pattern='uniform'):
#         """
#         Parameters:
#         -----------
#         pilot_ratio : float
#             Ratio of pilots to total symbols: n_pilots / (n_pilots + n_data)
#             Must be in range (0, 1)
#         pattern : str
#             Pilot insertion pattern: 'uniform', 'block', or 'random'
#         """
#         if pilot_ratio <= 0 or pilot_ratio >= 1:
#             raise ValueError(f"pilot_ratio must be in (0, 1), got {pilot_ratio}")
        
#         self.pilot_ratio = pilot_ratio
#         self.pattern = pattern
#         self.pilot_sequence = None
#         self.pilot_positions = None
        
#         # QPSK constellation for pilot symbols
#         # Note: This matches the QPSKModulator constellation
#         np.random.seed(123)
#         self.qpsk_constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)


#     def insert_pilots(self, data_symbols):
#         n_data = len(data_symbols)
#         n_pilots = int(np.ceil(n_data * self.pilot_ratio / (1 - self.pilot_ratio)))
#         n_total = n_data + n_pilots
        
#         pilot_indices = np.random.randint(0, 4, n_pilots)
#         self.pilot_sequence = self.qpsk_constellation[pilot_indices]

#         if self.pattern == 'uniform':
#             # Ensure unique positions
#             indices = np.linspace(0, n_total - 1, n_pilots)
#             self.pilot_positions = np.unique(indices.astype(int))
#             # Handle potential duplicates from rounding
#             n_pilots = len(self.pilot_positions)
#             self.pilot_sequence = self.pilot_sequence[:n_pilots]
#             n_total = n_data + n_pilots # Recalculate total
            
#         elif self.pattern == 'block':
#             self.pilot_positions = np.arange(n_pilots)
#         else: # Random
#             self.pilot_positions = np.sort(np.random.choice(n_total, n_pilots, replace=False))

#         frame = np.zeros(n_total, dtype=complex)
#         data_idx = 0
#         pilot_idx = 0
        
#         # More robust insertion loop
#         pilot_set = set(self.pilot_positions)
#         for i in range(n_total):
#             if i in pilot_set:
#                 if pilot_idx < n_pilots: 
#                     frame[i] = self.pilot_sequence[pilot_idx]
#                     pilot_idx += 1
#             else:
#                 if data_idx < n_data: 
#                     frame[i] = data_symbols[data_idx]
#                     data_idx += 1
#                 # Handle case where n_data was less than available slots
#                 # (e.g., due to pilot position rounding)
#                 # In this case, frame[i] remains 0 (padding)
        
#         # Trim any excess if data ran out
#         if data_idx < n_data:
#             print(f"Warning: Not all data symbols were inserted. {n_data - data_idx} remaining.")
        
#         return frame, self.pilot_positions

#     def extract_pilots(self, received_frame):
#         rx_pilots = received_frame[self.pilot_positions]
#         data_mask = np.ones(len(received_frame), dtype=bool)
#         data_mask[self.pilot_positions] = False
#         data_symbols = received_frame[data_mask]
#         return data_symbols, rx_pilots

#     def estimate_channel(self, rx_pilots, method='LS'):
#         """
#         Estimates channel gain using received pilot symbols.
        
#         Uses Least Squares (LS) estimation: h_est = mean(rx_pilots / tx_pilots)
        
#         Parameters:
#         -----------
#         rx_pilots : array_like
#             Received pilot symbols (complex)
#         method : str
#             Estimation method ('LS' for Least Squares, currently only LS supported)
        
#         Returns:
#         --------
#         h_est : complex
#             Estimated channel gain
#         """
#         if len(rx_pilots) == 0:
#             return 1.0  # No pilots, assume ideal channel
        
#         if self.pilot_sequence is None or len(self.pilot_sequence) == 0:
#             return 1.0  # No pilot sequence available
        
#         # Get transmitted pilots (trim to match received length)
#         tx_pilots = self.pilot_sequence[:len(rx_pilots)]
        
#         # Avoid division by zero
#         with np.errstate(divide='ignore', invalid='ignore'):
#             ratios = np.divide(rx_pilots, tx_pilots, 
#                              out=np.zeros_like(rx_pilots, dtype=complex),
#                              where=tx_pilots != 0)
        
#         # Least Squares estimation: average of ratios
#         # For AWGN channel: h_est = E[rx_pilots / tx_pilots]
#         h_est = np.mean(ratios[tx_pilots != 0]) if np.any(tx_pilots != 0) else 1.0
        
#         return h_est

# class encodingRunner:
#     def __init__(self,
#                  spatial_modes=None, # <-- RECTIFIED: Renamed from oam_modes
#                  wavelength=1550e-9,
#                  w0=25e-3,
#                  fec_rate=0.8,
#                  pilot_ratio=0.1,
#                  # NEW: Transmitter parameters
#                  P_tx_watts=1.0,
#                  laser_linewidth_kHz=None,
#                  timing_jitter_ps=None,
#                  tx_aperture_radius=None,
#                  beam_tilt_x_rad=0.0,
#                  beam_tilt_y_rad=0.0):
#         """
#         Initialize FSO-MDM transmitter with encoding and beam generation.
        
#         NEW TRANSMITTER PARAMETERS (all optional):
#         - P_tx_watts: Transmit power [W]. Default 1.0 W.
#         - laser_linewidth_kHz: Laser linewidth [kHz] for phase noise. Default None (no noise).
#         - timing_jitter_ps: RMS timing jitter [ps]. Default None (no jitter).
#         - tx_aperture_radius: Transmitter aperture radius [m]. Default None (no clipping).
#         - beam_tilt_x_rad: Beam tilt in x-direction [rad]. Default 0.0.
#         - beam_tilt_y_rad: Beam tilt in y-direction [rad]. Default 0.0.
#         """
#         if spatial_modes is None:
#             self.spatial_modes = [(0, -1), (0, 1)] # <-- RECTIFIED: Default is now (p,l) tuples
#         else:
#             self.spatial_modes = spatial_modes
            
#         self.n_modes = len(self.spatial_modes)
#         self.wavelength = wavelength
#         self.w0 = w0
#         self.qpsk = QPSKModulator(symbol_energy=1.0)
#         self.ldpc = SimplifiedLDPC(n=1024, rate=fec_rate)
#         self.pilot_handler = PilotHandler(pilot_ratio=pilot_ratio, pattern='uniform')
        
#         # NEW: Store transmitter parameters
#         self.P_tx_watts = P_tx_watts
#         self.laser_linewidth_kHz = laser_linewidth_kHz
#         self.timing_jitter_ps = timing_jitter_ps
#         self.tx_aperture_radius = tx_aperture_radius
#         self.beam_tilt_x_rad = beam_tilt_x_rad
#         self.beam_tilt_y_rad = beam_tilt_y_rad
        
#         self.lg_beams = {}
#         for p, l in self.spatial_modes: # <-- RECTIFIED: Iterate over (p, l) tuples
#             # Use (p, l) tuple as the dictionary key
#             self.lg_beams[(p, l)] = LaguerreGaussianBeam(p=p, l=l, wavelength=wavelength, w0=w0) # <-- RECTIFIED: Pass p
            
#         print(f"Spatial Modes (p, l): {self.spatial_modes}") # <-- RECTIFIED
#         print(f"Number of Modes: {self.n_modes}")
#         print(f"Wavelength: {wavelength*1e9:.0f} [nm]")
#         print(f"Beam Waist w0: {w0*1e3:.2f} [mm]")
#         print(f"LDPC Code Rate: {fec_rate}")
#         print(f"Pilot Ratio: {pilot_ratio:.1%}")
        
#         # NEW: Print transmitter configuration
#         print(f"Transmitter Power: {P_tx_watts:.2f} W ({10*np.log10(P_tx_watts*1000):.1f} dBm)")
#         if laser_linewidth_kHz is not None:
#             print(f"Laser Linewidth: {laser_linewidth_kHz:.1f} kHz")
#         if timing_jitter_ps is not None:
#             print(f"Timing Jitter: {timing_jitter_ps:.1f} ps RMS")
#         if tx_aperture_radius is not None:
#             print(f"TX Aperture: {tx_aperture_radius*1e3:.1f} mm radius")
#         if beam_tilt_x_rad != 0.0 or beam_tilt_y_rad != 0.0:
#             print(f"Beam Tilt: ({np.degrees(beam_tilt_x_rad):.3f}, {np.degrees(beam_tilt_y_rad):.3f}) deg")
    
#     def generate_phase_noise_for_symbols(self, num_symbols, symbol_time_s, seed=None):
#         """
#         Generates phase noise sequences for all modes for symbol-by-symbol transmission.
        
#         Useful when simulating realistic transmission where each symbol experiences
#         cumulative phase noise from laser linewidth.
        
#         Parameters:
#         -----------
#         num_symbols : int
#             Number of symbols (should match frame length)
#         symbol_time_s : float
#             Symbol period [s]
#         seed : int, optional
#             Random seed for reproducibility
        
#         Returns:
#         --------
#         phase_noise_sequences : dict
#             Dictionary mapping mode_key -> phase_noise_array (rad)
#         """
#         if self.laser_linewidth_kHz is None or self.laser_linewidth_kHz <= 0:
#             # Return zeros for all modes
#             return {mode_key: np.zeros(num_symbols) for mode_key in self.lg_beams.keys()}
        
#         phase_noise_sequences = {}
#         for mode_key, beam in self.lg_beams.items():
#             phase_noise = beam.generate_phase_noise_sequence(
#                 num_symbols, symbol_time_s, self.laser_linewidth_kHz, seed=seed
#             )
#             phase_noise_sequences[mode_key] = phase_noise
        
#         return phase_noise_sequences
    
#     def get_tx_summary(self):
#         """
#         Returns a summary dictionary of transmitter parameters and effects.
        
#         Returns:
#         --------
#         summary : dict
#             Dictionary with transmitter configuration and calculated effects
#         """
#         # Use first beam for calculations (all beams share same transmitter)
#         first_beam = list(self.lg_beams.values())[0]
#         return first_beam.get_tx_parameters_summary(
#             P_tx_watts=self.P_tx_watts,
#             laser_linewidth_kHz=self.laser_linewidth_kHz,
#             timing_jitter_ps=self.timing_jitter_ps,
#             tx_aperture_radius=self.tx_aperture_radius,
#             beam_tilt_x_rad=self.beam_tilt_x_rad,
#             beam_tilt_y_rad=self.beam_tilt_y_rad
#         )
    
#     def validate_transmitter(self, tx_signals=None, frame=None, verbose=True):
#         """
#         PHASE 3: System validation method.
        
#         Validates:
#         - Equal symbols per mode
#         - Mode orthogonality (inner products)
#         - Frame structure integrity
        
#         Parameters:
#         -----------
#         tx_signals : dict, optional
#             Transmitted signals (backward compat)
#         frame : FSO_MDM_Frame, optional
#             Frame object
#         verbose : bool
#             Print validation results
        
#         Returns:
#         --------
#         validation_results : dict
#             Dictionary with validation metrics
#         """
#         if frame is not None:
#             tx_signals = frame.tx_signals
#             symbols_per_mode = frame.symbols_per_mode
#         elif tx_signals is not None:
#             symbols_per_mode = list(tx_signals.values())[0]['n_symbols']
#         else:
#             raise ValueError("Must provide either 'tx_signals' or 'frame'")
        
#         results = {
#             'equal_symbols': True,
#             'mode_orthogonality': {},
#             'frame_valid': True,
#             'errors': []
#         }
        
#         # Check equal symbols
#         symbol_lengths = [sig['n_symbols'] for sig in tx_signals.values()]
#         if len(set(symbol_lengths)) > 1:
#             results['equal_symbols'] = False
#             results['errors'].append(f"Unequal symbol lengths: {set(symbol_lengths)}")
        
#         # Check mode orthogonality (compute inner products at z=0)
#         # Use small grid for fast computation
#         test_grid_size = 128
#         test_extent = 3 * self.w0 * 1e-3  # 3x beam waist
#         x_test = np.linspace(-test_extent, test_extent, test_grid_size)
#         y_test = np.linspace(-test_extent, test_extent, test_grid_size)
#         X_test, Y_test = np.meshgrid(x_test, y_test)
#         R_test = np.sqrt(X_test**2 + Y_test**2)
#         PHI_test = np.arctan2(Y_test, X_test)
#         dA = (2 * test_extent / test_grid_size)**2
        
#         mode_fields = {}
#         for mode_key, sig_data in tx_signals.items():
#             beam = sig_data['beam']
#             mode_fields[mode_key] = beam.generate_beam_field(
#                 R_test, PHI_test, 0,
#                 P_tx_watts=self.P_tx_watts,
#                 tx_aperture_radius=self.tx_aperture_radius
#             )
        
#         # Compute inner products between modes
#         mode_keys_list = list(mode_fields.keys())
#         for i, mode_i in enumerate(mode_keys_list):
#             for j, mode_j in enumerate(mode_keys_list):
#                 if i <= j:  # Only compute upper triangle
#                     field_i = mode_fields[mode_i]
#                     field_j = mode_fields[mode_j]
#                     inner_product = np.sum(field_i * np.conj(field_j)) * dA
#                     results['mode_orthogonality'][(mode_i, mode_j)] = inner_product
        
#         # Check orthogonality (should be ~0 for i != j, ~1 for i == j)
#         orthogonality_errors = []
#         for (mode_i, mode_j), ip in results['mode_orthogonality'].items():
#             if mode_i == mode_j:
#                 # Should be ~1 (normalized power)
#                 if abs(ip - 1.0) > 0.1:
#                     orthogonality_errors.append(f"Mode {mode_i} normalization: {ip:.4f} (expected ~1.0)")
#             else:
#                 # Should be ~0 (orthogonal)
#                 if abs(ip) > 0.1:
#                     orthogonality_errors.append(f"Modes {mode_i}, {mode_j} not orthogonal: {ip:.4f}")
        
#         if orthogonality_errors:
#             results['errors'].extend(orthogonality_errors)
#             results['frame_valid'] = False
        
#         if verbose:
#             print("\n" + "="*70)
#             print("TRANSMITTER VALIDATION")
#             print("="*70)
#             print(f"Equal Symbols Per Mode: {'✓' if results['equal_symbols'] else '✗'}")
#             print(f"Frame Structure Valid: {'✓' if results['frame_valid'] else '✗'}")
#             if results['errors']:
#                 print("\nErrors Found:")
#                 for error in results['errors']:
#                     print(f"  - {error}")
#             else:
#                 print("\n✓ All validation checks passed!")
#             print("="*70)
        
#         return results

#     def transmit(self, data_bits, verbose=True):
#         if verbose: print(f"\nInput: {len(data_bits)} [info bits]")
        
#         # CORRECTED LOGGING
#         encoded_bits = self.ldpc.encode(data_bits)
        
#         # Calculate the ACTUAL code rate after padding
#         padded_info_len = int(np.ceil(len(data_bits) / self.ldpc.k)) * self.ldpc.k
#         actual_rate = len(data_bits) / len(encoded_bits) if len(encoded_bits) > 0 else 0
        
#         if verbose: 
#             print(f"After LDPC (target rate {self.ldpc.rate:.2f}, actual {actual_rate:.2f}): {len(encoded_bits)} [coded bits]")
#             if padded_info_len != len(data_bits):
#                 print(f"  (Input was padded from {len(data_bits)} to {padded_info_len} info bits to fit block structure)")

#         qpsk_symbols = self.qpsk.modulate(encoded_bits)
#         if verbose: print(f"After QPSK: {len(qpsk_symbols)} [symbols]")
        
#         # Check if there are enough symbols for pilots
#         if len(qpsk_symbols) == 0:
#             print("Warning: No QPSK symbols generated. Aborting transmit.")
#             return {}
            
#         frame_with_pilots, pilot_pos = self.pilot_handler.insert_pilots(qpsk_symbols)
#         if verbose: print(f"After Pilot Insertion: {len(frame_with_pilots)} [symbols] ({len(pilot_pos)} pilots)")
        
#         # PHASE 1 FIX: Enforce equal symbol allocation per mode
#         # Ensure frame length is multiple of n_modes for perfect alignment
#         total_symbols = len(frame_with_pilots)
#         frame_len = (total_symbols // self.n_modes) * self.n_modes
        
#         if frame_len < total_symbols:
#             if verbose:
#                 print(f"  Truncating frame from {total_symbols} to {frame_len} symbols "
#                       f"(ensuring equal {frame_len // self.n_modes} symbols per mode)")
#             frame_with_pilots = frame_with_pilots[:frame_len]
#             # Update pilot positions to be within new frame length
#             pilot_pos = pilot_pos[pilot_pos < frame_len]
        
#         symbols_per_mode = frame_len // self.n_modes
        
#         # Distribute symbols equally across modes
#         tx_signals = {}
#         start_idx = 0
        
#         for idx, (p, l) in enumerate(self.spatial_modes):
#             end_idx = start_idx + symbols_per_mode
#             mode_symbols = frame_with_pilots[start_idx:end_idx]
            
#             mode_key = (p, l)
            
#             tx_signals[mode_key] = {
#                 'symbols': mode_symbols,
#                 'frame': mode_symbols, 
#                 'beam': self.lg_beams[mode_key],
#                 'n_symbols': len(mode_symbols)
#             }
#             if verbose: print(f"  Mode (p={p}, l={l:+2d}): {len(mode_symbols)} symbols")
#             start_idx = end_idx
        
#         # PHASE 1 FIX: Create unified frame interface
#         frame = FSO_MDM_Frame(
#             tx_signals=tx_signals,
#             symbols_per_mode=symbols_per_mode,
#             n_modes=self.n_modes,
#             spatial_modes=self.spatial_modes,
#             pilot_positions=pilot_pos,
#             grid_info=None,  # Will be set by generate_spatial_field if needed
#             metadata={
#                 'n_info_bits': len(data_bits),
#                 'n_encoded_bits': len(encoded_bits),
#                 'n_qpsk_symbols': len(qpsk_symbols),
#                 'pilot_ratio': self.pilot_handler.pilot_ratio,
#                 'ldpc_rate': self.ldpc.rate
#             }
#         )
        
#         # Return both frame object (new) and tx_signals dict (backward compatibility)
#         # Receivers expecting dict format will still work
#         if hasattr(self, '_return_frame_object') and self._return_frame_object:
#             return frame
#         else:
#             # Default: return dict for backward compatibility, but store frame
#             self._last_frame = frame
#             return tx_signals

#     def generate_spatial_field(self, tx_signals=None, frame=None, z=500, grid_size=256, extent_mm=None, 
#                                generate_multiplexed_sequence=False):
#         """
#         Generates spatial field(s) from transmitted signals.
        
#         PHASE 1 ENHANCEMENT: Can now accept either tx_signals dict (backward compat)
#         or FSO_MDM_Frame object. Optionally generates full 3D multiplexed sequence.
        
#         Parameters:
#         -----------
#         tx_signals : dict, optional
#             Transmitted signals per mode (backward compatibility)
#         frame : FSO_MDM_Frame, optional
#             Unified frame object (preferred)
#         z : float
#             Propagation distance [m]
#         grid_size : int
#             Grid resolution (N×N)
#         extent_mm : float, optional
#             Spatial extent [mm]. Auto-computed if None.
#         generate_multiplexed_sequence : bool
#             If True, generates full 3D array (n_symbols, grid_size, grid_size).
#             WARNING: Expensive for large n_symbols/grid_size.
        
#         Returns:
#         --------
#         If generate_multiplexed_sequence=False:
#             total_intensity, grid_info (2D snapshot at t=0)
#         If generate_multiplexed_sequence=True:
#             multiplexed_field_3d, grid_info (3D array for full sequence)
#         """
#         # Handle input: prefer frame object, fall back to tx_signals
#         if frame is not None:
#             tx_signals = frame.tx_signals
#             symbols_per_mode = frame.symbols_per_mode
#         elif tx_signals is not None:
#             # Extract from dict
#             symbols_per_mode = list(tx_signals.values())[0]['n_symbols']
#         else:
#             raise ValueError("Must provide either 'tx_signals' or 'frame'")
        
#         if extent_mm is None:
#             max_beam_radius = 0
#             for mode_key, sig_data in tx_signals.items():
#                 beam = sig_data['beam']
#                 w_z = beam.beam_waist(z)
#                 max_beam_radius = max(max_beam_radius, w_z)
#             extent_mm = 3 * max_beam_radius * 1e3
        
#         extent = extent_mm * 1e-3
#         x = np.linspace(-extent, extent, grid_size)
#         y = np.linspace(-extent, extent, grid_size)
#         X, Y = np.meshgrid(x, y)
#         R = np.sqrt(X**2 + Y**2)
#         PHI = np.arctan2(Y, X)

#         grid_info = {'x': x, 'y': y, 'X': X, 'Y': Y, 'extent_mm': extent_mm, 'grid_size': grid_size}
        
#         # PHASE 1: Optionally generate full 3D multiplexed sequence
#         if generate_multiplexed_sequence:
#             print(f"  Generating 3D multiplexed field sequence ({symbols_per_mode} symbols × {grid_size}×{grid_size} grid)...")
            
#             # Pre-compute beam fields for all modes (same for all symbols)
#             mode_fields = {}
#             for mode_key, sig_data in tx_signals.items():
#                 beam = sig_data['beam']
#                 mode_fields[mode_key] = beam.generate_beam_field(
#                     R, PHI, z,
#                     P_tx_watts=self.P_tx_watts,
#                     laser_linewidth_kHz=self.laser_linewidth_kHz,
#                     timing_jitter_ps=self.timing_jitter_ps,
#                     tx_aperture_radius=self.tx_aperture_radius,
#                     beam_tilt_x_rad=self.beam_tilt_x_rad,
#                     beam_tilt_y_rad=self.beam_tilt_y_rad
#                 )
            
#             # Generate 3D array: (n_symbols, grid_size, grid_size)
#             multiplexed_field_3d = np.zeros((symbols_per_mode, grid_size, grid_size), dtype=complex)
            
#             for t in range(symbols_per_mode):
#                 total_field_t = np.zeros((grid_size, grid_size), dtype=complex)
#                 for mode_key, sig_data in tx_signals.items():
#                     if sig_data['n_symbols'] > t:
#                         symbol_t = sig_data['symbols'][t]
#                         field = mode_fields[mode_key]
#                         total_field_t += field * symbol_t
#                 multiplexed_field_3d[t] = total_field_t
            
#             # Update frame if provided
#             if frame is not None:
#                 frame.multiplexed_field = multiplexed_field_3d
#                 frame.grid_info = grid_info
            
#             return multiplexed_field_3d, grid_info
#         else:
#             # Default: 2D snapshot at t=0 (backward compatible)
#             total_field = np.zeros((grid_size, grid_size), dtype=complex) 
#             for mode_key, sig_data in tx_signals.items():
#                 if sig_data['n_symbols'] > 0: 
#                     beam = sig_data['beam']
#                     symbol = sig_data['symbols'][0]  # Snapshot at t=0
#                     field = beam.generate_beam_field(
#                         R, PHI, z,
#                         P_tx_watts=self.P_tx_watts,
#                         laser_linewidth_kHz=self.laser_linewidth_kHz,
#                         timing_jitter_ps=self.timing_jitter_ps,
#                         tx_aperture_radius=self.tx_aperture_radius,
#                         beam_tilt_x_rad=self.beam_tilt_x_rad,
#                         beam_tilt_y_rad=self.beam_tilt_y_rad
#                     )
#                     modulated_field = field * symbol
#                     total_field += modulated_field 
            
#             total_intensity = np.abs(total_field)**2
#             return total_intensity, grid_info

#     def plot_system_summary(self, data_bits, tx_signals):
#         n_modes = len(self.spatial_modes) # <-- RECTIFIED
#         if n_modes == 0:
#             print("No modes to plot.")
#             return None
            
#         modes_per_row = min(4, n_modes)
#         n_mode_rows = int(np.ceil(n_modes / modes_per_row))
#         total_rows = 1 + n_mode_rows
#         fig = plt.figure(figsize=(16, 5 * total_rows))
#         fig.suptitle("FSO-MDM Transmitter System Summary", fontsize=16, fontweight='bold') # <-- RECTIFIED: MDM

#         ax1 = plt.subplot(total_rows, 3, 1)
#         self.qpsk.plot_constellation(ax=ax1)
#         ax2 = plt.subplot(total_rows, 3, 2)
        
#         first_mode = self.spatial_modes[0] # <-- RECTIFIED
#         if tx_signals[first_mode]['n_symbols'] > 0:
#             symbols = tx_signals[first_mode]['symbols'][:100]
#             ax2.plot(symbols.real, symbols.imag, 'b.', alpha=0.6, markersize=8)
#             ax2.plot(self.qpsk.constellation_points.real,
#                     self.qpsk.constellation_points.imag,
#                     'ro', markersize=12, label='Ideal')
#             ax2.grid(True, alpha=0.3); ax2.set_xlabel('In-Phase'); ax2.set_ylabel('Quadrature')
#             ax2.set_title(f'Tx Symbols - Mode (p={first_mode[0]}, l={first_mode[1]})'); ax2.legend(); ax2.axis('equal') # <-- RECTIFIED
#             ax3 = plt.subplot(total_rows, 3, 3)
#             ax3.plot(np.abs(symbols), 'b-', linewidth=1.5, label='|s(n)|')
#             ax3.plot(np.angle(symbols), 'r-', linewidth=1.5, alpha=0.7, label='∠s(n)')
#             ax3.grid(True, alpha=0.3); ax3.set_xlabel('Symbol Index'); ax3.set_ylabel('Amplitude / Phase')
#             ax3.set_title('Symbol Sequence'); ax3.legend()

#         for idx, (p, l) in enumerate(self.spatial_modes): # <-- RECTIFIED
#             mode_key = (p, l)
#             mode_row = idx // modes_per_row
#             mode_col = idx % modes_per_row
#             ax = plt.subplot2grid((total_rows, modes_per_row), (1 + mode_row, mode_col), fig=fig)
            
#             beam = self.lg_beams[mode_key]
#             extent_mm = 3 * self.w0 * 1e3
#             grid_size = 200
#             extent = extent_mm * 1e-3
#             x = np.linspace(-extent, extent, grid_size)
#             y = np.linspace(-extent, extent, grid_size)
#             X, Y = np.meshgrid(x, y)
#             R = np.sqrt(X**2 + Y**2)
#             PHI = np.arctan2(Y, X)
            
#             # NEW: Pass transmitter parameters to calculate_intensity
#             intensity = beam.calculate_intensity(
#                 R, PHI, 0,
#                 P_tx_watts=self.P_tx_watts,
#                 laser_linewidth_kHz=self.laser_linewidth_kHz,
#                 timing_jitter_ps=self.timing_jitter_ps,
#                 tx_aperture_radius=self.tx_aperture_radius,
#                 beam_tilt_x_rad=self.beam_tilt_x_rad,
#                 beam_tilt_y_rad=self.beam_tilt_y_rad
#             )
#             im = ax.imshow(intensity, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm], cmap='hot', origin='lower')
#             ax.set_xlabel('x [mm]', fontsize=8); ax.set_ylabel('y [mm]', fontsize=8)
#             # <-- RECTIFIED: Correctly uses beam.p and beam.l
#             ax.set_title(f'LG$_{{{beam.p}}}^{{{beam.l}}}$ M²={beam.M_squared:.1f}', fontweight='bold', fontsize=9)
#             plt.colorbar(im, ax=ax, fraction=0.046, label='I [a.u.]') 

#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         return fig

# if __name__ == "__main__":
#     # --- Define Simulation Parameters ---
#     WAVELENGTH = 1550e-9  
#     W0 = 25e-3  
#     Z_PROPAGATION = 1000
    
#     # <-- RECTIFIED: Changed from OAM_MODES to SPATIAL_MODES (p, l)
#     # You can now add complex modes like (1, 1)
#     SPATIAL_MODES = [(0, -4), (0, -2), (0, 2), (0, 4), (1, 1), (1, -1)] 
    
#     FEC_RATE = 0.8  
#     PILOT_RATIO = 0.1  
#     QPSK_ENERGY = 1.0  
#     N_INFO_BITS = 4096  
#     GRID_SIZE = 512
#     DPI = 300 # <-- Reduced DPI for faster testing
    
#     PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
    
#     # <-- RECTIFIED: FIG_NAME is no longer used globally
#     # FIG_NAME = f"lg_p{0}_l{'_'.join(map(str, OAM_MODES))}_beam.png" 

#     print(f"Link distance:      {Z_PROPAGATION}m ({Z_PROPAGATION/1000:.1f}km)")
#     print(f"Wavelength:         {WAVELENGTH*1e9:.0f}nm")
#     print(f"Beam waist:         {W0*1e3:.1f}mm")
#     print(f"Spatial modes (p,l):{SPATIAL_MODES}") # <-- RECTIFIED
#     print(f"FEC rate:           {FEC_RATE}")
#     print(f"Info bits:          {N_INFO_BITS}")

#     os.makedirs(PLOT_DIR, exist_ok=True)
#     print(f"Plot directory ensured at: {PLOT_DIR}\n") 
    
#     system = encodingRunner(
#         spatial_modes=SPATIAL_MODES, # <-- RECTIFIED
#         wavelength=WAVELENGTH,
#         w0=W0,
#         fec_rate=FEC_RATE,
#         pilot_ratio=PILOT_RATIO
#     )
    
#     data_bits = np.random.randint(0, 2, N_INFO_BITS)
#     print(f"Generated {N_INFO_BITS} random data bits")

#     tx_signals = system.transmit(data_bits, verbose=True)

#     print("\nGenerating system summary plot...")
#     fig = system.plot_system_summary(data_bits, tx_signals)
#     if fig:
#         summary_path = os.path.join(PLOT_DIR, 'encoding_summary_generalized.png') # <-- RECTIFIED
#         fig.savefig(summary_path, dpi=DPI, bbox_inches='tight')

#     print(f"Generating multiplexed field at z={Z_PROPAGATION}m...")
#     total_field, grid_info = system.generate_spatial_field(
#         tx_signals, 
#         z=Z_PROPAGATION,
#         grid_size=GRID_SIZE
#     )

#     fig2, (ax_trans, ax_long) = plt.subplots(1, 2, figsize=(16, 7))
#     fig2.suptitle("MDM Multiplexed Beam Propagation", fontsize=16, fontweight='bold') # <-- RECTIFIED
    
#     extent_mm = grid_info['extent_mm']
#     im1 = ax_trans.imshow(total_field, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
#                   cmap='hot', origin='lower', interpolation='bilinear')
#     ax_trans.set_xlabel('x [mm]', fontsize=12); ax_trans.set_ylabel('y [mm]', fontsize=12)
#     ax_trans.set_title(f'Transverse Field at z={Z_PROPAGATION/1000:.1f}km\n(Coherent Sum Snapshot)',
#                 fontsize=14, fontweight='bold')
#     ax_trans.set_aspect('equal')
#     plt.colorbar(im1, ax=ax_trans, label='Intensity [a.u.]')
    
#     print("Calculating longitudinal propagation (this may take a moment)...")
#     z_max = Z_PROPAGATION * 1.5
#     num_z = 150; num_r = 200
#     z_array = np.linspace(1e-6, z_max, num_z)
    
#     max_beam_radius = 0
#     for mode_key in system.spatial_modes: # <-- RECTIFIED
#         beam = system.lg_beams[mode_key]
#         w_z = beam.beam_waist(z_max)
#         max_beam_radius = max(max_beam_radius, w_z)
    
#     r_max = 3 * max_beam_radius
#     r_array = np.linspace(-r_max, r_max, num_r)
    
#     intensity_long = np.zeros((num_r, num_z))
    
#     for i, z in enumerate(z_array):
#         total_field_slice = np.zeros(num_r, dtype=complex) # Must be complex
#         for mode_key, sig_data in tx_signals.items(): # <-- RECTIFIED
#             if sig_data['n_symbols'] > 0:
#                 beam = sig_data['beam']
#                 symbol = sig_data['symbols'][0]
#                 # NEW: Pass transmitter parameters to generate_beam_field
#                 field_slice = beam.generate_beam_field(
#                     np.abs(r_array), 0, z,
#                     P_tx_watts=system.P_tx_watts,
#                     laser_linewidth_kHz=system.laser_linewidth_kHz,
#                     timing_jitter_ps=system.timing_jitter_ps,
#                     tx_aperture_radius=system.tx_aperture_radius,
#                     beam_tilt_x_rad=system.beam_tilt_x_rad,
#                     beam_tilt_y_rad=system.beam_tilt_y_rad
#                 )
#                 total_field_slice += field_slice * symbol
        
#         intensity_long[:, i] = np.abs(total_field_slice)**2
#     z_array_km = z_array / 1000
#     r_max_mm = r_max * 1e3
    
#     vmax = np.percentile(intensity_long, 99.8) 
#     im2 = ax_long.imshow(intensity_long, extent=[0, z_array_km[-1], -r_max_mm, r_max_mm],
#                        aspect='auto', cmap='hot', origin='lower', interpolation='bilinear',
#                        vmax=vmax)
    
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(system.spatial_modes))) # <-- RECTIFIED
#     for idx, mode_key in enumerate(system.spatial_modes): # <-- RECTIFIED
#         beam = system.lg_beams[mode_key]
#         w_z_array = np.array([beam.beam_waist(z)*1e3 for z in z_array])
#         ax_long.plot(z_array_km, w_z_array, '--', linewidth=1.5, 
#                     color=colors[idx], alpha=0.8, label=f'({beam.p},{beam.l}) w(z)') # <-- RECTIFIED
#         ax_long.plot(z_array_km, -w_z_array, '--', linewidth=1.5, 
#                     color=colors[idx], alpha=0.8)
    
#     ax_long.axvline(Z_PROPAGATION/1000, color='lime', linestyle=':', linewidth=2.5,
#                    label=f'z={Z_PROPAGATION/1000:.1f}km', alpha=0.8)
    
#     ax_long.set_xlabel('Propagation Distance z [km]', fontsize=12)
#     ax_long.set_ylabel('Radial Position r [mm]', fontsize=12)
#     ax_long.set_title(f'Longitudinal Propagation (Coherent Sum)\nλ={WAVELENGTH*1e9:.0f}nm, w₀={W0*1e3:.1f}mm',
#                      fontsize=14, fontweight='bold')
#     ax_long.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
#     ax_long.grid(True, alpha=0.3)
#     ax_long.set_ylim(-r_max_mm, r_max_mm) 
#     plt.colorbar(im2, ax=ax_long, label='Intensity [a.u.]', fraction=0.046, pad=0.04)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     prop_path = os.path.join(PLOT_DIR, 'mdm_multiplexed_field.png') # <-- RECTIFIED
#     fig2.savefig(prop_path, dpi=DPI, bbox_inches='tight')


#     print("\nTx Summary") # <-- RECTIFIED

#     print(f"Information bits:      {N_INFO_BITS} [bits]")
#     print(f"Code rate:             {system.ldpc.rate}")
#     # Re-encode to get an accurate count, as padding might occur
#     coded_bits_len = len(system.ldpc.encode(data_bits)) 
#     print(f"Coded bits:            {coded_bits_len} [bits]")

#     total_symbols = sum([sig['n_symbols'] for sig in tx_signals.values()])
#     n_pilots = len(system.pilot_handler.pilot_positions)
#     print(f"Total symbols:         {total_symbols} [symbols] (incl. {n_pilots} pilots)")
    
#     # <-- RECTIFIED: More descriptive print
#     print("Symbols per mode:")
#     for mode_key, sig in tx_signals.items():
#         print(f"  Mode (p={mode_key[0]}, l={mode_key[1]}): {sig['n_symbols']} symbols")

#     # This calculation remains valid, but it's the *total system* efficiency
#     # not per-mode efficiency, as it's info_bits / total_symbols_all_modes
#     total_system_efficiency = N_INFO_BITS / total_symbols 
#     print(f"Total System Spectral Efficiency: {total_system_efficiency:.3f} [bits/symbol]")

#     print(f"\n{'BEAM PARAMETERS AT z=' + str(Z_PROPAGATION) + 'm:'}")
#     for mode_key in system.spatial_modes: # <-- RECTIFIED
#         beam = system.lg_beams[mode_key]
#         theta_0, theta_eff = beam.effective_divergence_angle # <-- Access as property
#         w_z = beam.beam_waist(Z_PROPAGATION)
#         z_R = beam.z_R
#         print(f"  Mode (p={beam.p}, l={beam.l:+2d}): M²={beam.M_squared:.1f}, "
#               f"z_R={z_R:.1f}m, "
#               f"w(z)={w_z*1e3:.1f}mm, "
#               f"θ_eff={theta_eff*1e6:.1f}μrad")
#     plt.show()



