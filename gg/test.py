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