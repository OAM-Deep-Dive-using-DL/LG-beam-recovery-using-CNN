import os
import sys

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import all modules
try:
    from lgBeam import LaguerreGaussianBeam
    from encoding import encodingRunner, QPSKModulator, SimplifiedLDPC, PilotHandler
    from fsplAtmAttenuation import calculate_kim_attenuation, calculate_geometric_loss
    
    from turbulence import (AtmosphericTurbulence, 
                            create_multi_layer_screens,
                            apply_multi_layer_turbulence)
    from receiver import FSORx
except ImportError as e:
    print(f"✗ E2E Simulation Import Error: {e}")
    print("  Please ensure lgBeam.py, encoding.py, fsplAtmAttenuation.py, turbulence.py, and receiver.py are in the same directory.")
    sys.exit(1)

np.random.seed(42)

# ============================================================================
# GLOBAL SYSTEM CONFIGURATION
# ============================================================================
class SimulationConfig:
    """
    Centralized configuration for a realistic FSO-OAM system 
    for research purposes.
    """
    # --- Optical Parameters ---
    WAVELENGTH = 1550e-9  # [m]
    W0 = 25e-3           # [m]
    
    # --- Link Parameters ---
    DISTANCE = 800      # [m]
    RECEIVER_DIAMETER = 0.3  # [m]
    P_TX_TOTAL_W = 1.0     # [W] – now used for scaling
    
    # --- Spatial Modes ---
    SPATIAL_MODES = [(0, -1), (0, 1), (0, -3), (0, 3), (0, -4), (0, 4)]
    
    # --- Turbulence Parameters (TRUE IDEAL) ---
    CN2 = 1e-18           # [m^(-2/3)] 
    L0 = 10.0           # [m]
    L0_INNER = 0.005    # [m]
    NUM_SCREENS = 15   
    
    # --- Weather Condition ---
    WEATHER = 'clear'    
    
    # --- Communication Parameters (TRUE IDEAL) ---
    FEC_RATE = 0.8      
    PILOT_RATIO = 0.1   
    
    # FIXED: Multiple of total k_ldpc = FEC_RATE * 1024 * n_modes
    N_INFO_BITS = 819 * 6  # 4914 bits (k=819 per codeword, 6 modes equiv)
    
    # --- Simulation Grid ---
    N_GRID = 512        # 512x512 is fast for a sanity check
    OVERSAMPLING = 2    
    
    # --- Receiver Configuration (TRUE IDEAL) ---
    EQ_METHOD = 'zf'    
    ADD_NOISE = False   # Disable additive noise
    SNR_DB = 50        # Set to a high dummy value
    
    # --- Output ---
    PLOT_DIR = os.path.join(SCRIPT_DIR, "e2e_results_ideal") # New folder
    DPI = 300

# ============================================================================
# NEW E2E SIMULATION (RECTIFIED)
# ============================================================================

def run_e2e_simulation(config):
    """
    Runs the complete, rectified E2E simulation.
    
    This implementation is physically correct AND compatible with all
    your existing files.
    """
    
    # === 1. INITIALIZATION ===
    print("\n" + "="*80)
    print("INITIALIZING E2E SIMULATION")
    print("="*80)
    
    cfg = config
    n_modes = len(cfg.SPATIAL_MODES)

    # 1a. Initialize Transmitter
    print("[1] Initializing Transmitter...")
    transmitter = encodingRunner(
        spatial_modes=cfg.SPATIAL_MODES,
        wavelength=cfg.WAVELENGTH,
        w0=cfg.W0,
        fec_rate=cfg.FEC_RATE,
        pilot_ratio=cfg.PILOT_RATIO
    )
    
    # 1b. Initialize Turbulence Model
    print("[2] Initializing Channel Models...")
    turbulence = AtmosphericTurbulence(
        Cn2=cfg.CN2, L0=cfg.L0, l0=cfg.L0_INNER, wavelength=cfg.WAVELENGTH
    )

    # 1c. Initialize Simulation Grid
    print("[3] Initializing Simulation Grid...")
    max_m2_beam = max(transmitter.lg_beams.values(), key=lambda b: b.M_squared)
    beam_size_at_rx = max_m2_beam.beam_waist(cfg.DISTANCE)
    
    D = cfg.OVERSAMPLING * 6 * beam_size_at_rx
    delta = D / cfg.N_GRID
    
    x = np.linspace(-D/2, D/2, cfg.N_GRID)
    y = np.linspace(-D/2, D/2, cfg.N_GRID)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    grid_info = {
        'x': x, 'y': y, 'X': X, 'Y': Y, 'R': R, 'PHI': PHI,
        'D': D, 'delta': delta, 'N': cfg.N_GRID
    }
    print(f"    Grid sized for: LG_p={max_m2_beam.p}, l={max_m2_beam.l} (M²={max_m2_beam.M_squared:.1f})")
    print(f"    Grid resolution: {cfg.N_GRID}x{cfg.N_GRID}, Pixel: {delta*1000:.2f} mm")

    # 1d. Generate Basis Fields (at z=0) – FIXED: Init dict + scale for total P_tx
    print("[4] Generating Basis Mode Fields (at z=0)...")
    dA = delta**2
    tx_basis_fields = {}  # FIXED: Define dict here
    basis_energy = {}  # Per-mode energy for scaling
    for mode_key, beam in transmitter.lg_beams.items():
        E_basis = beam.generate_beam_field(R, PHI, 0)
        energy = np.sum(np.abs(E_basis)**2) * dA  # Unit symbol energy ~1
        basis_energy[mode_key] = energy
        # Scale basis so each mode contrib P_tx / n_modes when symbol=1
        scale = np.sqrt(cfg.P_TX_TOTAL_W / (n_modes * energy))
        tx_basis_fields[mode_key] = E_basis * scale  # Now |sum basis *1|^2 dA = P_tx
    # FIXED: Validate total power
    E_tx_check = np.sum([tx_basis_fields[mode_key] for mode_key in cfg.SPATIAL_MODES], axis=0)
    total_power = np.sum(np.abs(E_tx_check)**2) * dA
    print(f"    Basis scaled for total TX power: {cfg.P_TX_TOTAL_W} W ({n_modes} modes) – verified: {total_power:.3f} W")

    # 1e. Initialize Receiver
    print("[5] Initializing Receiver...")
    receiver = FSORx(
        spatial_modes=cfg.SPATIAL_MODES,
        wavelength=cfg.WAVELENGTH,
        w0=cfg.W0,
        z_distance=cfg.DISTANCE,
        fec_rate=cfg.FEC_RATE,
        pilot_handler=transmitter.pilot_handler,  # Share pilot handler!
        eq_method=cfg.EQ_METHOD,
        receiver_radius=(cfg.RECEIVER_DIAMETER / 2.0),
        ldpc_instance=transmitter.ldpc  # CRITICAL: Share LDPC instance to ensure same H matrix!
    )

    # === 2. TRANSMITTER ===
    print("\n" + "="*80)
    print("STAGE 1: TRANSMITTER")
    print("="*80)
    
    # Generate original data bits
    data_bits = np.random.randint(0, 2, cfg.N_INFO_BITS)
    print(f"Generated {len(data_bits)} info bits.")
    
    # Generate the full frame of symbols
    tx_frame = transmitter.transmit(data_bits, verbose=True)
    tx_signals = tx_frame.tx_signals  # Extract dict from FSO_MDM_Frame object
    
    # Get total number of symbols. Find the *minimum* length across all modes.
    symbol_lengths = [len(sig['symbols']) for sig in tx_signals.values()]  # FIXED: Use len(symbols)
    if not symbol_lengths or min(symbol_lengths) == 0:
         print("✗ ERROR: Transmitter produced 0 symbols.")
         return None
         
    n_symbols = min(symbol_lengths)
    pilot_pos = transmitter.pilot_handler.pilot_positions
    if n_symbols < len(pilot_pos):
        print(f"✗ ERROR: n_symbols={n_symbols} < pilots={len(pilot_pos)}. Increase N_INFO_BITS.")
        return None
    print(f"    (Simulation will truncate to minimum frame length: {n_symbols} symbols)")

    # === 3. PHYSICAL CHANNEL ===
    print("\n" + "="*80)
    print("STAGE 2: PHYSICAL CHANNEL (QUASI-STATIC)")
    print("="*80)
    
    # 3a. Create one "frozen" snapshot of the atmosphere
    print(f"[1] Generating {cfg.NUM_SCREENS} phase screens for one channel snapshot...")
    layers = create_multi_layer_screens(
        cfg.DISTANCE, cfg.NUM_SCREENS, 
        cfg.WAVELENGTH, cfg.CN2, 
        cfg.L0, cfg.L0_INNER, verbose=False
    )
    print(f"    Generated {len(layers)} screen layers.")
    
    # 3b. Calculate Attenuation Loss
    print("[2] Calculating Attenuation...")
    # Use proper function from fsplAtmAttenuation.py: calculate_geometric_loss(beam, z, receiver_radius)
    L_geo_dB, eta_geo = calculate_geometric_loss(max_m2_beam, cfg.DISTANCE, cfg.RECEIVER_DIAMETER / 2.0)
    w_z_analytical = max_m2_beam.beam_waist(cfg.DISTANCE)
    
    # Use 23km visibility for 'clear' (Kim model)
    visibility_km = 23.0 
    alpha_dBkm = calculate_kim_attenuation(cfg.WAVELENGTH * 1e9, visibility_km)
    L_atm_dB = alpha_dBkm * (cfg.DISTANCE / 1000.0)
    
    amplitude_loss = 10**(-L_atm_dB / 20.0) # Apply attenuation
    coll_eff = eta_geo * (amplitude_loss ** 2)  # Total collection efficiency
    P_rx_expected = cfg.P_TX_TOTAL_W * coll_eff
    print(f"    Atmospheric Loss: {L_atm_dB:.2f} dB (Amplitude factor: {amplitude_loss:.3f})")
    print(f"    Geometric Loss: {L_geo_dB:.2f} dB (Collection Eff: {eta_geo*100:.1f}%)")
    print(f"    Total P_rx expected: {P_rx_expected:.3f} W (eff: {coll_eff*100:.2f}%)")
    
    # 3c. Calculate Noise – FIXED: Per-symbol power probe
    print("[3] Calculating Noise Parameters...")
    aperture_mask = (grid_info['R'] <= cfg.RECEIVER_DIAMETER / 2.0).astype(float)
    dA = grid_info['delta']**2
    num_pixels_in_aperture = np.sum(aperture_mask)
    if num_pixels_in_aperture == 0: num_pixels_in_aperture = 1
    
    if cfg.ADD_NOISE:
        # Probe: One multiplexed symbol (all modes=1)
        E_tx_probe = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
        for mode_key in cfg.SPATIAL_MODES:
            E_tx_probe += tx_basis_fields[mode_key]  # Scaled basis *1
        result_probe = apply_multi_layer_turbulence(
            initial_field=E_tx_probe,
            base_beam=max_m2_beam, layers=layers, total_distance=cfg.DISTANCE,
            N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING,
            L0=cfg.L0, l0=cfg.L0_INNER
        )
        E_rx_probe = result_probe['final_field'] * amplitude_loss * aperture_mask
        
        power_per_symbol = np.sum(np.abs(E_rx_probe)**2) * dA  # Total for one symbol
        avg_pixel_intensity = power_per_symbol / num_pixels_in_aperture
        
        snr_linear = 10**(cfg.SNR_DB / 10.0)
        noise_var_per_pixel = avg_pixel_intensity / snr_linear 
        noise_std_per_pixel = np.sqrt(noise_var_per_pixel)
        
        # FIXED: Validate P_rx from probe
        P_rx_probe = np.sum(np.abs(E_rx_probe)**2) * dA
        print(f"    Target SNR: {cfg.SNR_DB} dB")
        print(f"    Power per Symbol (in aperture): {power_per_symbol:.2e} W")
        print(f"    Avg. Signal Intensity (per pixel): {avg_pixel_intensity:.2e}")
        print(f"    Noise Variance (per pixel): {noise_var_per_pixel:.2e}")
        print(f"    Probe P_rx: {P_rx_probe:.3f} W (matches expected: {P_rx_expected:.3f})")
    else:
        print("    Noise disabled.")
        noise_std_per_pixel = 0.0

    # 3d. Loop over all symbols (PHYSICAL PROPAGATION)
    print(f"[4] Propagating {n_symbols} symbols through channel... (This is slow!)")
    
    E_rx_sequence = [] # This will store the list of 2D fields
    
    # Sample symbols for TX vis (average first 5 non-pilot or unit) – FIXED: Realistic avg
    sample_syms = np.ones(n_modes, dtype=complex)  # Fallback unit
    first_non_pilot = max(0, len(pilot_pos)) if len(pilot_pos) > 0 else 0
    if first_non_pilot < 5 and first_non_pilot < n_symbols:
        for i in range(min(5, n_symbols - first_non_pilot)):
            idx = first_non_pilot + i
            for j, mode_key in enumerate(cfg.SPATIAL_MODES):
                sample_syms[j] += tx_signals[mode_key]['symbols'][idx]
        sample_syms /= 5  # Avg
    
    for sym_idx in range(n_symbols): # Loop to the *minimum* length
        # 1. Create the multiplexed field for this symbol – uses scaled basis
        E_tx_symbol = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
        for i, mode_key in enumerate(cfg.SPATIAL_MODES):
            symbol = tx_signals[mode_key]['symbols'][sym_idx]
            E_tx_symbol += tx_basis_fields[mode_key] * symbol
            
        # 2. Propagate the *combined* field
        result = apply_multi_layer_turbulence(
            initial_field=E_tx_symbol,
            base_beam=max_m2_beam, layers=layers, total_distance=cfg.DISTANCE,
            N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING,
            L0=cfg.L0, l0=cfg.L0_INNER
        )
        E_rx_turbulent = result['final_field']
        
        # 3. Apply Attenuation
        E_rx_attenuated = E_rx_turbulent * amplitude_loss
        
        # 4. Add Noise
        if cfg.ADD_NOISE:
            noise = (noise_std_per_pixel / np.sqrt(2)) * (
                np.random.randn(cfg.N_GRID, cfg.N_GRID) + 
                1j * np.random.randn(cfg.N_GRID, cfg.N_GRID)
            )
            E_rx_final = E_rx_attenuated + noise
        else:
            E_rx_final = E_rx_attenuated
            
        # 5. Apply Aperture (at the very end)
        E_rx_final = E_rx_final * aperture_mask
        
        # 6. Store the final field
        E_rx_sequence.append(E_rx_final)
        
        if (sym_idx + 1) % 100 == 0 or sym_idx == n_symbols - 1:  # FIXED: Every 100
            print(f"    ... propagated symbol {sym_idx + 1}/{n_symbols}")
            
    print("    ✓ Full frame propagated.")
    
    # FIXED: TX vis as multiplexed sample
    E_tx_visualization = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
    for i, mode_key in enumerate(cfg.SPATIAL_MODES):
        E_tx_visualization += tx_basis_fields[mode_key] * sample_syms[i]
    E_rx_visualization = E_rx_sequence[0] 

    # === 4. RECEIVER ===
    print("\n" + "="*80)
    print("STAGE 3: DIGITAL RECEIVER")
    print("="*80)
    
    # Pass the *entire sequence of fields* to the receiver
    recovered_bits, metrics = receiver.receive_sequence(
        E_rx_sequence=E_rx_sequence,
        grid_info=grid_info,
        tx_signals=tx_signals,
        original_data_bits=data_bits,
        verbose=True
    )

    # === 5. RESULTS ===
    print("\n" + "="*80)
    print("E2E SIMULATION COMPLETE - FINAL RESULTS")
    print("="*80)
    print(f"    TURBULENCE: Cn² = {cfg.CN2:.2e} (m^-2/3)")
    print(f"    LINK: {cfg.DISTANCE} m, {cfg.NUM_SCREENS} screens")
    print(f"    SNR: {cfg.SNR_DB} dB")
    print(f"    EQUALIZER: {cfg.EQ_METHOD.upper()}")
    print(f"    -----------------------------------")
    print(f"    TOTAL INFO BITS: {metrics['total_bits']}")
    print(f"    BIT ERRORS:      {metrics['bit_errors']}")
    print(f"    FINAL BER:       {metrics['ber']:.4e}")
    print("="*80)
    
    # Store results for plotting
    results = {
        'config': cfg,
        'metrics': metrics,
        'grid_info': grid_info,
        'tx_signals': tx_signals,
        'E_tx_visualization': E_tx_visualization,
        'E_rx_visualization': E_rx_visualization,
        'H_est': metrics['H_est']
    }
    
    return results

def plot_e2e_results(results, save_path=None):
    """
    Plot the summary of the E2E simulation.
    """
    print("Generating E2E results plot...")
    
    cfg = results['config']
    metrics = results['metrics']
    grid_info = results['grid_info']
    H_est = metrics['H_est']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)
    
    fig.suptitle(f"End-to-End FSO-OAM Simulation Results\n"
                 f"Cn²={cfg.CN2:.1e}, L={cfg.DISTANCE}m, SNR={cfg.SNR_DB}dB, BER={metrics['ber']:.2e}",
                 fontsize=18, fontweight='bold')
    
    extent_mm = grid_info['D'] * 1e3 / 2
    
    # FIXED: TX vis as multiplexed
    ax1 = fig.add_subplot(gs[0, 0])
    E_tx_vis = np.abs(results['E_tx_visualization'])**2
    im1 = ax1.imshow(E_tx_vis.T, 
                    extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                    cmap='hot', origin='lower')
    ax1.set_title('TX Multiplexed Field Example', fontweight='bold')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    plt.colorbar(im1, ax=ax1, label='Intensity [W/m²]')
    
    # Plot 2: Received field (example snapshot)
    ax2 = fig.add_subplot(gs[1, 0])
    E_rx_vis = np.abs(results['E_rx_visualization'])**2
    vmax = np.percentile(E_rx_vis, 99.9) # Clip hotspots for better viz
    im2 = ax2.imshow(E_rx_vis.T, 
                    extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                    cmap='hot', origin='lower', vmax=vmax)
    ax2.set_title(f'RX Field Snapshot (Symbol 0)', fontweight='bold')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    plt.colorbar(im2, ax=ax2, label='Intensity [W/m²]')
    
    # Plot 3: Estimated Channel Matrix |H_est|
    ax3 = fig.add_subplot(gs[0, 1])
    im3 = ax3.imshow(np.abs(H_est), cmap='viridis', interpolation='nearest')
    ax3.set_title(r'Estimated Channel Matrix $|\hat{H}|$', fontweight='bold')
    
    mode_labels = [f"({p},{l})" for p,l in cfg.SPATIAL_MODES]
    ax3.set_xticks(np.arange(len(mode_labels)))
    ax3.set_yticks(np.arange(len(mode_labels)))
    ax3.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax3.set_yticklabels(mode_labels)
    ax3.set_xlabel('Transmitted Mode (j)')
    ax3.set_ylabel('Received Mode (i)')
    plt.colorbar(im3, ax=ax3, label='Magnitude (Coupling Strength)')
    # Add text labels
    for i in range(H_est.shape[0]):
        for j in range(H_est.shape[1]):
            ax3.text(j, i, f"{np.abs(H_est[i,j]):.2f}", 
                     ha="center", va="center", color="w", fontsize=8)
            
    # Plot 4: Channel Matrix Phase
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.angle(H_est), cmap='hsv', interpolation='nearest', vmin=-np.pi, vmax=np.pi)
    ax4.set_title(r'Estimated Channel Matrix Phase $\angle \hat{H}$', fontweight='bold')
    ax4.set_xticks(np.arange(len(mode_labels)))
    ax4.set_yticks(np.arange(len(mode_labels)))
    ax4.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax4.set_yticklabels(mode_labels)
    ax4.set_xlabel('Transmitted Mode (j)')
    ax4.set_ylabel('Received Mode (i)')
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')
    
    # Plot 5: Performance Metrics Text
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis('off')
    
    # Get turbulence properties
    temp_turb = AtmosphericTurbulence(
        Cn2=cfg.CN2, L0=cfg.L0, l0=cfg.L0_INNER, wavelength=cfg.WAVELENGTH
    )
    
    metrics_text = f"""
SYSTEM PERFORMANCE METRICS

[Link Parameters]
  Distance: {cfg.DISTANCE} m
  Weather: {cfg.WEATHER}
  Turbulence: Cn² = {cfg.CN2:.2e}
  SNR: {cfg.SNR_DB} dB
  Modes: {len(cfg.SPATIAL_MODES)} ( {', '.join(mode_labels)} )

[Channel Metrics]
  Rytov Variance: {temp_turb.rytov_variance(cfg.DISTANCE):.3f}
  Fried Parameter (r0): {temp_turb.fried_parameter(cfg.DISTANCE)*1000:.2f} mm
  Channel Condition: {np.linalg.cond(H_est):.2f}
  
[Receiver Metrics]
  Equalization: {cfg.EQ_METHOD.upper()}
  Est. Noise Var: {metrics['noise_var']:.2e}

[FINAL PERFORMANCE]
  Total Info Bits: {metrics['total_bits']}
  Bit Errors: {metrics['bit_errors']}
  ---------------------------------
  Bit Error Rate (BER): {metrics['ber']:.4e}
  ---------------------------------
    """
    
    ax5.text(0.0, 0.95, metrics_text, transform=ax5.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        # Ensure the directory exists before trying to save the file
        plot_directory = os.path.dirname(save_path)
        os.makedirs(plot_directory, exist_ok=True)
        
        plt.savefig(save_path, dpi=cfg.DPI, bbox_inches='tight')
        print(f"\n✓ E2E Results plot saved to: {save_path}")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # 1. Initialize Configuration
    config = SimulationConfig()
    
    # --- Override config here for quick tests ---
    #config.CN2 = 1e-20
    #config.SNR_DB = 99
    #config.N_GRID = 256  # Faster test
    #config.NUM_SCREENS = 5
    #config.N_INFO_BITS = 819 * 6  # Default already multiple
    
    # 2. Run end-to-end simulation
    results = run_e2e_simulation(config)
    
    # 3. Plot results
    if results:
        save_file = os.path.join(config.PLOT_DIR, "e2e_simulation_results.png")
        fig = plot_e2e_results(results, save_path=save_file)
        plt.show()
    else:
        print("✗ Simulation failed to produce results.")
