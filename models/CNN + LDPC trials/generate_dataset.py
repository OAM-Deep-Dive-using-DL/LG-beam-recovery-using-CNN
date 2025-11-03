"""
generate_dataset.py
Physics-informed data generator for the FSO-OAM CNN.
Generates (X, Y_bits, Y_H) triplets.
- X: The received 2D field (128x128x2)
- Y_bits: The transmitted bits (12,)
- Y_H: The "ground truth" 6x6 complex channel matrix (6,6,2)
"""

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import warnings

# --- Import Your Physics Engine ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

try:
    from lgBeam import LaguerreGaussianBeam
    from turbulence import ( create_multi_layer_screens, apply_multi_layer_turbulence, angular_spectrum_propagation, generate_phase_screen) 
    from fsplAtmAttenuation import calculate_kim_attenuation
except ImportError as e:
    print(f"✗ E2E Simulation Import Error: {e}")
    sys.exit(1)

# --- QPSKModulator (copied from your encoding.py) ---
class QPSKModulator:
    def __init__(self, symbol_energy=1.0):
        self.A = np.sqrt(symbol_energy)
        self.constellation_map = {
            (0, 0): self.A * (1 + 1j) / np.sqrt(2),
            (0, 1): self.A * (-1 + 1j) / np.sqrt(2),
            (1, 1): self.A * (-1 - 1j) / np.sqrt(2),
            (1, 0): self.A * (1 - 1j) / np.sqrt(2)
        }
    def modulate_bits(self, bits):
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([self.constellation_map[tuple(pair)] for pair in bit_pairs])
        return symbols
# --- End of QPSKModulator ---


# --- Data Generation Configuration ---
class DataConfig:
    # Link
    WAVELENGTH = 1550e-9
    W0 = 25e-3
    DISTANCE = 1200
    RECEIVER_DIAMETER = 0.3
    SPATIAL_MODES = [(0, -1), (0, 1), (0, -3), (0, 3), (0, -4), (0, 4)]
    
    # --- DIVERSITY PARAMETERS ---
    # We will sample C_n^2 logarithmically from 10^-17 (weak) to 10^-12 (very strong)
    CN2_RANGE = [1e-18, 1e-11] 
    
    # We will randomly vary the number of screens to simulate different path profiles
    NUM_SCREENS_RANGE = [5, 35] 
    
    # --- BASE PARAMETERS ---
    L0 = 10.0
    L0_INNER = 0.005
    
    # Simulation
    N_GRID = 128
    OVERSAMPLING = 2
    
    # Dataset parameters
    N_SNAPSHOTS = 1000  # Number of *different turbulence screens*
    N_SAMPLES_PER_SNAPSHOT = 300 # Number of bit combinations per screen
    
    # Derived
    @property
    def N_MODES(self): return len(self.SPATIAL_MODES)
    @property
    def N_BITS_PER_SYMBOL(self): return self.N_MODES * 2
    
    @property
    def TOTAL_SAMPLES(self): return self.N_SNAPSHOTS * self.N_SAMPLES_PER_SNAPSHOT
    
    @property
    def INPUT_SHAPE(self): return (self.N_GRID, self.N_GRID, 2)
    @property
    def OUTPUT_SHAPE_BITS(self): return (self.N_BITS_PER_SYMBOL,)
    @property
    def OUTPUT_SHAPE_H(self): return (self.N_MODES, self.N_MODES, 2)
    @property
    def OUTPUT_SHAPE_ENV(self): return (2,) # To store [cn2, num_screens]


def project_field(E_rx, E_ref, delta):
    """Calculates the normalized projection of E_rx onto E_ref."""
    dA = delta**2
    # Normalization factor: <E_ref, E_ref>
    ref_energy = np.sum(np.abs(E_ref)**2) * dA
    if ref_energy < 1e-20:
        return 0.0 + 0.0j
    # Projection: <E_rx, E_ref>
    projection = np.sum(E_rx * np.conj(E_ref)) * dA
    return projection / ref_energy



def generate_dataset(config, filename="fso_cnn_dataset.h5"):
    """
    Generates and saves the full (X, Y_bits, Y_H, Y_env) dataset to an HDF5 file.
    Samples Cn2 and NUM_SCREENS randomly for each snapshot.
    """
    cfg = config
    
    # --- 1. Initialize Components ---
    qpsk = QPSKModulator(symbol_energy=1.0)
    
    # Initialize Grid
    max_m2_beam = LaguerreGaussianBeam(0, max(abs(l) for p,l in cfg.SPATIAL_MODES), cfg.WAVELENGTH, cfg.W0) # Corrected to use abs(l)
    beam_size_at_rx = max_m2_beam.beam_waist(cfg.DISTANCE)
    D = cfg.OVERSAMPLING * 6 * beam_size_at_rx
    delta = D / cfg.N_GRID
    x = np.linspace(-D/2, D/2, cfg.N_GRID); y = np.linspace(-D/2, D/2, cfg.N_GRID)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2); PHI = np.arctan2(Y, X)
    
    # Attenuation and Aperture
    L_atm_dB = calculate_kim_attenuation(cfg.WAVELENGTH*1e9, 23.0) * (cfg.DISTANCE / 1000.0)
    amplitude_loss = 10**(-L_atm_dB / 20.0)
    aperture_mask = (R <= cfg.RECEIVER_DIAMETER / 2.0).astype(float)
    
    # --- 2. Generate Basis Fields (TX and RX) ---
    print("Generating basis fields...")
    tx_basis_fields = {} # z=0 fields
    rx_basis_fields = {} # z=DISTANCE, free-space propagated fields (our reference)
    
    for mode_key in cfg.SPATIAL_MODES:
        p, l = mode_key
        beam = LaguerreGaussianBeam(p, l, cfg.WAVELENGTH, cfg.W0)
        E_z0 = beam.generate_beam_field(R, PHI, 0)
        tx_basis_fields[mode_key] = E_z0
        
        # Propagate free-space (no turbulence) to get the RX reference mode
        E_zL_pristine = angular_spectrum_propagation(E_z0.copy(), delta, cfg.WAVELENGTH, cfg.DISTANCE)
        rx_basis_fields[mode_key] = E_zL_pristine * amplitude_loss * aperture_mask

    # --- 3. Initialize HDF5 File ---
    print(f"Creating HDF5 file at {filename}...")
    with h5py.File(filename, "w") as hf:
        # Create resizable datasets
        X_dset = hf.create_dataset("X", 
                                   shape=(cfg.TOTAL_SAMPLES, *cfg.INPUT_SHAPE), 
                                   maxshape=(None, *cfg.INPUT_SHAPE), 
                                   dtype='float32', chunks=True)
        
        Y_bits_dset = hf.create_dataset("Y_bits", 
                                        shape=(cfg.TOTAL_SAMPLES, *cfg.OUTPUT_SHAPE_BITS), 
                                        maxshape=(None, *cfg.OUTPUT_SHAPE_BITS),
                                        dtype='int8', chunks=True)
        
        Y_H_dset = hf.create_dataset("Y_H", 
                                     shape=(cfg.TOTAL_SAMPLES, *cfg.OUTPUT_SHAPE_H), 
                                     maxshape=(None, *cfg.OUTPUT_SHAPE_H),
                                     dtype='float32', chunks=True)
        
        Y_env_dset = hf.create_dataset("Y_env",
                                       shape=(cfg.TOTAL_SAMPLES, *cfg.OUTPUT_SHAPE_ENV),
                                       maxshape=(None, *cfg.OUTPUT_SHAPE_ENV),
                                       dtype='float32', chunks=True)
        
        
        # --- 4. Main Generation Loop ---
        idx_counter = 0
        for i_snap in tqdm(range(cfg.N_SNAPSHOTS), desc="Generating Snapshots"):
            
            # --- A. Create a new RANDOMIZED turbulence snapshot ---
            log_cn2_min = np.log10(cfg.CN2_RANGE[0])
            log_cn2_max = np.log10(cfg.CN2_RANGE[1])
            random_log_cn2 = np.random.uniform(log_cn2_min, log_cn2_max)
            current_cn2 = 10**random_log_cn2
            
            current_num_screens = np.random.randint(cfg.NUM_SCREENS_RANGE[0], 
                                                    cfg.NUM_SCREENS_RANGE[1] + 1)
            
            layers = create_multi_layer_screens(
                cfg.DISTANCE, current_num_screens, cfg.WAVELENGTH, 
                current_cn2, cfg.L0, cfg.L0_INNER, verbose=False
            )
            
            # --- CRITICAL FIX: PRE-GENERATE PHASE SCREENS ---
            # This ensures the same physical turbulence is used for
            # H-probing and all 300 data samples in this snapshot.
            phase_screens_list = []
            for layer in layers:
                phi = generate_phase_screen(
                    layer['r0_layer'], 
                    cfg.N_GRID, 
                    delta, 
                    cfg.L0, 
                    cfg.L0_INNER
                )
                phase_screens_list.append(phi)
            # --- END OF FIX ---

            # --- B. Probe H_true for this snapshot ---
            H_true = np.zeros((cfg.N_MODES, cfg.N_MODES), dtype=complex)
            for j, tx_key in enumerate(cfg.SPATIAL_MODES):
                E_tx_probe = tx_basis_fields[tx_key]
                
                # --- CRITICAL FIX: PASS PRE-GENERATED SCREENS ---
                result = apply_multi_layer_turbulence(
                    E_tx_probe, max_m2_beam, layers, cfg.DISTANCE,
                    N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING, 
                    L0=cfg.L0, l0=cfg.L0_INNER,
                    phase_screens=phase_screens_list # <-- PASS SCREENS
                )
                E_rx_turb = result['final_field'] * amplitude_loss * aperture_mask
                
                for i, rx_key in enumerate(cfg.SPATIAL_MODES):
                    E_ref = rx_basis_fields[rx_key]
                    H_true[i, j] = project_field(E_rx_turb, E_ref, delta)
            
            H_true_ri = np.stack([np.real(H_true), np.imag(H_true)], axis=-1)
            
            # --- C. Generate data samples for this H ---
            for _ in range(cfg.N_SAMPLES_PER_SNAPSHOT):
                if idx_counter >= cfg.TOTAL_SAMPLES: break
                
                tx_bits = np.random.randint(0, 2, cfg.N_BITS_PER_SYMBOL)
                tx_symbols = qpsk.modulate_bits(tx_bits)
                
                E_tx_mux = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
                for j, tx_key in enumerate(cfg.SPATIAL_MODES):
                    E_tx_mux += tx_symbols[j] * tx_basis_fields[tx_key]
                
                # --- CRITICAL FIX: PASS PRE-GENERATED SCREENS ---
                result = apply_multi_layer_turbulence(
                    E_tx_mux, max_m2_beam, layers, cfg.DISTANCE,
                    N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING, 
                    L0=cfg.L0, l0=cfg.L0_INNER,
                    phase_screens=phase_screens_list # <-- PASS SAME SCREENS
                )
                E_rx_mux_turb = result['final_field'] * amplitude_loss * aperture_mask
                
                # Store (X, Y_bits, Y_H, Y_env)
                X_dset[idx_counter, :, :, 0] = np.real(E_rx_mux_turb)
                X_dset[idx_counter, :, :, 1] = np.imag(E_rx_mux_turb)
                Y_bits_dset[idx_counter] = tx_bits
                Y_H_dset[idx_counter] = H_true_ri
                Y_env_dset[idx_counter] = [current_cn2, current_num_screens] # Save metadata
                
                idx_counter += 1
                
    print(f"\nDataset generation complete.")
    print(f"  Total samples: {idx_counter}")
    print(f"  X shape: {X_dset.shape}")
    print(f"  Y_bits shape: {Y_bits_dset.shape}")
    print(f"  Y_H shape: {Y_H_dset.shape}")
    print(f"  Y_env shape: {Y_env_dset.shape}")

# This part is fine
if __name__ == "__main__":
    np.random.seed(42)
    config = DataConfig()
    generate_dataset(config, filename="fso_cnn_dataset.h5")