"""
evaluate_with_fec.py
Evaluates the *full* system performance by wrapping the trained
CNN "soft demodulator" with an external LDPC code.

This script IS the final simulation.
"""
import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import sys
import os

# --- Import our custom files ---
import ldpc_decoder as ldpc # <-- IMPORT OUR NEW FILE
from cnn_model import FSO_MultiTask_CNN
from generate_dataset import DataConfig, QPSKModulator, project_field # Need to re-import
from turbulence import (create_multi_layer_screens, 
                        apply_multi_layer_turbulence, 
                        generate_phase_screen)
from fsplAtmAttenuation import calculate_kim_attenuation
from lgBeam import LaguerreGaussianBeam

# --- 1. Simulation Setup ---
MODEL_FILE = "fso_multitask_model.pth"
TEST_CN2 = 1e-14  
N_TEST_WORDS = 100 
N_SCREENS = 15
LDPC_MAX_ITER = 20 # Max iterations for the soft decoder

# --- LDPC Code Setup ---
n_ldpc = 512  # Codeword length (bits)
k_ldpc = 256  # Information bits
m_ldpc = n_ldpc - k_ldpc
d_v, d_c = 3, 6 # Variable and check node degrees (must satisfy m*d_c = n*d_v)
if m_ldpc*d_c != n_ldpc*d_v:
    raise ValueError(f"Invalid LDPC parameters: {m_ldpc}*{d_c} != {n_ldpc}*{d_v}")

print("Generating LDPC code...")
ldpc_H = ldpc.make_ldpc_H(m_ldpc, n_ldpc, d_v, d_c)
ldpc_G = ldpc.get_G_from_H(ldpc_H)
print(f"LDPC Code: n={n_ldpc}, k={k_ldpc}, Rate={k_ldpc/n_ldpc:.2f}")


# --- 2. Load Trained CNN Model ---
print(f"--- Loading trained CNN from {MODEL_FILE} ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
cfg = DataConfig()
model = FSO_MultiTask_CNN(n_modes=cfg.N_MODES).to(device)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
except Exception as e:
    print(f"Error loading model: {e}. Did you run train.py?")
    sys.exit(1)
model.eval()
print("Model loaded successfully.")

# --- 3. Initialize Physics Engine (Copied from generate_dataset.py) ---
print("Initializing physics simulation engine...")
qpsk = QPSKModulator(symbol_energy=1.0)
max_m2_beam = LaguerreGaussianBeam(0, max(abs(l) for p,l in cfg.SPATIAL_MODES), cfg.WAVELENGTH, cfg.W0)
beam_size_at_rx = max_m2_beam.beam_waist(cfg.DISTANCE)
D = cfg.OVERSAMPLING * 6 * beam_size_at_rx
delta = D / cfg.N_GRID
x = np.linspace(-D/2, D/2, cfg.N_GRID); y = np.linspace(-D/2, D/2, cfg.N_GRID)
X, Y = np.meshgrid(x, y, indexing='ij')
R = np.sqrt(X**2 + Y**2); PHI = np.arctan2(Y, X)
L_atm_dB = calculate_kim_attenuation(cfg.WAVELENGTH*1e9, 23.0) * (cfg.DISTANCE / 1000.0)
amplitude_loss = 10**(-L_atm_dB / 20.0)
aperture_mask = (R <= cfg.RECEIVER_DIAMETER / 2.0).astype(float)
tx_basis_fields = {}
for mode_key in cfg.SPATIAL_MODES:
    p, l = mode_key
    beam = LaguerreGaussianBeam(p, l, cfg.WAVELENGTH, cfg.W0)
    tx_basis_fields[mode_key] = beam.generate_beam_field(R, PHI, 0)

# --- 4. Run Full System Simulation ---
print(f"--- Starting evaluation for {N_TEST_WORDS} codewords at Cn2={TEST_CN2:.1e} ---")

total_info_bits = 0
total_info_errors = 0
total_coded_bits = 0
total_coded_errors = 0

for _ in tqdm(range(N_TEST_WORDS), desc="Simulating Codewords"):
    
    # --- A. TX: Generate and Encode Info Bits ---
    info_bits = np.random.randint(0, 2, k_ldpc) # 256 info bits
    coded_bits = ldpc.ldpc_encode(ldpc_G, info_bits) # 512 coded bits
    
    # --- B. TX: Modulate and Propagate ---
    n_chunks = int(np.ceil(n_ldpc / cfg.N_BITS_PER_SYMBOL)) # 512 / 12 = 43 chunks
    padding = n_chunks * cfg.N_BITS_PER_SYMBOL - n_ldpc # 516 - 512 = 4 bits
    coded_bits_padded = np.pad(coded_bits, (0, padding), 'constant')
    
    received_llrs = []
    
    layers = create_multi_layer_screens(
        cfg.DISTANCE, N_SCREENS, cfg.WAVELENGTH, 
        TEST_CN2, cfg.L0, cfg.L0_INNER, verbose=False
    )
    phase_screens_list = []
    for layer in layers:
        phase_screens_list.append(generate_phase_screen(
            layer['r0_layer'], cfg.N_GRID, delta, cfg.L0, cfg.L0_INNER
        ))
    
    for i in range(n_chunks):
        tx_bits_chunk = coded_bits_padded[i*cfg.N_BITS_PER_SYMBOL : (i+1)*cfg.N_BITS_PER_SYMBOL]
        tx_symbols = qpsk.modulate_bits(tx_bits_chunk)
        
        E_tx_mux = np.zeros((cfg.N_GRID, cfg.N_GRID), dtype=complex)
        for j, tx_key in enumerate(cfg.SPATIAL_MODES):
            E_tx_mux += tx_symbols[j] * tx_basis_fields[tx_key]
        
        result = apply_multi_layer_turbulence(
            E_tx_mux, max_m2_beam, layers, cfg.DISTANCE,
            N=cfg.N_GRID, oversampling=cfg.OVERSAMPLING, 
            L0=cfg.L0, l0=cfg.L0_INNER,
            phase_screens=phase_screens_list 
        )
        E_rx_mux_turb = result['final_field'] * amplitude_loss * aperture_mask
        
        # --- C. RX: CNN Demodulation (Get LLRs) ---
        X_rx = np.stack([np.real(E_rx_mux_turb), np.imag(E_rx_mux_turb)], axis=-1)
        X_tensor = torch.from_numpy(X_rx).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            llrs_pred, _ = model(X_tensor) 
        
        received_llrs.extend(llrs_pred.cpu().numpy().flatten())

    # --- D. RX: Decode LLRs ---
    llrs_codeword = np.array(received_llrs[:n_ldpc]) # Un-pad
    
    # --- RECTIFIED: Call our new SOFT decoder ---
    # Our CNN outputs logits (LLRs), and our decoder takes LLRs. No conversion needed.
    # LLR = log(P(0)/P(1))
    final_llrs, decoded_coded_bits = ldpc.ldpc_decode_spa(ldpc_H, llrs_codeword, max_iter=LDPC_MAX_ITER)
    
    # The decoder returns the *full* corrected codeword. We extract the info bits.
    decoded_info_bits = decoded_coded_bits[:k_ldpc]
    
    # --- E. Calculate Errors ---
    # 1. "Pre-FEC" / Coded Bit Error Rate (Raw CNN performance)
    # This is the "hard decision" on the CNN's output, *before* decoding
    hard_bits_pre_fec = (llrs_codeword < 0).astype(int) 
    coded_errors = np.sum(hard_bits_pre_fec != coded_bits)
    
    # 2. "Post-FEC" / Information Bit Error Rate (Final system performance)
    info_errors = np.sum(decoded_info_bits != info_bits)
    
    total_coded_bits += n_ldpc
    total_coded_errors += coded_errors
    total_info_bits += k_ldpc
    total_info_errors += info_errors

# --- 5. Report Final Results ---
print("\n--- FINAL EVALUATION COMPLETE ---")
pre_fec_ber = (total_coded_errors / total_coded_bits) if total_coded_bits > 0 else 0
post_fec_ber = (total_info_errors / total_info_bits) if total_info_bits > 0 else 0

print(f"  Test Conditions: Cn2 = {TEST_CN2:.1e}, Codewords = {N_TEST_WORDS}")
print(f"\n  Pre-FEC (Raw CNN) BER: {pre_fec_ber:.4e}")
print(f"     (Coded Bits: {total_coded_bits}, Coded Errors: {total_coded_errors})")
print(f"\n  Post-FEC (Final) BER:  {post_fec_ber:.4e}")
print(f"     (Info Bits: {total_info_bits}, Info Errors: {total_info_errors})")

if post_fec_ber == 0.0 and total_coded_errors > 0:
    print("\n✓ SUCCESS: The LDPC code corrected all errors!")
elif total_coded_errors == 0:
    print("\n✓ SUCCESS: The CNN was perfect (no errors to correct).")
else:
    print(f"\n✗ WARNING: The LDPC code could not correct all errors (or was perfect).")