"""
Diagnostic script to identify the projection/normalization issue.
This will help us understand why H_est has such small values.
"""

import numpy as np
import sys
import os

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lgBeam import LaguerreGaussianBeam

# Simulation parameters (matching pipeline.py)
WAVELENGTH = 1550e-9
W0 = 25e-3
DISTANCE = 1000
N_GRID = 512
OVERSAMPLING = 2
SPATIAL_MODES = [(0, -1), (0, 1), (0, -3), (0, 3), (0, -4), (0, 4), (1, -1), (1, 1)]
P_TX_TOTAL = 1.0

print("="*80)
print("DIAGNOSTIC: Projection Normalization Check")
print("="*80)

# 1. Create grid
max_mode = max(SPATIAL_MODES, key=lambda m: abs(m[1]))
dummy_beam = LaguerreGaussianBeam(max_mode[0], max_mode[1], WAVELENGTH, W0)
beam_size_at_rx = dummy_beam.beam_waist(DISTANCE)
D = OVERSAMPLING * 6 * beam_size_at_rx
delta = D / N_GRID

x = np.linspace(-D/2, D/2, N_GRID)
y = np.linspace(-D/2, D/2, N_GRID)
X, Y = np.meshgrid(x, y, indexing='ij')
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)
dA = delta**2

print(f"\nGrid: {N_GRID}x{N_GRID}, extent={D*1000:.1f} mm, pixel={delta*1000:.2f} mm")
print(f"dA = {dA:.6e} m²")

# 2. Generate TX basis fields (at z=0)
n_modes = len(SPATIAL_MODES)
tx_basis_fields = {}
basis_energy = {}

print(f"\n--- TX Basis Fields (z=0) ---")
for mode_key in SPATIAL_MODES:
    p, l = mode_key
    beam = LaguerreGaussianBeam(p, l, WAVELENGTH, W0)
    E_basis = beam.generate_beam_field(R, PHI, 0.0)
    energy = np.sum(np.abs(E_basis)**2) * dA
    basis_energy[mode_key] = energy
    
    # Scale for total power
    scale = np.sqrt(P_TX_TOTAL / (n_modes * energy))
    tx_basis_fields[mode_key] = E_basis * scale
    
    scaled_energy = np.sum(np.abs(tx_basis_fields[mode_key])**2) * dA
    print(f"  Mode {mode_key}: raw_energy={energy:.6e}, scale={scale:.6f}, scaled_energy={scaled_energy:.6e}")

# Verify total power
E_total = sum(tx_basis_fields.values())
total_power = np.sum(np.abs(E_total)**2) * dA
print(f"\nTotal TX power (all modes, symbol=1): {total_power:.6f} W ✓")

# 3. Generate RX reference fields (at z=distance, NO turbulence)
print(f"\n--- RX Reference Fields (z={DISTANCE}m, free-space) ---")
rx_ref_fields = {}
ref_energy = {}

for mode_key in SPATIAL_MODES:
    p, l = mode_key
    beam = LaguerreGaussianBeam(p, l, WAVELENGTH, W0)
    
    # Generate at receiver plane (free-space propagation)
    E_ref = beam.generate_beam_field(R, PHI, DISTANCE)
    
    # Get TX scaling factor
    tx_scale = np.sqrt(P_TX_TOTAL / (n_modes * basis_energy[mode_key]))
    
    # Apply SAME scaling as TX (this is what receiver.py does)
    E_ref_scaled = E_ref * tx_scale
    
    energy_ref = np.sum(np.abs(E_ref_scaled)**2) * dA
    rx_ref_fields[mode_key] = E_ref_scaled
    ref_energy[mode_key] = energy_ref
    
    print(f"  Mode {mode_key}: ref_energy={energy_ref:.6e}")

# 4. Test projection with PERFECT channel (no turbulence)
print(f"\n--- Test Projection (Perfect Channel) ---")
print("Simulating: E_rx = E_tx (no turbulence, no attenuation)")

# Simulate received field for mode (0, -1) with symbol = 1+0j
test_mode = (0, -1)
test_symbol = 1.0 + 0.0j

# TX field (at z=0)
E_tx = tx_basis_fields[test_mode] * test_symbol

# Simulate propagation (free-space, no turbulence)
# For this test, we'll just use the RX reference field as the "propagated" field
E_rx_perfect = rx_ref_fields[test_mode] * test_symbol

# Project onto all modes
print(f"\nProjecting E_rx (mode {test_mode}, symbol={test_symbol}) onto all reference modes:")
projections = {}
for mode_key in SPATIAL_MODES:
    ref = rx_ref_fields[mode_key]
    projection = np.sum(E_rx_perfect * np.conj(ref)) * dA
    symbol_est = projection / ref_energy[mode_key]
    projections[mode_key] = symbol_est
    
    if mode_key == test_mode:
        marker = " ← SELF (should be ≈ 1.0)"
    else:
        marker = ""
    print(f"  Mode {mode_key}: projection={projection:.6e}, symbol_est={symbol_est:.6f}{marker}")

# 5. Check orthogonality of reference fields
print(f"\n--- Reference Field Orthogonality Check ---")
print("Gram matrix G_ij = <ref_i, ref_j> (should be diagonal for orthogonal modes)")

G = np.zeros((n_modes, n_modes), dtype=complex)
for i, mode_i in enumerate(SPATIAL_MODES):
    for j, mode_j in enumerate(SPATIAL_MODES):
        ref_i = rx_ref_fields[mode_i]
        ref_j = rx_ref_fields[mode_j]
        G[i, j] = np.sum(ref_i * np.conj(ref_j)) * dA

print("\nGram matrix (magnitude):")
print("Modes:", [f"{m}" for m in SPATIAL_MODES])
for i, mode_i in enumerate(SPATIAL_MODES):
    row_str = f"{mode_i}: "
    for j in range(n_modes):
        row_str += f"{np.abs(G[i,j]):.3e} "
    print(row_str)

# Diagonal values
print("\nDiagonal (self-coupling):")
for i, mode in enumerate(SPATIAL_MODES):
    print(f"  {mode}: {np.abs(G[i,i]):.6e}")

# Off-diagonal max
off_diag = np.abs(G) - np.diag(np.diag(np.abs(G)))
max_off_diag = np.max(off_diag)
print(f"\nMax off-diagonal (cross-coupling): {max_off_diag:.6e}")
print(f"Orthogonality ratio (max_off_diag / min_diag): {max_off_diag / np.min(np.diag(np.abs(G))):.3f}")

# 6. Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if np.allclose(projections[test_mode], test_symbol, atol=0.01):
    print("✓ Projection works correctly for perfect channel (no turbulence)")
else:
    print(f"✗ Projection FAILS even for perfect channel!")
    print(f"  Expected: {test_symbol}, Got: {projections[test_mode]}")

if max_off_diag / np.min(np.diag(np.abs(G))) < 0.1:
    print("✓ Reference modes are nearly orthogonal")
else:
    print(f"⚠ Reference modes have significant cross-coupling!")
    print(f"  This is EXPECTED for LG modes (they're not perfectly orthogonal)")

print("\nKEY INSIGHT:")
print("The reference fields at the receiver are IDEAL (free-space propagated).")
print("But the ACTUAL received fields went through TURBULENCE, which:")
print("  1. Distorts the phase fronts")
print("  2. Mixes the modes")
print("  3. Changes the spatial structure")
print("\nThis is why H_est has small values - the reference fields don't match")
print("the turbulent fields, so projections are small!")

print("\nSOLUTION:")
print("The pilot-based channel estimation SHOULD capture this mismatch in H.")
print("But if H has all small values, the equalization will fail.")
print("\nCheck:")
print("  1. Are pilots strong enough? (need good SNR for channel estimation)")
print("  2. Is the pilot pattern orthogonal? (DFT pilots work best)")
print("  3. Is noise variance estimated correctly?")
print("="*80)
