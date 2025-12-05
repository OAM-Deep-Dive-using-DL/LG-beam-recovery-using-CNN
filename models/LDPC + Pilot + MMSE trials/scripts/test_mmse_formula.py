"""
Minimal test to verify MMSE vs ZF equalization formulas.
"""

import numpy as np
from numpy.linalg import inv, pinv

# Create a simple 2x2 system
M = 2
H_true = np.array([[0.8, 0.2], [0.1, 0.9]])  # Well-conditioned channel

# Transmitted symbols (QPSK)
S_true = np.array([[1+1j, -1+1j, -1-1j, 1-1j],
                   [1-1j, 1+1j, -1-1j, -1+1j]]) / np.sqrt(2)

# Received symbols (no noise)
Y = H_true @ S_true

print("="*80)
print("EQUALIZATION TEST: ZF vs MMSE")
print("="*80)
print(f"\nTrue channel H:")
print(H_true)
print(f"\nTransmitted symbols S:")
print(S_true)
print(f"\nReceived symbols Y = H*S:")
print(Y)

# ZF Equalization
print("\n" + "="*80)
print("ZERO-FORCING (ZF)")
print("="*80)
W_zf = inv(H_true)
S_est_zf = W_zf @ Y
error_zf = np.linalg.norm(S_est_zf - S_true, 'fro')
print(f"W_zf = inv(H):")
print(W_zf)
print(f"\nEstimated symbols S_est = W_zf @ Y:")
print(S_est_zf)
print(f"\nReconstruction error: {error_zf:.6e}")

# MMSE Equalization (no noise case)
print("\n" + "="*80)
print("MMSE (no noise, σ²=0)")
print("="*80)
sigma2 = 0.0
# MMSE formula: W = H^H @ inv(H @ H^H + σ²*I)
W_mmse_nonoise = H_true.conj().T @ inv(H_true @ H_true.conj().T + sigma2 * np.eye(M))
S_est_mmse_nonoise = W_mmse_nonoise @ Y
error_mmse_nonoise = np.linalg.norm(S_est_mmse_nonoise - S_true, 'fro')
print(f"W_mmse = H^H @ inv(H @ H^H + 0*I):")
print(W_mmse_nonoise)
print(f"\nEstimated symbols S_est = W_mmse @ Y:")
print(S_est_mmse_nonoise)
print(f"\nReconstruction error: {error_mmse_nonoise:.6e}")

# MMSE with small noise
print("\n" + "="*80)
print("MMSE (with noise, σ²=0.01)")
print("="*80)
sigma2 = 0.01
W_mmse_noise = H_true.conj().T @ inv(H_true @ H_true.conj().T + sigma2 * np.eye(M))
S_est_mmse_noise = W_mmse_noise @ Y
error_mmse_noise = np.linalg.norm(S_est_mmse_noise - S_true, 'fro')
print(f"W_mmse = H^H @ inv(H @ H^H + 0.01*I):")
print(W_mmse_noise)
print(f"\nEstimated symbols S_est = W_mmse @ Y:")
print(S_est_mmse_noise)
print(f"\nReconstruction error: {error_mmse_noise:.6e}")

# Now test with ACTUAL noise
print("\n" + "="*80)
print("WITH ACTUAL NOISE")
print("="*80)
np.random.seed(42)
noise = np.sqrt(0.01/2) * (np.random.randn(M, 4) + 1j * np.random.randn(M, 4))
Y_noisy = Y + noise

print(f"Noise variance: 0.01")
print(f"Noisy received Y:")
print(Y_noisy)

# ZF with noise
S_est_zf_noisy = W_zf @ Y_noisy
error_zf_noisy = np.linalg.norm(S_est_zf_noisy - S_true, 'fro')
print(f"\nZF reconstruction error (with noise): {error_zf_noisy:.6e}")

# MMSE with noise
S_est_mmse_noisy = W_mmse_noise @ Y_noisy
error_mmse_noisy = np.linalg.norm(S_est_mmse_noisy - S_true, 'fro')
print(f"MMSE reconstruction error (with noise): {error_mmse_noisy:.6e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if error_mmse_noisy < error_zf_noisy:
    print(f"✓ MMSE is better than ZF (as expected with noise)")
    print(f"  MMSE error: {error_mmse_noisy:.6e}")
    print(f"  ZF error: {error_zf_noisy:.6e}")
    print(f"  Improvement: {(1 - error_mmse_noisy/error_zf_noisy)*100:.1f}%")
else:
    print(f"✗ Something is wrong! ZF should be worse than MMSE with noise")

print("\n" + "="*80)
print("FORMULA VERIFICATION")
print("="*80)
print("\nThe MMSE formula being used:")
print("  W_mmse = H^H @ inv(H @ H^H + σ²*I)")
print("\nThis is CORRECT for the system Y = H*S + N")
print("where N is complex Gaussian noise with variance σ²")
print("\nFor your system:")
print("  - H_est has small values (0.01-0.13)")
print("  - noise_var = 0.25 (large!)")
print("  - This means MMSE will heavily regularize")
print("\nThe issue is NOT the MMSE formula, but:")
print("  1. H_est is too small (projection mismatch)")
print("  2. 'noise_var' is actually MODEL ERROR, not noise")
print("  3. MMSE thinks there's huge noise, so it over-regularizes")
print("="*80)
