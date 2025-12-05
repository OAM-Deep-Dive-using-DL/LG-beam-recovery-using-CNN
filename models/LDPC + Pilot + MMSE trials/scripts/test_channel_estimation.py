"""
Test channel estimation with synthetic data to verify the LS formula is correct.
"""

import numpy as np
from numpy.linalg import inv, pinv

print("="*80)
print("CHANNEL ESTIMATION TEST")
print("="*80)

# Create a synthetic channel
M = 8  # Number of modes
N_pilots = 116  # Number of pilot symbols

# True channel matrix (random for testing)
np.random.seed(42)
H_true = (np.random.randn(M, M) + 1j * np.random.randn(M, M)) / np.sqrt(2 * M)
# Make it more diagonal-dominant (realistic for weak turbulence)
H_true = H_true * 0.1 + np.eye(M) * 0.5

print(f"\nTrue channel H (magnitude):")
print(np.abs(H_true))
print(f"cond(H_true) = {np.linalg.cond(H_true):.2e}")

# Generate pilot symbols (random QPSK, like your system)
pilot_constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
pilot_indices = np.random.randint(0, 4, size=(M, N_pilots))
P_p = pilot_constellation[pilot_indices]  # (M, N_pilots)

print(f"\nPilot matrix P_p: shape {P_p.shape}")
print(f"Pilot power (per symbol): {np.mean(np.abs(P_p)**2):.3f} (should be ≈1.0)")

# Check pilot Gram matrix
PPH = P_p @ P_p.conj().T
print(f"\nPilot Gram matrix P*P^H:")
print(f"  Diagonal (should be ≈{N_pilots}):")
for i in range(M):
    print(f"    Mode {i}: {np.abs(PPH[i,i]):.1f}")
print(f"  Max off-diagonal: {np.max(np.abs(PPH - np.diag(np.diag(PPH)))):.2f}")
print(f"  cond(P*P^H) = {np.linalg.cond(PPH):.2e}")

# Generate received pilots (no noise)
Y_p = H_true @ P_p  # (M, N_pilots)

print(f"\nReceived pilot matrix Y_p: shape {Y_p.shape}")

# Estimate channel using LS
print("\n--- LS Channel Estimation ---")
H_est_method1 = Y_p @ P_p.conj().T @ inv(PPH)
print(f"Method 1: H_est = Y_p @ P_p^H @ inv(P_p @ P_p^H)")
print(f"  H_est magnitude:")
print(np.abs(H_est_method1))

# Alternative method (should be equivalent)
H_est_method2 = Y_p @ pinv(P_p)
print(f"\nMethod 2: H_est = Y_p @ pinv(P_p)")
print(f"  H_est magnitude:")
print(np.abs(H_est_method2))

# Check error
error1 = np.linalg.norm(H_est_method1 - H_true, 'fro') / np.linalg.norm(H_true, 'fro')
error2 = np.linalg.norm(H_est_method2 - H_true, 'fro') / np.linalg.norm(H_true, 'fro')

print(f"\n--- Estimation Error ---")
print(f"Method 1 relative error: {error1:.6e}")
print(f"Method 2 relative error: {error2:.6e}")

if error1 < 1e-10 and error2 < 1e-10:
    print("\n✓ LS estimation is PERFECT (as expected for noiseless case)")
else:
    print(f"\n✗ LS estimation has error! (should be ~0 for noiseless)")

# Now test with NOISE
print("\n" + "="*80)
print("TEST WITH NOISE")
print("="*80)

noise_var = 0.01  # Small noise
noise = np.sqrt(noise_var/2) * (np.random.randn(M, N_pilots) + 1j * np.random.randn(M, N_pilots))
Y_p_noisy = Y_p + noise

H_est_noisy = Y_p_noisy @ P_p.conj().T @ inv(PPH)

error_noisy = np.linalg.norm(H_est_noisy - H_true, 'fro') / np.linalg.norm(H_true, 'fro')
print(f"Noise variance: {noise_var:.3e}")
print(f"Estimation error with noise: {error_noisy:.6e}")

# Estimate noise variance from residuals
residual = Y_p_noisy - H_est_noisy @ P_p
noise_var_est = np.mean(np.abs(residual)**2)
print(f"True noise variance: {noise_var:.3e}")
print(f"Estimated noise variance: {noise_var_est:.3e}")
print(f"Ratio: {noise_var_est / noise_var:.3f} (should be ≈1.0)")

# Key insight
print("\n" + "="*80)
print("KEY INSIGHT FOR YOUR SYSTEM")
print("="*80)
print("\nYour estimated noise variance: 0.2522")
print("But ADD_NOISE = False, so true noise variance ≈ 0")
print("\nThis means the residual error is NOT from noise, but from:")
print("  1. Model mismatch (turbulence changes the channel)")
print("  2. Projection errors (reference fields don't match turbulent fields)")
print("  3. Mode non-orthogonality")
print("\nThe 'noise variance' of 0.25 is actually the MODEL ERROR.")
print("This is HUGE - it means the linear model Y = H*P doesn't fit!")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("\nThe problem is NOT the channel estimation formula (it's correct).")
print("The problem is that the PROJECTIONS are wrong.")
print("\nWhen you project the turbulent field onto ideal reference modes:")
print("  projection = <E_rx_turbulent, ref_ideal>")
print("\nYou get small values because the fields don't match!")
print("\nThis causes:")
print("  1. Small Y_p values (received pilot projections)")
print("  2. H_est = Y_p @ pinv(P_p) has small values")
print("  3. Large residuals (model doesn't fit)")
print("  4. Large 'noise variance' estimate")
print("  5. MMSE over-regularizes")
print("  6. Equalization fails")

print("\n" + "="*80)
print("THE REAL FIX")
print("="*80)
print("\nOption 1: Accept that H is small and rescale")
print("  - The small H values might be CORRECT for your system")
print("  - The issue is that the overall power is lost in projection")
print("  - Solution: Normalize H so diagonal ≈ 1")
print("\nOption 2: Use better reference modes")
print("  - Propagate reference modes through turbulence")
print("  - Use adaptive basis (Karhunen-Loève)")
print("\nOption 3: Use different receiver architecture")
print("  - Intensity-based detection (no phase)")
print("  - Machine learning (CNN)")

print("\nLet's try Option 1 first...")
print("="*80)
