"""
Check if the issue is with how we're estimating the noise variance.
A large "noise" variance when ADD_NOISE=False suggests channel model mismatch.
"""

import numpy as np

# From your output:
noise_var_est = 0.2522
coded_ber = 0.4967

# Expected noise variance for noiseless case
expected_noise_var = 0.0  # Should be ~0 if no noise added!

print("="*80)
print("NOISE VARIANCE ANALYSIS")
print("="*80)
print(f"\nEstimated noise variance: {noise_var_est:.4f}")
print(f"Expected (ADD_NOISE=False): ~0.0")
print(f"\nDiscrepancy: {noise_var_est:.4f} (this is HUGE!)")

print("\nWhat does this mean?")
print("-" * 80)
print("The 'noise variance' is estimated from pilot residuals:")
print("  residual = Y_pilots - H_est @ P_pilots")
print("  noise_var = mean(|residual|²)")
print("\nIf noise_var is large when no noise was added, it means:")
print("  1. H_est doesn't fit the pilot observations well")
print("  2. The channel model Y = H*P is WRONG")
print("  3. There's unmodeled effects (turbulence mode mixing)")

print("\nThe problem:")
print("-" * 80)
print("Your reference fields assume:")
print("  - Free-space propagation (analytical LG modes)")
print("  - No turbulence distortion")
print("  - Perfect mode orthogonality")

print("\nBut the actual received field has:")
print("  - Turbulence-induced phase distortion")
print("  - Mode mixing (non-diagonal H)")
print("  - Aperture truncation effects")

print("\nThis mismatch causes:")
print("  1. Small projection values (H elements ~ 0.01-0.13 instead of ~1.0)")
print("  2. Large 'noise' variance (residual error from model mismatch)")
print("  3. MMSE over-regularizes (reg = σ² = 0.25 is huge!)")
print("  4. Equalized symbols are garbage")
print("  5. BER ≈ 50% (random guessing)")

print("\n" + "="*80)
print("ROOT CAUSE IDENTIFIED")
print("="*80)
print("\nThe projection formula:")
print("  symbol_est = <E_rx, ref> / <ref, ref>")
print("\nAssumes that E_rx can be decomposed as:")
print("  E_rx = sum(symbol_i * ref_i)")
print("\nBut after turbulence:")
print("  E_rx = turbulent_field (NOT a linear combination of ideal refs!)")
print("\nSo the projections give small, incorrect values.")

print("\n" + "="*80)
print("SOLUTION")
print("="*80)
print("\nOption 1: Use turbulent reference fields")
print("  - Propagate each reference mode through the SAME turbulence")
print("  - Use these turbulent refs for projection")
print("  - Problem: Need to know the turbulence realization")
print("\nOption 2: Use pilot-based blind estimation (CURRENT APPROACH)")
print("  - Don't assume anything about the channel")
print("  - Use pilots to estimate H directly")
print("  - Problem: H is estimated in the 'wrong' basis (ideal refs)")
print("\nOption 3: Normalize H_est properly")
print("  - The issue might be that H_est is in the wrong scale")
print("  - Check if we need to rescale H based on pilot power")

print("\nLet me check Option 3...")
print("="*80)

# From your output, let's analyze the power
print("\nPower analysis from your output:")
print("-" * 80)
print("E_rx power (in aperture): 0.956 W")
print("E_ref power (per mode): 0.120 W")
print("Expected projection (if perfect match): sqrt(0.956 * 0.120) ≈ 0.339")
print("\nBut H_est values are: 0.008 to 0.125")
print("This is 3-40x smaller than expected!")

print("\nHypothesis: The projection is dividing by the wrong normalization")
print("-" * 80)

# Let's trace through the math
print("\nProjection formula in receiver.py:")
print("  projection = sum(E_rx * conj(ref)) * dA")
print("  symbol_est = projection / ref_energy")
print("  where ref_energy = sum(|ref|²) * dA")

print("\nFor a single mode with symbol s:")
print("  E_tx = ref * s  (at TX)")
print("  E_rx = E_tx * attenuation * turbulence_distortion")
print("  E_rx ≈ ref * s * α * T  (where α=atten, T=turbulence)")

print("\nProjection:")
print("  proj = sum((ref * s * α * T) * conj(ref)) * dA")
print("       = s * α * sum(T * |ref|²) * dA")
print("       ≠ s * α * ref_energy  (because T ≠ 1)")

print("\nSo the projection is WRONG because turbulence T changes the field!")

print("\n" + "="*80)
print("ACTUAL SOLUTION")
print("="*80)
print("\nThe pilot-based channel estimation is the RIGHT approach.")
print("But we need to check:")
print("\n1. Are the pilots orthogonal enough?")
print("   - Use DFT pilots: P[i,j] = exp(2πij*k/N)")
print("   - This ensures P @ P^H is well-conditioned")

print("\n2. Is the LS estimation correct?")
print("   - H_est = Y_pilots @ pinv(P_pilots)")
print("   - Check if pinv is stable")

print("\n3. Is the noise variance estimate reasonable?")
print("   - noise_var = 0.25 is HUGE for a noiseless system")
print("   - This suggests H_est is wrong")

print("\n4. CRITICAL: Check the pilot power!")
print("   - If pilots have different power than data, H_est will be scaled wrong")
print("   - Pilots should have SAME power as data symbols")

print("\n" + "="*80)
print("ACTION ITEMS")
print("="*80)
print("\n[ ] Check pilot generation in encoding.py")
print("    - Verify pilots are unit-energy QPSK symbols")
print("    - Verify pilot pattern is orthogonal (DFT)")
print("\n[ ] Check channel estimation in receiver.py")
print("    - Print Y_pilots and P_pilots to verify")
print("    - Check if H_est = Y @ pinv(P) is computed correctly")
print("\n[ ] Check projection normalization")
print("    - Verify ref_energy calculation")
print("    - Check if scaling_factor is applied correctly")
print("\n[ ] Run with Cn²=0 (no turbulence) to verify baseline")
print("    - Should get BER ≈ 0 if everything is correct")

print("="*80)
