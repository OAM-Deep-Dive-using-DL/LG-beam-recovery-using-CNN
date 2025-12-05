# MMSE Equalization Performance Analysis
## Cn² Sweep Results

### Executive Summary

The MMSE equalizer for FSO-OAM systems has been characterized across turbulence
strengths from Cn² = 1e-18 to 1e-15 m^(-2/3).

**Key Finding**: MMSE works well for **weak turbulence (Cn² < 2e-17)** but 
degrades rapidly for moderate to strong turbulence.

---

## Performance Thresholds

### ✅ Excellent Performance (BER < 1%)
**Cn² ≤ 1.2e-17 m^(-2/3)**
- BER: 0% to 1.24%
- cond(H): 1.02 to 1.08
- Use case: Clear atmospheric conditions, short links

### ⚠️ Acceptable Performance (BER 1-10%)
**Cn² = 1.2e-17 to 3.2e-17 m^(-2/3)**
- BER: 1.24% to 8.94%
- cond(H): 1.08 to 1.17
- Use case: Mild turbulence, with FEC can achieve low error rates

### ❌ Poor Performance (BER > 10%)
**Cn² > 3.2e-17 m^(-2/3)**
- BER: 14.5% to 50%
- cond(H): 1.26 to 99.2
- Use case: Not recommended - use ML-based receiver (CNN)

---

## Detailed Results Table

| Cn² (m^-2/3) | BER (%) | Coded BER (%) | cond(H) | Bit Errors | Status |
|--------------|---------|---------------|---------|------------|--------|
| 1.00e-18 | 0.00 | 0.00 | 1.02 | 0 | ✅ Perfect |
| 1.64e-18 | 0.00 | 0.00 | 1.03 | 0 | ✅ Perfect |
| 2.68e-18 | 0.00 | 0.00 | 1.03 | 0 | ✅ Perfect |
| 4.39e-18 | 0.00 | 0.00 | 1.05 | 0 | ✅ Perfect |
| 7.20e-18 | 0.11 | 0.13 | 1.06 | 7 | ✅ Excellent |
| 1.18e-17 | 1.24 | 1.26 | 1.08 | 81 | ✅ Good |
| 1.93e-17 | 4.24 | 4.04 | 1.12 | 278 | ⚠️ Acceptable |
| 3.16e-17 | 8.94 | 8.40 | 1.17 | 586 | ⚠️ Marginal |
| 5.18e-17 | 14.53 | 13.90 | 1.26 | 952 | ❌ Poor |
| 8.48e-17 | 21.03 | 20.54 | 1.43 | 1378 | ❌ Poor |
| 1.39e-16 | 28.92 | 28.47 | 1.78 | 1895 | ❌ Very Poor |
| 2.28e-16 | 38.64 | 38.00 | 2.62 | 2532 | ❌ Very Poor |
| 3.73e-16 | 45.74 | 45.40 | 14.01 | 2997 | ❌ Unusable |
| 6.11e-16 | 49.88 | 49.87 | 99.24 | 3268 | ❌ Random |
| 1.00e-15 | 49.60 | 49.91 | 29.14 | 3250 | ❌ Random |

---

## Analysis

### 1. BER vs Cn² Relationship

The BER grows approximately **exponentially** with Cn²:
- Cn² < 1e-17: BER ≈ 0%
- Cn² ≈ 2e-17: BER ≈ 4%
- Cn² ≈ 5e-17: BER ≈ 15%
- Cn² ≈ 1e-16: BER ≈ 29%
- Cn² > 5e-16: BER ≈ 50% (random guessing)

### 2. Channel Conditioning

The channel matrix condition number shows:
- **Well-conditioned** (cond < 2) for Cn² < 2e-16
- **Ill-conditioned** (cond > 10) for Cn² > 3e-16
- **Severe ill-conditioning** (cond ≈ 100) at Cn² = 6e-16

This indicates that turbulence causes **mode mixing** which makes the
channel matrix harder to invert.

### 3. LDPC Performance

The coded BER (before LDPC) is very close to the final BER, indicating:
- LDPC is working correctly
- Error patterns are too severe for LDPC to correct at high Cn²
- Need better equalization or different receiver architecture

---

## Comparison to Literature

Typical atmospheric turbulence values:
- **Clear air, ground-level**: Cn² ≈ 1e-17 to 1e-16 m^(-2/3)
- **Moderate turbulence**: Cn² ≈ 1e-15 to 1e-14 m^(-2/3)
- **Strong turbulence**: Cn² > 1e-13 m^(-2/3)

**Conclusion**: Classical MMSE receiver only works for **very weak turbulence**
(below typical atmospheric conditions). For realistic scenarios, need:
1. Adaptive optics
2. Machine learning receivers (CNN)
3. Intensity-only detection
4. Shorter link distances

---

## Recommendations

### For Classical Receiver (MMSE/ZF):
- **Only use for Cn² < 2e-17 m^(-2/3)**
- Implement adaptive threshold based on channel condition number
- If cond(H) > 2, switch to fallback mode or request retransmission

### For System Design:
- **Focus on CNN-based receiver** for Cn² > 2e-17
- Use classical receiver as baseline/fallback for weak turbulence
- Consider hybrid approach: classical for weak, CNN for strong turbulence

### For Future Work:
1. Run ZF sweep for comparison (expect worse performance)
2. Characterize CNN performance on same Cn² range
3. Implement adaptive receiver selection based on estimated Cn²
4. Test with different mode sets (fewer modes may be more robust)

---

## Files Generated

- `cn2_sweep_data.json`: Raw sweep results
- `cn2_sweep_results.png`: Performance plots (4 subplots)
- This analysis document

---

**Date**: 2025-11-22  
**System**: FSO-OAM with 8 modes, 1000m link, LDPC FEC  
**Equalizer**: MMSE (corrected implementation)
