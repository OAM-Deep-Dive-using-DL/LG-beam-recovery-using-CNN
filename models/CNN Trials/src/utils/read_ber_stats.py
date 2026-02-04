import numpy as np
from pathlib import Path

# Paths based on the plot_comparison.py script
vanilla_path = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery/models/CNN Trials/outputs/logs/cnn_results.npz")
cbam_path = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery/models/CNN Trials/outputs/logs/cnn_results.npz") # Use the same one if CBAM is not separate or valid


# MMSE Data from the script
mmse_cn2 = np.array([1e-18, 5e-17, 1e-16, 2e-16, 5e-16, 1e-15, 2e-15, 5e-15, 1e-14, 1e-12])
mmse_ber = np.array([0.000, 0.000, 0.009, 0.040, 0.150, 0.280, 0.350, 0.450, 0.490, 0.510])

results = {}

if vanilla_path.exists():
    data = np.load(vanilla_path)
    results['vanilla'] = {'cn2': data['cn2'], 'ber': data['ber']}
    print("Loaded Vanilla ResNet results")

# Try to look for CBAM results - plot_comparison.py line 9 just said "cnn_results.npz"
# Let's search for it if we can't find it easily, but for now focus on vanilla if that's what "resne" refers to.
if cbam_path.exists():
    data = np.load(cbam_path)
    results['cbam'] = {'cn2': data['cn2'], 'ber': data['ber']}
    print("Loaded CBAM ResNet results")

# For now, let's assume "resnet" refers to the Vanilla ResNet logic or we can show both.
# We will inspect values at matching Cn2 points.

print("\nComparison Table (Classical MMSE vs ResNet):")
print(f"{'Cn2':<10} | {'MMSE BER':<10} | {'ResNet BER':<10} | {'Improvement (%)':<15}")
print("-" * 55)

if 'vanilla' in results:
    vanilla_cn2 = results['vanilla']['cn2']
    vanilla_ber = results['vanilla']['ber']
    
    improvements = []
    
    # Iterate through MMSE points and closest Resnet point
    for i, cn in enumerate(mmse_cn2):
        m_ber = mmse_ber[i]
        
        # Find closest match in ResNet data
        # Assuming log scale closeness or exact match
        # Let's just find the index of the closest cn2 value
        idx = (np.abs(vanilla_cn2 - cn)).argmin()
        r_cn = vanilla_cn2[idx]
        r_ber = vanilla_ber[idx]
        
        # Check if the cn2 values are reasonably close (within 10% tolerance?)
        if abs(np.log10(r_cn) - np.log10(cn)) < 0.1: # Comparing in log scale
             imp = ((m_ber - r_ber) / m_ber) * 100 if m_ber > 0 else 0
             print(f"{cn:.1e}    | {m_ber:.4f}     | {r_ber:.4f}     | {imp:.2f}%")
             
             # Collect stats for Moderate/Strong regimes (1e-16 to 1e-14)
             # We include 1e-16 and up to 1e-14
             if 1e-16 <= cn <= 1e-14:
                improvements.append(imp)
        else:
            # Maybe ResNet didn't evaluate at this point
            pass

    if improvements:
        avg_imp = sum(improvements) / len(improvements)
        print("-" * 55)
        print(f"Average Improvement (Moderate to Strong Turbulence [1e-16, 1e-14]): {avg_imp:.2f}%")

else:
    print("Could not load Vanilla ResNet results.")
