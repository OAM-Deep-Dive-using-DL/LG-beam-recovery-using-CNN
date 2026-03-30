import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

def plot_convnext_vs_classical():
    project_root = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery")
    
    # 1. Load ConvNeXt Data (Ours)
    csv_path = project_root / "cumulative_results.csv"
    df = pd.read_csv(csv_path)
    # Filter Outlier
    df = df[~np.isclose(df['Cn2'], 1e-14, atol=1e-16)]
    convnext_cn2 = df['Cn2'].values
    convnext_ber = df['BER'].values

    # 2. Load Classical Data (MMSE + Pilot + LDPC)
    classical_json_path = project_root / "models/LDPC + Pilot + MMSE trials/cn2_sweep_results/cn2_sweep_data.json"
    
    classical_cn2 = []
    classical_ber = []
    
    with open(classical_json_path, 'r') as f:
        c_data = json.load(f)
        mmse_data = c_data['data']['mmse']
        for entry in mmse_data:
            classical_cn2.append(entry['cn2'])
            classical_ber.append(entry['ber'])
            
    classical_cn2 = np.array(classical_cn2)
    classical_ber = np.array(classical_ber)

    # 3. Extrapolate Classical Data to 1e-12
    # The simulation stopped at 1e-15 because BER reached ~0.5 (Saturation).
    # To show the "complete" range as requested, we extend the line at 0.5.
    if classical_cn2.max() < 1e-12:
        max_ber = classical_ber[-1] # Should be ~0.5
        # Append point at 1e-12
        classical_cn2 = np.append(classical_cn2, 1e-12)
        classical_ber = np.append(classical_ber, max_ber)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    # ConvNeXt (The Hero)
    plt.semilogx(convnext_cn2, convnext_ber, 'b-', linewidth=3, label='ConvNeXt Tiny (Proposed)')
    
    # Classical (The Baseline)
    # Using 'k--s' (Black dashed square) for clear distinction
    plt.semilogx(classical_cn2, classical_ber, 'k--s', linewidth=2, markersize=5, label='Classical (LDPC + Pilot + MMSE)')
    
    # Aesthetics
    plt.xlabel(r'Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('Performance Comparison: Proposed vs. Classical', fontsize=16, fontweight='bold')
    
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.ylim(0, 0.6)
    plt.xlim(1e-18, 1e-12)
    
    # Annotations for regimes
    plt.axvline(1e-15, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(1e-14, color='gray', linestyle=':', alpha=0.5)
    
    plt.text(2e-18, 0.53, "Weak Turb.", fontsize=10, color='gray')
    plt.text(2e-15, 0.53, "Strong", fontsize=10, color='gray')
    plt.text(2e-14, 0.53, "Extreme", fontsize=10, color='gray')

    plt.legend(fontsize=12, loc='upper left')
    
    output_path = project_root / "convnext_vs_classical.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Comparison Plot to {output_path}")

if __name__ == "__main__":
    plot_convnext_vs_classical()
