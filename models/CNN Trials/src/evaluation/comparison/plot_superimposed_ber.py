import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_superimposed():
    # Paths
    root_classical = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery/models/LDPC + Pilot + MMSE trials/cn2_sweep_results")
    root_convnext = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery/results/model saved results - npz or csv")
    
    # 1. Load Classical Data
    with open(root_classical / "cn2_sweep_data.json", 'r') as f:
        classical_data = json.load(f)
    
    cl_cn2 = np.array(classical_data['cn2_values'])
    cl_ber = np.array([d['ber'] for d in classical_data['data']['mmse']])
    
    # 2. Load ConvNeXt Data
    conv_df = pd.read_csv(root_convnext / "cumulative_results.csv")
    cnn_cn2 = conv_df['Cn2'].values
    cnn_ber = conv_df['BER'].values
    
    # REMOVE OUTLIER around 1e-14 (User Request)
    # FIX: Must set atol=0 because default is 1e-8, which makes all 1e-18 values look equal to 1e-14!
    mask = ~np.isclose(cnn_cn2, 1e-14, rtol=0.1, atol=0) 
    cnn_cn2 = cnn_cn2[mask]
    cnn_ber = cnn_ber[mask]
    
    # 3. Plotting
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'serif'
    
    # Classical Curve (Red)
    # Use semilogx for Linear Y axis, Log X axis
    plt.semilogx(cl_cn2, cl_ber, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='Classical MMSE Baseline')
    
    # ConvNeXt Curve (Blue)
    plt.semilogx(cnn_cn2, cnn_ber, '^-', color='#1f77b4', linewidth=2.5, markersize=8, label='ConvNeXt Tiny (Ours)')
    
    # Reference Lines
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random Guess (0.5)')
    
    # Annotations
    plt.xlabel('Turbulence Strength $C_n^2$ ($m^{-2/3}$)', fontsize=14, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER) - Linear Scale', fontsize=14, fontweight='bold')
    plt.title('Performance Comparison (Linear Scale): Classical vs Deep Learning', fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12, loc='upper left')
    
    plt.xlim(1e-18, 1e-12)
    plt.ylim(0.0, 0.6) # Linear Scale 0 to 0.6
    
    output_path = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery/models/CNN Trials/superimposed_ber_comparison_linear.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved superimposed plot to {output_path}")

if __name__ == "__main__":
    plot_superimposed()
