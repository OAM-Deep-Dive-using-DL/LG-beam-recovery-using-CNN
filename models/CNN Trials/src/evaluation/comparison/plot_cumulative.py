import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cumulative():
    # Define file paths (relative to project root, which is ../../ from here)
    # Actually, we run this from models/CNN Trials usually, so project root is ../../
    # But let's use absolute paths or relative to script to be safe.
    
    # We will assume we run this from models/CNN Trials
    # The npz files are in ../../
    
    files = [
        "../../cnn_results_convnext_tiny_curriculum_lvl1_ideal.npz",
        "../../cnn_results_convnext_tiny_curriculum_lvl2_weak.npz",
        "../../cnn_results_convnext_tiny_curriculum_lvl3_moderate.npz",
        "../../cnn_results_convnext_tiny_curriculum_lvl4_strong.npz",
        "../../cnn_results_convnext_tiny_curriculum_lvl5_extreme.npz"
    ]
    
    labels = [
        "Lvl 1: Ideal",
        "Lvl 2: Weak",
        "Lvl 3: Moderate",
        "Lvl 4: Strong",
        "Lvl 5: Extreme"
    ]
    
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    
    all_cn2 = []
    all_ber = []
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual segments
    for fpath, label, color in zip(files, labels, colors):
        try:
            data = np.load(fpath)
            cn2 = data['cn2']
            ber = data['ber']
            
            # Scatter plot for the existing points
            plt.loglog(cn2, ber, 'o', markersize=6, label=label, color=color, alpha=0.7)
            
            all_cn2.append(cn2)
            all_ber.append(ber)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            return

    # Combine and Sort for the Line
    full_cn2 = np.concatenate(all_cn2)
    full_ber = np.concatenate(all_ber)
    
    sort_idx = np.argsort(full_cn2)
    full_cn2 = full_cn2[sort_idx]
    full_ber = full_ber[sort_idx]
    
    # Plot the connecting line
    plt.loglog(full_cn2, full_ber, '-', linewidth=2, color='black', alpha=0.5, label='Overall Trend')
    
    # Add Reference Lines/Regions from evaluate.py
    plt.axvline(5e-14, color='red', linestyle='--', alpha=0.7, linewidth=2, label='5e-14 Threshold')
    plt.axhline(0.01, color='orange', linestyle=':', alpha=0.7, linewidth=1.5, label='1% BER Target')
    plt.axhline(0.1, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='10% BER Target')
    
    plt.axvspan(1e-18, 5e-14, alpha=0.1, color='green', label='Low BER Region')
    plt.axvspan(5e-14, 1e-12, alpha=0.1, color='red', label='High BER Region')
    
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('BER vs Turbulence Strength - UNIFIED CURRICULUM (Levels 1-5)', fontsize=16, fontweight='bold')
    
    # Unified Legend
    plt.legend(fontsize=10, loc='best')
    
    # Scale Y axis as requested (Unified Scale)
    plt.ylim(bottom=1e-4, top=1.0)
    plt.xlim(1e-18, 1e-12)
    
    # Save
    output_path = "../../cumulative_ber_vs_turbulence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Unified Plot to {output_path}")

if __name__ == "__main__":
    plot_cumulative()
