import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cumulative_linear():
    # Define file paths (absolute paths to ROOT where we found them)
    project_root = Path("/Users/srivatsadavuluri/Developer/FSO beam recovery")
    
    files = [
        project_root / "cnn_results_convnext_tiny_curriculum_lvl1_ideal.npz",
        project_root / "cnn_results_convnext_tiny_curriculum_lvl2_weak.npz",
        project_root / "cnn_results_convnext_tiny_curriculum_lvl3_moderate.npz",
        project_root / "cnn_results_convnext_tiny_curriculum_lvl4_strong.npz",
        project_root / "cnn_results_convnext_tiny_curriculum_lvl5_extreme.npz"
    ]
    
    labels = [
        "Lvl 1: Ideal",
        "Lvl 2: Weak",
        "Lvl 3: Moderate",
        "Lvl 4: Strong",
        "Lvl 5: Extreme"
    ]
    
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    
    plt.figure(figsize=(10, 6))
    
    # Store all data to sort and plot a single continuous line
    all_cn2 = []
    all_ber = []
    
    # Plot individual segments properties
    for fpath, label, color in zip(files, labels, colors):
        try:
            data = np.load(fpath)
            cn2 = data['cn2']
            ber = data['ber']
            
            # Scatter for points
            # plt.semilogx(cn2, ber, 'o', markersize=4, label=label, color=color, alpha=0.6)
            
            # Outlier Cleaning (Level 5 Start Artifact)
            if "extreme" in str(fpath) and len(cn2) > 0:
                 # Remove the first point if it's 1e-14 (known single-sample outlier)
                 if np.isclose(cn2[0], 1e-14, atol=1e-16):
                     print(f"Removing outlier from {fpath}: {cn2[0]} (BER={ber[0]})")
                     cn2 = cn2[1:]
                     ber = ber[1:]
            
            all_cn2.append(cn2)
            all_ber.append(ber)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    if not all_cn2:
        print("No data loaded.")
        return

    # Combine and Sort for the Line
    full_cn2 = np.concatenate(all_cn2)
    full_ber = np.concatenate(all_ber)
    
    sort_idx = np.argsort(full_cn2)
    full_cn2 = full_cn2[sort_idx]
    full_ber = full_ber[sort_idx]
    
    # Plot the Single Continuous Line (Linear Y, Log X)
    plt.semilogx(full_cn2, full_ber, '-', linewidth=2.5, color='darkblue', label='ConvNeXt Tiny (Ours)')
    
    # Labels and Title
    plt.xlabel(r'Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('BER vs Turbulence Strength (Linear Scale)', fontsize=16, fontweight='bold')
    
    # Grid
    plt.grid(True, which="both", ls="-", alpha=0.4)
    
    # Set Y-Axis Limits (Linear from 0 to max observed + margin)
    plt.ylim(0, 0.6)
    
    # Add ResNet-like formatting
    plt.axvline(5e-14, color='red', linestyle='--', alpha=0.5, label='Regime Threshold')
    
    plt.legend(fontsize=12, loc='upper left')
    
    # Save
    output_path = project_root / "cumulative_ber_linear.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Linear Plot to {output_path}")

if __name__ == "__main__":
    plot_cumulative_linear()
