import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

def plot_comparison():
    # Load New Results (CBAM)
    try:
        data_cbam = np.load("cnn_results.npz")
        cbam_cn2 = data_cbam['cn2']
        cbam_ber = data_cbam['ber']
    except FileNotFoundError:
        print("Error: cnn_results.npz (CBAM) not found.")
        return

    # Load Old Results (Vanilla)
    vanilla_path = Path("models/CNN Trials/outputs/final_results/cnn_results.npz")
    if vanilla_path.exists():
        data_vanilla = np.load(vanilla_path)
        vanilla_cn2 = data_vanilla['cn2']
        vanilla_ber = data_vanilla['ber']
    else:
        print("Warning: Vanilla results not found. Plotting CBAM only.")
        vanilla_cn2 = None
        vanilla_ber = None

    # Define MMSE Baseline Points (Observed from Logs)
    mmse_points_cn2 = np.array([1e-18, 5e-17, 1e-16, 2e-16, 5e-16, 1e-15, 2e-15, 5e-15, 1e-14, 1e-12])
    mmse_points_ber = np.array([0.000, 0.000, 0.009, 0.040, 0.150, 0.280, 0.350, 0.450, 0.490, 0.510])
    
    # Interpolate MMSE for smooth curve
    f_mmse = interp1d(np.log10(mmse_points_cn2), mmse_points_ber, kind='linear', fill_value="extrapolate")
    mmse_cn2_smooth = np.logspace(np.log10(1e-18), np.log10(1e-12), 100)
    mmse_ber_smooth = f_mmse(np.log10(mmse_cn2_smooth))
    mmse_ber_smooth = np.clip(mmse_ber_smooth, 0, 0.5)

    # Plot
    plt.figure(figsize=(10, 6))
    
    # 1. MMSE
    plt.semilogx(mmse_cn2_smooth, mmse_ber_smooth, 'k--', linewidth=2, label='Classical MMSE', alpha=0.6)
    
    # 2. Vanilla ResNet
    if vanilla_cn2 is not None:
        plt.semilogx(vanilla_cn2, vanilla_ber, 'b-o', linewidth=2, label='DL (ResNet-18)', markersize=6, alpha=0.7)
        
    # 3. CBAM ResNet
    plt.semilogx(cbam_cn2, cbam_ber, 'r-s', linewidth=3, label='DL (ResNet-18 + CBAM)', markersize=7)

    # Formatting
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('Performance Comparison: Architecture Evolution', fontsize=14)
    plt.legend(fontsize=11)
    
    # Annotations
    plt.axvline(1e-14, color='gray', linestyle=':', alpha=0.5)
    plt.text(1e-14, 0.52, 'Strong Turbulence\n(Deep Fade)', ha='center', va='bottom', fontsize=9, color='gray')

    # Regimes
    plt.axvspan(1e-18, 1e-16, color='green', alpha=0.05, label='_nolegend_')
    plt.text(3e-18, 0.45, 'Weak', color='green', alpha=0.6, fontweight='bold')
    
    plt.axvspan(1e-16, 1e-14, color='orange', alpha=0.05, label='_nolegend_')
    plt.text(3e-16, 0.45, 'Moderate', color='orange', alpha=0.6, fontweight='bold')
    
    plt.axvspan(1e-14, 1e-12, color='red', alpha=0.05, label='_nolegend_')
    plt.text(3e-14, 0.45, 'Strong', color='red', alpha=0.6, fontweight='bold')
    
    # FEC Limit
    plt.axhline(0.2, color='red', linestyle=':', linewidth=1)
    plt.text(1e-18, 0.21, 'Soft-Dedocing FEC Limit (~20%)', color='red', fontsize=8)

    plt.ylim(0, 0.55)
    plt.xlim(1e-18, 1e-12)
    
    plt.tight_layout()
    plt.savefig("comparison_architecture_plot.png", dpi=300)
    print("Saved 'comparison_architecture_plot.png'")

if __name__ == "__main__":
    plot_comparison()
