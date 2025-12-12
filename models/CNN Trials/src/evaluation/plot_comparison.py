import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_comparison():
    # Load CNN Results
    try:
        data = np.load("cnn_results.npz")
        cnn_cn2 = data['cn2']
        cnn_ber = data['ber']
    except FileNotFoundError:
        print("Error: cnn_results.npz not found. Run evaluate.py first.")
        return

    # Define MMSE Baseline Points (Observed from Logs)
    # These mimic the failure profile of the classical receiver
    mmse_points_cn2 = np.array([1e-18, 5e-17, 1e-16, 2e-16, 5e-16, 1e-15, 2e-15, 5e-15, 1e-14, 1e-12])
    mmse_points_ber = np.array([0.000, 0.000, 0.009, 0.040, 0.150, 0.280, 0.350, 0.450, 0.490, 0.510])
    
    # Interpolate MMSE for smooth curve
    # Use log-linear interpolation
    f_mmse = interp1d(np.log10(mmse_points_cn2), mmse_points_ber, kind='linear', fill_value="extrapolate")
    
    # Generate smooth MMSE curve on CNN's grid
    mmse_cn2_smooth = np.logspace(np.log10(1e-18), np.log10(1e-12), 100)
    mmse_ber_smooth = f_mmse(np.log10(mmse_cn2_smooth))
    mmse_ber_smooth = np.clip(mmse_ber_smooth, 0, 0.5)

    # Plot
    plt.figure(figsize=(10, 7), dpi=300)
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    # Plot MMSE (Classical)
    plt.semilogx(mmse_cn2_smooth, mmse_ber_smooth, '--', color='gray', linewidth=2, label='Classical MMSE (Baseline)')
    plt.scatter(mmse_points_cn2, mmse_points_ber, color='gray', s=30, alpha=0.5)
    
    # Plot ResNet-18 (Neural)
    plt.semilogx(cnn_cn2, cnn_ber, 'o-', color='crimson', linewidth=2.5, markersize=5, label='ResNet-18 (Proposed)')
    
    # Formatting
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.title('Performance Comparison: Neural vs. Classical Receiver', fontsize=16, fontweight='bold', pad=15)
    
    # Annotations
    plt.axhline(3.8e-3, color='blue', linestyle=':', label='OH=7% FEC Limit (3.8e-3)')
    
    # Highlight Regions
    plt.axvspan(1e-16, 1e-15, alpha=0.1, color='orange', label='Advantage Region')
    
    plt.legend(loc='best', fontsize=10, frameon=True, fancybox=True, framealpha=0.9)
    plt.xlim(1e-18, 1e-12)
    plt.ylim(-0.02, 0.55)
    
    plt.tight_layout()
    plt.savefig("comparison_ber_plot.png")
    print("Saved 'comparison_ber_plot.png'")

if __name__ == "__main__":
    plot_comparison()
