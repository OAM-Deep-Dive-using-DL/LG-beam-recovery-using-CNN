import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# Use relative path resolution
SCRIPT_DIR = Path(__file__).parent.parent.parent  # models/CNN Trials/

def plot_comparison():
    import json

    # 1. Load Real MMSE Results
    mmse_json_path = SCRIPT_DIR.parent.parent / "LDPC + Pilot + MMSE trials" / "cn2_sweep_results" / "cn2_sweep_data.json"
    if mmse_json_path.exists():
        with open(mmse_json_path, 'r') as f:
            mmse_data = json.load(f)
        
        # Extract MMSE data points
        mmse_raw_cn2 = []
        mmse_raw_ber = []
        for entry in mmse_data['data']['mmse']:
            mmse_raw_cn2.append(entry['cn2'])
            mmse_raw_ber.append(entry['ber'])
        
        mmse_points_cn2 = np.array(mmse_raw_cn2)
        mmse_points_ber = np.array(mmse_raw_ber)
        print(f"Loaded Real MMSE data from {mmse_json_path}")
    else:
        print(f"Warning: MMSE data not found at {mmse_json_path}. Using synthetic baseline.")
        # Fallback to synthetic points
        mmse_points_cn2 = np.array([1e-18, 5e-17, 1e-16, 2e-16, 5e-16, 1e-15, 2e-15, 5e-15, 1e-14, 1e-12])
        mmse_points_ber = np.array([0.000, 0.000, 0.009, 0.040, 0.150, 0.280, 0.350, 0.450, 0.490, 0.510])

    # 2. Load ResNet (Baseline DL)
    # Search for ResNet result file
    resnet_files = list(Path(__file__).parent.parent.parent.glob("**/cnn_results_resnet*.npz"))
    # Or generically cnn_results.npz if it exists and looks like the old one
    resnet_path = "cnn_results.npz" 
    
    if Path("cnn_results_resnet_cbam.npz").exists():
        resnet_path = "cnn_results_resnet_cbam.npz"
    elif len(resnet_files) > 0:
        resnet_path = resnet_files[0]
        
    resnet_cn2 = None
    resnet_ber = None
    
    if Path(resnet_path).exists():
        try:
            data_resnet = np.load(resnet_path)
            resnet_cn2 = data_resnet['cn2']
            resnet_ber = data_resnet['ber']
            print(f"Loaded ResNet baseline from {resnet_path}")
        except:
            print(f"Could not load ResNet data from {resnet_path}")
    
    # 3. Load ConvNeXt (Ours)
    convnext_path = "cnn_results_convnext_tiny.npz"
    convnext_cn2 = None
    convnext_ber = None
    
    if Path(convnext_path).exists():
        data_convnext = np.load(convnext_path)
        convnext_cn2 = data_convnext['cn2']
        convnext_ber = data_convnext['ber']
        print(f"Loaded ConvNeXt results from {convnext_path}")

    
    # Interpolate MMSE for smooth curve plotting
    f_mmse = interp1d(np.log10(mmse_points_cn2), mmse_points_ber, kind='linear', fill_value="extrapolate")
    mmse_cn2_smooth = np.logspace(np.log10(min(mmse_points_cn2)), np.log10(max(mmse_points_cn2)), 100)
    mmse_ber_smooth = f_mmse(np.log10(mmse_cn2_smooth))
    mmse_ber_smooth = np.clip(mmse_ber_smooth, 0, 0.5)

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot MMSE (Smooth + Points)
    plt.semilogx(mmse_cn2_smooth, mmse_ber_smooth, 'k--', linewidth=2, label='Classical MMSE', alpha=0.6)
    plt.semilogx(mmse_points_cn2, mmse_points_ber, 'kx', markersize=5, alpha=0.4) # Raw points
    
    # Plot ResNet
    if resnet_cn2 is not None:
         plt.semilogx(resnet_cn2, resnet_ber, 'b-o', linewidth=2, label='ResNet + CBAM', markersize=6, alpha=0.7)
        
    # Plot ConvNeXt
    if convnext_cn2 is not None:
        plt.semilogx(convnext_cn2, convnext_ber, 'r-s', linewidth=3, label='ConvNeXt Tiny (Ours)', markersize=7)

    # Formatting
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('Performance Comparison: MMSE vs Deep Learning', fontsize=14)
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
    plt.text(1e-18, 0.21, 'Soft-Decoding FEC Limit (~20%)', color='red', fontsize=8)

    plt.ylim(0, 0.55)
    plt.xlim(1e-18, 1e-12)
    
    plt.tight_layout()
    plt.savefig("comparison_architecture_plot.png", dpi=300)
    print("Saved 'comparison_architecture_plot.png'")

if __name__ == "__main__":
    plot_comparison()
