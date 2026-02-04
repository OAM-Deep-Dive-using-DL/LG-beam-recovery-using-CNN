import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'physics'))

from model import FSOModel
from utils.dataset import FSODataset
from utils.utils import qpsk_demodulate, compute_ber


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    test_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_test.h5", 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load Model with proper path resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Initializing {args.backbone}...")
    model = FSOModel(n_modes=test_dataset.n_modes, backbone_name=args.backbone).to(device)
    
    # Try to find the best model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_name = f"best_model_{args.backbone}.pth"
        model_path = Path(__file__).parent / model_name
    
    # If not found, try searching in parent directory or generic names
    if not model_path.exists():
        # Check parent folder
        parent_path = Path(__file__).parent.parent / model_name
        if parent_path.exists():
            model_path = parent_path
        else:
             # Try generic "best_model.pth"
            generic_path = Path(__file__).parent / "best_model.pth"
            if generic_path.exists():
                model_path = generic_path
    
    if not model_path.exists():
        print(f"\nError: Model file not found: {model_path}")
        print("Available models:")
        for p in Path(__file__).parent.parent.glob("*.pth"):
            print(f"  {p.name}")
        # sys.exit(1) # Don't exit, just warn if we are debugging
    
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Detected checkpoint dictionary. Loading 'model_state_dict'...")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")
            print("Running with random weights (for testing pipeline).")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for imgs, syms, pwrs in tqdm(test_loader):
            imgs = imgs.to(device)
            
            # Predict
            pred_syms, pred_pwrs = model(imgs)
            
            # Move to CPU
            all_preds.append(pred_syms.cpu().numpy())
            all_targets.append(syms.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)   # [N, 8, 2]
    all_targets = np.concatenate(all_targets, axis=0) # [N, 8, 2]
    
    # Convert to complex
    preds_complex = all_preds[..., 0] + 1j * all_preds[..., 1]
    targets_complex = all_targets[..., 0] + 1j * all_targets[..., 1]
    
    # 2. Bit Error Rate (BER) - FIXED using proper QPSK demodulation
    # Convert complex symbols to bits using proper constellation mapping
    pred_bits = qpsk_demodulate(preds_complex, soft=False)  # [N, 8, 2]
    target_bits = qpsk_demodulate(targets_complex, soft=False)  # [N, 8, 2]
    
    # Flatten for BER calculation
    pred_bits_flat = pred_bits.reshape(-1)
    target_bits_flat = target_bits.reshape(-1)
    
    ber = compute_ber(target_bits_flat, pred_bits_flat)
    ser = np.mean(pred_bits != target_bits)  # SER at symbol level
    
    print(f"\n{'='*40}")
    print(f"Results on TEST set ({len(test_dataset)} samples)")
    print(f"{'='*40}")
    print(f"Overall SER: {ser:.4f}")
    print(f"Overall BER: {ber:.4f}")

    # 3. Breakdown by Cn2
    # We need to get cn2 values corresponding to the test set order
    # Since DataLoader with shuffle=False preserves order, we can just use test_dataset.cn2
    # But wait, if batch_size doesn't divide perfectly, or if we used shuffle (we didn't), 
    # it's safer to collect them in the loop or just access directly if we are sure.
    # We used shuffle=False.
    
    # all_cn2 = test_dataset.cn2 # AttributeError: 'FSODataset' object has no attribute 'cn2'
    # Fix: Read directly from H5 file
    with h5py.File(test_dataset.h5_path, 'r') as f:
        all_cn2 = f['cn2'][:]
    # Get unique Cn2 values (original approach - the issue was in the matching, not the uniqueness)
    unique_cn2 = np.unique(all_cn2)
    print(f"\nUnique Cn2 values: {len(unique_cn2)} points")
    print(f"Range: {unique_cn2[0]:.2e} to {unique_cn2[-1]:.2e}")
    
    print(f"\nBreakdown by Turbulence Strength (Cn2):")
    print(f"{'Cn2':<12} | {'BER':<12} | {'SER':<12} | {'Samples':<8}")
    print("-" * 54)
    
    ber_per_cn2 = []
    
    for val in unique_cn2:
        mask = (all_cn2 == val)
        
        # Filter predictions and targets for this Cn2
        subset_preds = preds_complex[mask]
        subset_targets = targets_complex[mask]
        
        # Calculate BER for this subset using proper demodulation
        if len(subset_preds) > 0:
            subset_pred_bits = qpsk_demodulate(subset_preds, soft=False)
            subset_target_bits = qpsk_demodulate(subset_targets, soft=False)
            
            subset_ber = compute_ber(subset_target_bits.flatten(), subset_pred_bits.flatten())
            subset_ser = np.mean(subset_pred_bits != subset_target_bits)
        else:
            subset_ber = 0.0
            subset_ser = 0.0
        
        count = np.sum(mask)
        
        print(f"{val:.2e} | {subset_ber:.2e} | {subset_ser:.2e} | {count:<8}")
        ber_per_cn2.append(subset_ber)

    # --- Added breakdown for Low vs High Turbulence ---
    THRESHOLD_CN2 = 5e-14
    
    print(f"\n{'-'*54}")
    print(f"Aggregated Performance by Turbulence Regime:")
    print(f"Low  (< {THRESHOLD_CN2:.2e}) vs High (>= {THRESHOLD_CN2:.2e})")
    print(f"{'-'*54}")
    print(f"{'Regime':<12} | {'BER':<12} | {'SER':<12} | {'Samples':<8}")
    print(f"{'-'*54}")
    
    low_mask = (all_cn2 < THRESHOLD_CN2)
    high_mask = (all_cn2 >= THRESHOLD_CN2)
    
    for name, mask in [("Low", low_mask), ("High", high_mask)]:
        count = np.sum(mask)
        if count > 0:
            regime_preds = preds_complex[mask]
            regime_targets = targets_complex[mask]
            
            regime_pred_bits = qpsk_demodulate(regime_preds, soft=False)
            regime_target_bits = qpsk_demodulate(regime_targets, soft=False)
            
            regime_ber = compute_ber(regime_target_bits.flatten(), regime_pred_bits.flatten())
            regime_ser = np.mean(regime_pred_bits != regime_target_bits)
            print(f"{name:<12} | {regime_ber:.2e} | {regime_ser:.2e} | {count:<8}")
        else:
             print(f"{name:<12} | {'N/A':<12} | {'N/A':<12} | {0:<8}")
    print(f"{'='*54}")

    # 4. Diagnosis Statistics
    print(f"\n{'='*40}")
    print(f"Diagnosis Statistics")
    print(f"{'='*40}")
    
    # Magnitude Check
    mean_mag_pred = np.mean(np.abs(preds_complex))
    mean_mag_true = np.mean(np.abs(targets_complex))
    print(f"Mean Magnitude (Pred): {mean_mag_pred:.4f} (Target: {mean_mag_true:.4f})")
    
    # Phase Check
    # Calculate phase difference: pred * conj(target)
    # If pred = target * exp(j*theta), then product is |target|^2 * exp(j*theta)
    phase_diff = np.angle(preds_complex * np.conj(targets_complex))
    mean_phase_bias = np.degrees(np.mean(phase_diff))
    phase_jitter = np.degrees(np.std(phase_diff))
    
    print(f"Mean Phase Bias:     {mean_phase_bias:.2f} degrees")
    print(f"Phase Jitter (Std):  {phase_jitter:.2f} degrees")
    
    if mean_mag_pred < 0.1:
        print(">> DIAGNOSIS: Model is outputting ZEROS (Confusion/Collapse).")
    elif abs(mean_phase_bias) > 10 and phase_jitter < 45:
        print(">> DIAGNOSIS: Systematic PHASE ROTATION. (Pilot ambiguity?)")
    elif phase_jitter > 60:
        print(">> DIAGNOSIS: Random Guessing / High Noise.")
    else:
        print(">> DIAGNOSIS: Mixed/Unknown errors.")

    # 5. Plot BER vs Cn2 (MAIN PLOT - Your Preferred Detailed Style)
    plt.figure(figsize=(12, 8))
    plt.semilogx(unique_cn2, ber_per_cn2, 'o-', linewidth=2, markersize=6, color='blue')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs Turbulence Strength - DETAILED TREND', fontsize=14, fontweight='bold')
    
    # Add reference lines and regions (your preferred enhancements)
    plt.axvline(5e-14, color='red', linestyle='--', alpha=0.7, linewidth=2, label='5e-14 Threshold')
    plt.axhline(0.01, color='orange', linestyle=':', alpha=0.7, linewidth=1.5, label='1% BER Target')
    plt.axhline(0.1, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='10% BER Target')
    plt.axvspan(1e-18, 5e-14, alpha=0.1, color='green', label='Low BER Region')
    plt.axvspan(5e-14, 1e-12, alpha=0.1, color='red', label='High BER Region')
    
    plt.legend(fontsize=10)
    plt.ylim(bottom=0, top=max(ber_per_cn2) * 1.1)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    plt.savefig(f"../evaluation_ber_curve_{args.backbone}.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved MAIN PLOT 'evaluation_ber_curve_{args.backbone}.png' (Your Preferred Detailed Style)")
    
    # Save Data for Comparison Plotting
    output_filename = f"../../cnn_results_{args.backbone}_{args.dataset_name}.npz"
    np.savez(output_filename, cn2=unique_cn2, ber=ber_per_cn2)
    print(f"Saved '{output_filename}'")
    
    # 5. Constellation Plot (Subset)
    plt.figure(figsize=(8, 8))
    # Plot a subset of points to avoid clutter
    subset = 2000
    flat_preds = preds_complex.flatten()[:subset]
    flat_targets = targets_complex.flatten()[:subset]
    
    plt.scatter(np.real(flat_targets), np.imag(flat_targets), c='red', marker='x', label='True', alpha=0.5)
    plt.scatter(np.real(flat_preds), np.imag(flat_preds), c='blue', marker='.', label='Pred', alpha=0.3)
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"Recovered Constellation (Overall BER={ber:.4f})")
    plt.savefig("../../evaluation_constellation.png")
    print("Saved 'evaluation_constellation.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FSO-OAM CNN Receiver")
    parser.add_argument('--data_dir', type=Path, default=Path('../../data/dataset'),
                       help='Path to dataset directory (default: ../../data/dataset)')
    parser.add_argument('--dataset_name', type=str, default='fso_oam_turbulence_v1',
                       help='Dataset name prefix (e.g., fso_oam_turbulence_v1)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help='Backbone architecture')
    parser.add_argument('--model_path', type=str, default=None, help='Explicit path to model file')
    args = parser.parse_args()
    
    # Validate dataset exists
    test_path = args.data_dir / f"{args.dataset_name}_test.h5"
    
    if not test_path.exists():
        print(f"\nError: Test dataset not found: {test_path}")
        print(f"\nGenerate dataset first using the canonical generator:")
        print(f"  cd ../../data/generators")
        print(f"  python generate_dataset.py --config configs/config.json --split test")
        print(f"\nOr specify custom dataset location with --data_dir flag.\n")
        import sys
        sys.exit(1)
    
    evaluate(args)
