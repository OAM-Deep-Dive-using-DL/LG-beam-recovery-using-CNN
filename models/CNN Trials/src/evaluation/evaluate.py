import torch
from torch.utils.data import DataLoader, Subset
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


def unwrap_dataset(dataset):
    """Return the underlying FSODataset if wrapped by torch.utils.data.Subset."""
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not mps_available:
            raise RuntimeError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option: {device_arg}")


def get_runtime_config(device: torch.device, workers: int) -> dict:
    pin_memory = device.type == "cuda"
    persistent_workers = workers > 0
    return {
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "non_blocking": pin_memory,
    }


def ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "savefig.dpi": 600,
    })


def save_dual(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)


def plot_ber_curve(unique_cn2, ber_per_cn2, output_path: Path):
    ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.semilogy(unique_cn2, ber_per_cn2, marker='o', color='#1f77b4', label='CNN')
    ax.set_xscale('log')
    ax.set_xlabel(r"$C_n^2$ [$m^{-2/3}$]")
    ax.set_ylabel("BER")
    ax.grid(True, which='major', alpha=0.25)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax.set_ylim(1e-5, 1.0)
    save_dual(fig, output_path)
    plt.close(fig)


def plot_constellation(preds_complex, targets_complex, output_path: Path, max_points=2000):
    ieee_style()
    fig, ax = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
    flat_preds = preds_complex.flatten()[:max_points]
    flat_targets = targets_complex.flatten()[:max_points]
    ax.scatter(np.real(flat_targets), np.imag(flat_targets), c='#d62728', marker='x', alpha=0.45, s=10, label='Target')
    ax.scatter(np.real(flat_preds), np.imag(flat_preds), c='#1f77b4', marker='o', alpha=0.30, s=8, label='Predicted')
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.6)
    ax.axvline(0, color='black', linewidth=0.6, alpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    save_dual(fig, output_path)
    plt.close(fig)


def evaluate(args):
    device = resolve_device(args.device)
    runtime = get_runtime_config(device, args.workers)
    print(f"Using device: {device}")
    print(
        "Runtime config: "
        f"batch_size={args.batch_size}, workers={args.workers}, "
        f"pin_memory={runtime['pin_memory']}, "
        f"persistent_workers={runtime['persistent_workers']}"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    test_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_test.h5", 'test', normalize_mode='none')
    if args.max_test_samples is not None:
        max_test = min(args.max_test_samples, len(test_dataset))
        test_dataset = Subset(test_dataset, range(max_test))
        print(f"Test subset enabled: {max_test} samples.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=runtime["pin_memory"],
        persistent_workers=runtime["persistent_workers"],
    )
    test_dataset_base = unwrap_dataset(test_dataset)
    
    # Load Model with proper path resolution
    print(f"Initializing {args.backbone}...")
    model = FSOModel(n_modes=test_dataset_base.n_modes, backbone_name=args.backbone).to(device)
    
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
        sys.exit(1)
    
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
            print(f"Error: Failed to load weights: {e}")
            sys.exit(1)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for imgs, syms, _pwrs in tqdm(test_loader):
            imgs = imgs.to(device, non_blocking=runtime["non_blocking"])
            
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
    ser = np.mean(np.any(pred_bits != target_bits, axis=-1))
    
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
    
    if isinstance(test_dataset, Subset):
        subset_indices = np.asarray(test_dataset.indices)
        all_cn2 = np.asarray(test_dataset_base.cn2)[subset_indices]
    else:
        all_cn2 = test_dataset_base.cn2
    # Binning Logic for Continuous Cn2
    num_bins = 20
    min_cn2 = np.min(all_cn2)
    max_cn2 = np.max(all_cn2)
    
    # helper to avoid log(0)
    def safelog10(x):
        return np.log10(np.maximum(x, 1e-20))

    if min_cn2 == max_cn2 or len(np.unique(all_cn2)) < num_bins:
        # Discrete case or too few points: Use unique values directly
        unique_bins = np.unique(all_cn2)
        print(f"\nUnique Cn2 values: {len(unique_bins)} points (Discrete/Few)")
        binned_cn2 = unique_bins
        
        # We still need to populate ber_per_cn2 for the loop below
        # To reuse logic, we can just treat these as the "bins"
        mapping_indices = np.searchsorted(unique_bins, all_cn2)
        unique_indices = np.unique(mapping_indices)
    else:
        # Continuous case: Use Logarithmic Binning
        print(f"\nDetected Continuous Cn2. Binning into {num_bins} logarithmic regions.")
        # Create bins
        bins = np.logspace(safelog10(min_cn2), safelog10(max_cn2), num_bins + 1)
        # Assign samples to bins (indices 1..num_bins)
        digitized = np.digitize(all_cn2, bins)
        
        # We want to iterate over the bins that actually have data
        unique_indices = np.unique(digitized)
        
    print(f"Range: {min_cn2:.2e} to {max_cn2:.2e}")
    
    print(f"\nBreakdown by Turbulence Strength (Binned):")
    print(f"{'Avg Cn2':<12} | {'BER':<12} | {'SER':<12} | {'Samples':<8}")
    print("-" * 54)
    
    ber_per_cn2 = []
    plot_cn2 = []
    
    # If binning, we iterate indices. If unique, we iterate unique values.
    # Let's standardize on using the mask.
    
    if min_cn2 == max_cn2 or len(np.unique(all_cn2)) < num_bins:
         iterator = unique_bins
         is_binned = False
    else:
         iterator = unique_indices
         is_binned = True

    for i in iterator:
        if is_binned:
            # i is the bin index from digitize
            mask = (digitized == i)
        else:
            # i is the actual value
            mask = (all_cn2 == i)
            
        count = np.sum(mask)
        if count == 0: continue
        
        # Calculate real average Cn2 for this bin (better than bin center)
        avg_cn2_val = np.mean(all_cn2[mask])
        
        # Filter predictions and targets
        subset_preds = preds_complex[mask]
        subset_targets = targets_complex[mask]
        
        # Calculate BER
        if len(subset_preds) > 0:
            subset_pred_bits = qpsk_demodulate(subset_preds, soft=False)
            subset_target_bits = qpsk_demodulate(subset_targets, soft=False)
            
            subset_ber = compute_ber(subset_target_bits.flatten(), subset_pred_bits.flatten())
            subset_ser = np.mean(np.any(subset_pred_bits != subset_target_bits, axis=-1))
        else:
            subset_ber = 0.0
            subset_ser = 0.0
            
        print(f"{avg_cn2_val:.2e} | {subset_ber:.2e} | {subset_ser:.2e} | {count:<8}")
        
        ber_per_cn2.append(subset_ber)
        plot_cn2.append(avg_cn2_val)

    # Update variables for plotting
    unique_cn2 = np.array(plot_cn2)
    ber_per_cn2 = np.array(ber_per_cn2)

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
            regime_ser = np.mean(np.any(regime_pred_bits != regime_target_bits, axis=-1))
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

    ber_plot_path = output_dir / f"cnn_cn2_vs_ber_{args.backbone}_{args.dataset_name}.png"
    plot_ber_curve(unique_cn2, ber_per_cn2, ber_plot_path)
    print(f"\nSaved BER plot to {ber_plot_path}")
    
    # Save Data for Comparison Plotting
    output_filename = output_dir / f"cnn_results_{args.backbone}_{args.dataset_name}.npz"
    np.savez(output_filename, cn2=unique_cn2, ber=ber_per_cn2)
    print(f"Saved '{output_filename}'")
    
    constellation_path = output_dir / f"cnn_constellation_{args.backbone}_{args.dataset_name}.png"
    plot_constellation(preds_complex, targets_complex, constellation_path)
    print(f"Saved constellation plot to {constellation_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FSO-OAM CNN Receiver")
    parser.add_argument('--data_dir', type=Path, default=Path('../../data/generated_curriculum'),
                       help='Path to dataset directory (default: ../../data/generated_curriculum)')
    parser.add_argument('--dataset_name', type=str, default='fso_oam_turbulence_v1',
                       help='Dataset name prefix (e.g., fso_oam_turbulence_v1)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device selection (default: auto = cuda > mps > cpu)')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help='Backbone architecture')
    parser.add_argument('--model_path', type=str, default=None, help='Explicit path to model file')
    parser.add_argument('--output_dir', type=Path, default=Path('../outputs/evaluation'),
                       help='Directory for evaluation plots and result arrays')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='Optional cap on test samples for smoke tests')
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
