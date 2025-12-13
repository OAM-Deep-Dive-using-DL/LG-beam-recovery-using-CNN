import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from model import MultiHeadResNet
from utils.dataset import FSODataset

def calculate_throughput(ber, n_modes=8, symbol_rate_ghz=1.0, ldpc_rate=0.8135, pilot_overhead=0.1):
    """
    Calculate effective throughput accounting for System Overhead.
    
    CORRECTION (Post-Audit):
    The neural network is trained on the full physical layer frames, which include:
    1. LDPC Encoding (Rate 0.8135)
    2. Pilot Symbols (10%)
    
    Therefore, the network acts as a "Neural Equalizer" replacing the MMSE block.
    It produces coded symbols which must still be LDPC decoded.
    Thus, the throughput ceiling is the same as the classical system (11.7 Gbps),
    NOT 16 Gbps. The advantage is RESILIENCE (Link Availability), not Peak Rate.
    
    Args:
        ber: Coded Bit Error Rate (Raw BER before LDPC)
        n_modes: Number of spatial modes (default: 8)
        symbol_rate_ghz: Symbol rate in GSymbol/s (default: 1.0)
    
    Returns:
        throughput_gbps: Effective throughput in Gbps
    """
    bits_per_symbol = 2  # QPSK
    
    # 1. Raw Line Rate
    raw_line_rate = n_modes * bits_per_symbol * symbol_rate_ghz # 16 Gbps
    
    # 2. Account for Pilots (10%)
    # Network outputs pilots, but they carry no data
    data_symbol_rate = raw_line_rate * (1 - pilot_overhead) # 14.4 Gbps
    
    # 3. Account for LDPC (Rate 0.8135)
    # Network outputs coded bits, we need info bits
    info_bit_rate = data_symbol_rate * ldpc_rate # 11.7 Gbps
    
    # LDPC Threshold (Soft Decision)
    fec_threshold = 0.038  # 3.8% Raw BER
    
    if ber < fec_threshold:
        # LDPC corrects errors -> Full Info Rate
        throughput = info_bit_rate
    elif ber < 0.15:
        # Partial degradation
        degradation = (ber - fec_threshold) / (0.15 - fec_threshold)
        throughput = info_bit_rate * (1 - 0.7 * degradation)
    else:
        # Link Failure
        throughput = 0.0
    
    return throughput

def qpsk_demod(symbols_complex):
    """
    Demodulate complex symbols to bits (QPSK).
    Constellation: (1+j, -1+j, -1-j, 1-j) / sqrt(2)
    Bits: 00, 01, 11, 10 (Gray coding) or just quadrant mapping.
    
    Let's use simple quadrant mapping:
    Re > 0, Im > 0 -> 00
    Re < 0, Im > 0 -> 01
    Re < 0, Im < 0 -> 11
    Re > 0, Im < 0 -> 10
    """
    # Simply check signs
    bits_re = (np.real(symbols_complex) < 0).astype(int)
    bits_im = (np.imag(symbols_complex) < 0).astype(int)
    return bits_re, bits_im

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    test_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_test.h5", 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load Model
    model = MultiHeadResNet(n_modes=test_dataset.n_modes, backbone_name=args.backbone).to(device)
    model_path = f"best_model_{args.backbone}.pth"
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    
    # 1. Symbol Error Rate (SER)
    # Hard decision
    # QPSK: 4 quadrants. If predicted quadrant != target quadrant => Error
    
    # Target quadrants
    t_re_sign = np.sign(np.real(targets_complex))
    t_im_sign = np.sign(np.imag(targets_complex))
    
    # Pred quadrants
    p_re_sign = np.sign(np.real(preds_complex))
    p_im_sign = np.sign(np.imag(preds_complex))
    
    # Errors
    errors = (t_re_sign != p_re_sign) | (t_im_sign != p_im_sign)
    ser = np.mean(errors)
    
    # 2. Bit Error Rate (BER)
    # Each symbol is 2 bits.
    bit_errors = (t_re_sign != p_re_sign).astype(int) + (t_im_sign != p_im_sign).astype(int)
    total_bits = all_targets.size * 2
    ber = np.sum(bit_errors) / total_bits
    
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
    
    all_cn2 = test_dataset.cn2
    unique_cn2 = np.unique(all_cn2)
    
    print(f"\nBreakdown by Turbulence Strength (Cn2):")
    print(f"{'Cn2':<12} | {'BER':<8} | {'SER':<8} | {'Samples':<8}")
    print("-" * 46)
    
    ber_per_cn2 = []
    throughput_per_cn2 = []
    
    for val in unique_cn2:
        mask = (all_cn2 == val)
        
        # Filter errors for this Cn2
        # bit_errors is [N, 8] (sum of re+im errors per symbol) -> No, it's [N, 8] ints (0, 1, or 2)
        # Wait, bit_errors calculation above:
        # bit_errors = (t_re_sign != p_re_sign).astype(int) + (t_im_sign != p_im_sign).astype(int)
        # Shape is [N, 8]
        
        subset_bit_errors = bit_errors[mask]
        subset_total_bits = subset_bit_errors.size * 2 # 2 bits per symbol
        # Wait, subset_bit_errors elements are 0, 1, or 2.
        # So sum(subset_bit_errors) is total bit errors.
        # Total bits is number of symbols * 2.
        
        subset_ber = np.sum(subset_bit_errors) / (subset_bit_errors.size * 2)
        
        # SER
        # errors is [N, 8] boolean
        subset_errors = errors[mask]
        subset_ser = np.mean(subset_errors)
        
        count = np.sum(mask)
        
        # Calculate throughput for this Cn2
        throughput_gbps = calculate_throughput(subset_ber)
        
        print(f"{val:.2e}     | {subset_ber:.4f}   | {subset_ser:.4f}   | {count:<8}")
        ber_per_cn2.append(subset_ber)
        throughput_per_cn2.append(throughput_gbps)

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

    # 5. Plot BER vs Cn2
    plt.figure(figsize=(10, 6))
    plt.semilogx(unique_cn2, ber_per_cn2, 'o-', linewidth=2)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel('Turbulence Strength ($C_n^2$)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs. Turbulence Strength')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.savefig("evaluation_ber_curve.png")
    print("\nSaved 'evaluation_ber_curve.png'")
    
    # 6. Plot Throughput vs Cn2
    plt.figure(figsize=(10, 6))
    plt.semilogx(unique_cn2, throughput_per_cn2, 's-', linewidth=2, color='green', markersize=6)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=12)
    plt.ylabel('Throughput (Gbps)', fontsize=12)
    plt.title('Effective Throughput vs. Turbulence Strength', fontsize=14)
    
    # Add reference line for max throughput
    max_throughput = calculate_throughput(0.0)  # BER = 0
    plt.axhline(max_throughput, color='blue', linestyle='--', linewidth=1, alpha=0.5, label=f'Max Rate ({max_throughput:.1f} Gbps)')
    
    # Add FEC threshold marker
    plt.axhline(max_throughput * 0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Degraded (50%)')
    plt.axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Link Failure')
    
    plt.legend(fontsize=10)
    plt.ylim(bottom=-0.5, top=max_throughput * 1.1)
    plt.savefig("evaluation_throughput_curve.png", dpi=300)
    print("Saved 'evaluation_throughput_curve.png'")
    
    # 7. Combined BER + Throughput Plot (Dual Y-axis)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # BER on left axis
    color1 = 'tab:red'
    ax1.set_xlabel('Turbulence Strength ($C_n^2$) [$m^{-2/3}$]', fontsize=12)
    ax1.set_ylabel('Bit Error Rate (BER)', color=color1, fontsize=12)
    ax1.semilogx(unique_cn2, ber_per_cn2, 'o-', color=color1, linewidth=2, markersize=6, label='BER')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.set_ylim(bottom=0, top=max(ber_per_cn2) * 1.1)
    
    # Throughput on right axis
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Throughput (Gbps)', color=color2, fontsize=12)
    ax2.semilogx(unique_cn2, throughput_per_cn2, 's-', color=color2, linewidth=2, markersize=6, label='Throughput')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=-0.5, top=max_throughput * 1.1)
    
    # Add FEC threshold annotation
    ax1.axhline(0.038, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(unique_cn2[0], 0.04, 'FEC Threshold (3.8%)', fontsize=9, color='orange')
    
    plt.title('BER and Throughput vs. Turbulence Strength', fontsize=14)
    fig.tight_layout()
    plt.savefig("evaluation_ber_throughput_combined.png", dpi=300)
    print("Saved 'evaluation_ber_throughput_combined.png'")
    
    # Save Data for Comparison Plotting
    np.savez("cnn_results.npz", cn2=unique_cn2, ber=ber_per_cn2, throughput=throughput_per_cn2)
    print("Saved 'cnn_results.npz'")
    
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
    plt.savefig("evaluation_constellation.png")
    print("Saved 'evaluation_constellation.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('dataset'))
    parser.add_argument('--dataset_name', type=str, default='fso_oam_turbulence_v1')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet18_cbam'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    evaluate(args)
