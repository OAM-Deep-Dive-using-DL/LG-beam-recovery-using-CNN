#!/usr/bin/env python3
"""
Cn² Sweep Script for MMSE/ZF Equalization Performance Characterization

Sweeps through different turbulence strengths (Cn²) to find the operating
limits of classical MMSE and ZF equalizers.

Usage:
    python cn2_sweep.py --equalizer mmse --num-points 15
    python cn2_sweep.py --equalizer zf --num-points 15
    python cn2_sweep.py --equalizer both --num-points 15
"""

import argparse
import json
import sys
import time
import io
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from pipeline
from pipeline import SimulationConfig, run_e2e_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cn² sweep for equalizer characterization")
    parser.add_argument("--cn2-min", type=float, default=1e-18, help="Minimum Cn² value")
    parser.add_argument("--cn2-max", type=float, default=1e-15, help="Maximum Cn² value")
    parser.add_argument("--num-points", type=int, default=15, help="Number of Cn² points to test")
    parser.add_argument("--equalizer", type=str, default="both", choices=["zf", "mmse", "both"],
                        help="Which equalizer(s) to test")
    parser.add_argument("--output-dir", type=Path, default=Path("cn2_sweep_results"),
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def generate_cn2_values(cn2_min: float, cn2_max: float, num_points: int) -> np.ndarray:
    """Generate logarithmically spaced Cn² values."""
    return np.logspace(np.log10(cn2_min), np.log10(cn2_max), num_points)


def run_single_cn2_test(cn2: float, eq_method: str, config: SimulationConfig) -> Dict:
    """Run a single simulation with given Cn² and equalizer."""
    # Update config
    config.CN2 = cn2
    config.EQ_METHOD = eq_method
    
    try:
        # Run simulation with output suppressed
        with redirect_stdout(io.StringIO()):
            results = run_e2e_simulation(config, verbose=False)
        
        if results is None:
            raise ValueError("Simulation returned None")
        
        metrics = results.get("metrics", {})
        
        return {
            "cn2": float(cn2),
            "equalizer": eq_method,
            "ber": float(metrics.get("ber", 1.0)),
            "coded_ber": float(metrics.get("coded_ber", 1.0)),
            "h_est_condition": float(metrics.get("cond_H", np.inf)),
            "noise_var_est": float(metrics.get("noise_var_est", 0.0)),
            "bit_errors": int(metrics.get("bit_errors", 0)),
            "total_bits": int(metrics.get("total_bits", 0)),
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "cn2": float(cn2),
            "equalizer": eq_method,
            "ber": 1.0,
            "coded_ber": 1.0,
            "h_est_condition": np.inf,
            "noise_var_est": 0.0,
            "bit_errors": 0,
            "total_bits": 0,
            "success": False,
            "error": str(e),
        }


def run_cn2_sweep(args: argparse.Namespace) -> Dict:
    """Run the full Cn² sweep."""
    # Generate Cn² values
    cn2_values = generate_cn2_values(args.cn2_min, args.cn2_max, args.num_points)
    
    # Determine which equalizers to test
    if args.equalizer == "both":
        equalizers = ["zf", "mmse"]
    else:
        equalizers = [args.equalizer]
    
    # Create base config
    config = SimulationConfig()
    
    # Results storage
    results = {
        "cn2_values": cn2_values.tolist(),
        "equalizers": equalizers,
        "data": {eq: [] for eq in equalizers},
        "config": {
            "cn2_min": args.cn2_min,
            "cn2_max": args.cn2_max,
            "num_points": args.num_points,
            "distance": config.DISTANCE,
            "wavelength": config.WAVELENGTH,
            "spatial_modes": config.SPATIAL_MODES,
            "n_modes": len(config.SPATIAL_MODES),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Run sweep
    total_tests = len(cn2_values) * len(equalizers)
    print(f"\n{'='*80}")
    print(f"Cn² SWEEP: {args.num_points} points from {args.cn2_min:.2e} to {args.cn2_max:.2e}")
    print(f"Equalizers: {', '.join(equalizers)}")
    print(f"Total tests: {total_tests}")
    print(f"{'='*80}\n")
    
    with tqdm(total=total_tests, desc="Running sweep") as pbar:
        for eq in equalizers:
            for cn2 in cn2_values:
                # Run test
                result = run_single_cn2_test(cn2, eq, config)
                results["data"][eq].append(result)
                
                # Update progress bar with current BER
                pbar.set_postfix({
                    "EQ": eq.upper(),
                    "Cn²": f"{cn2:.2e}",
                    "BER": f"{result['ber']:.4f}",
                })
                pbar.update(1)
    
    return results


def plot_results(results: Dict, output_dir: Path):
    """Generate plots from sweep results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cn2_values = np.array(results["cn2_values"])
    equalizers = results["equalizers"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cn² Sweep: Equalizer Performance Characterization", fontsize=14, fontweight='bold')
    
    # Plot 1: BER vs Cn²
    ax = axes[0, 0]
    for eq in equalizers:
        data = results["data"][eq]
        ber_values = [d["ber"] for d in data]
        ax.semilogy(cn2_values, ber_values, 'o-', label=eq.upper(), linewidth=2, markersize=6)
    
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% BER threshold')
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='10% BER threshold')
    ax.set_xlabel('Cn² [m$^{-2/3}$]', fontsize=11)
    ax.set_ylabel('BER (after LDPC)', fontsize=11)
    ax.set_title('Bit Error Rate vs Turbulence Strength', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    
    # Plot 2: Coded BER vs Cn²
    ax = axes[0, 1]
    for eq in equalizers:
        data = results["data"][eq]
        coded_ber_values = [d["coded_ber"] for d in data]
        ax.semilogy(cn2_values, coded_ber_values, 's-', label=f'{eq.upper()} (pre-LDPC)',
                    linewidth=2, markersize=6)
    
    ax.set_xlabel('Cn² [m$^{-2/3}$]', fontsize=11)
    ax.set_ylabel('Coded BER (before LDPC)', fontsize=11)
    ax.set_title('Pre-LDPC BER vs Turbulence Strength', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    
    # Plot 3: H condition number vs Cn²
    ax = axes[1, 0]
    for eq in equalizers:
        data = results["data"][eq]
        cond_values = [d["h_est_condition"] for d in data]
        ax.semilogy(cn2_values, cond_values, '^-', label=eq.upper(), linewidth=2, markersize=6)
    
    ax.set_xlabel('Cn² [m$^{-2/3}$]', fontsize=11)
    ax.set_ylabel('cond(H_est)', fontsize=11)
    ax.set_title('Channel Matrix Condition Number', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    
    # Plot 4: Noise variance estimate vs Cn²
    ax = axes[1, 1]
    for eq in equalizers:
        data = results["data"][eq]
        noise_var_values = [d["noise_var_est"] for d in data]
        ax.semilogy(cn2_values, noise_var_values, 'd-', label=eq.upper(), linewidth=2, markersize=6)
    
    ax.set_xlabel('Cn² [m$^{-2/3}$]', fontsize=11)
    ax.set_ylabel('Estimated Noise Variance σ²', fontsize=11)
    ax.set_title('Noise Variance Estimate (Model Error)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / "cn2_sweep_results.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    
    plt.close()


def print_summary(results: Dict):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for eq in results["equalizers"]:
        data = results["data"][eq]
        ber_values = np.array([d["ber"] for d in data])
        cn2_values = np.array([d["cn2"] for d in data])
        
        # Find threshold where BER < 1%
        threshold_1pct_idx = np.where(ber_values < 0.01)[0]
        if len(threshold_1pct_idx) > 0:
            max_cn2_1pct = cn2_values[threshold_1pct_idx[-1]]
            print(f"{eq.upper()} Equalizer:")
            print(f"  Max Cn² for BER < 1%:  {max_cn2_1pct:.2e} m^(-2/3)")
        else:
            print(f"{eq.upper()} Equalizer:")
            print(f"  Max Cn² for BER < 1%:  None (always > 1%)")
        
        # Find threshold where BER < 10%
        threshold_10pct_idx = np.where(ber_values < 0.1)[0]
        if len(threshold_10pct_idx) > 0:
            max_cn2_10pct = cn2_values[threshold_10pct_idx[-1]]
            print(f"  Max Cn² for BER < 10%: {max_cn2_10pct:.2e} m^(-2/3)")
        else:
            print(f"  Max Cn² for BER < 10%: None (always > 10%)")
        
        # Best and worst BER
        best_ber = np.min(ber_values)
        worst_ber = np.max(ber_values)
        print(f"  Best BER:  {best_ber:.4e} (Cn² = {cn2_values[np.argmin(ber_values)]:.2e})")
        print(f"  Worst BER: {worst_ber:.4e} (Cn² = {cn2_values[np.argmax(ber_values)]:.2e})")
        print()


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("Cn² SWEEP FOR EQUALIZER CHARACTERIZATION")
    print(f"{'='*80}")
    
    # Run sweep
    results = run_cn2_sweep(args)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "cn2_sweep_data.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Generate plots
    plot_results(results, args.output_dir)
    
    # Print summary
    print_summary(results)
    
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
