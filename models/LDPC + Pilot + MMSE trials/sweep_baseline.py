#!/usr/bin/env python3
"""
Canonical Cn^2 sweep for the classical baseline receiver defined in pipeline.py.

This script:
1. Freezes all system settings except Cn^2
2. Sweeps both MMSE and ZF equalizers
3. Repeats each operating point multiple times with different RNG seeds
4. Saves raw and aggregated JSON results
5. Produces minimalist IEEE-style figures (no titles, only axes/legends)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from pipeline import SimulationConfig, run_e2e_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IEEE-style Cn^2 sweep for MMSE/ZF baseline")
    parser.add_argument("--cn2-min", type=float, default=1e-18, help="Minimum Cn^2")
    parser.add_argument("--cn2-max", type=float, default=1e-12, help="Maximum Cn^2")
    parser.add_argument("--num-points", type=int, default=41, help="Log-spaced Cn^2 points")
    parser.add_argument("--repeats", type=int, default=3, help="Independent repeats per point")
    parser.add_argument("--base-seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--snr-db", type=float, default=35.0, help="Fixed SNR in dB")
    parser.add_argument("--n-grid", type=int, default=256, help="Simulation grid size")
    parser.add_argument("--num-screens", type=int, default=10, help="Number of phase screens")
    parser.add_argument("--ldpc-blocks", type=int, default=2, help="LDPC blocks per frame")
    parser.add_argument("--output-dir", type=Path, default=Path("ieee_cn2_sweep_results"),
                        help="Directory for JSON and figures")
    parser.add_argument("--representative-cn2", type=float, default=1e-14,
                        help="Cn^2 for representative strong-turbulence channel matrix")
    parser.add_argument("--representative-filename", type=str, default="representative_channel_matrix.png",
                        help="Filename for the representative channel matrix image")
    parser.add_argument("--representative-vmax", type=float, default=None,
                        help="Optional fixed colorbar maximum for channel matrix comparisons")
    parser.add_argument("--equalizers", nargs="+", default=["mmse"],
                        choices=["mmse", "zf"],
                        help="Equalizers to include in the sweep and plots")
    parser.add_argument("--input-json", type=Path, default=None,
                        help="Existing aggregated sweep JSON to replot without rerunning the sweep")
    return parser.parse_args()


def build_config(cn2: float, eq_method: str, args: argparse.Namespace) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.CN2 = float(cn2)
    cfg.EQ_METHOD = eq_method

    # Freeze all non-Cn^2 settings for the sweep.
    cfg.SNR_DB = float(args.snr_db)
    cfg.ADD_NOISE = True
    cfg.N_GRID = int(args.n_grid)
    cfg.NUM_SCREENS = int(args.num_screens)
    cfg.LDPC_BLOCKS = int(args.ldpc_blocks)
    cfg.ENABLE_POWER_PROBE = False

    return cfg


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def run_single_point(cn2: float, eq_method: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    seed_everything(seed)
    cfg = build_config(cn2, eq_method, args)

    started = time.time()
    try:
        coded_ber = 1.0
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            results = run_e2e_simulation(cfg, verbose=False)
        elapsed = time.time() - started

        if results is None:
            raise RuntimeError("run_e2e_simulation returned None")

        metrics = results["metrics"]
        h_est = np.asarray(metrics["H_est"])
        if metrics.get("coded_ber") is not None:
            coded_ber = float(metrics["coded_ber"])
        pre_info_ber = coded_ber
        if metrics.get("info_ber_pre_ldpc") is not None:
            pre_info_ber = float(metrics["info_ber_pre_ldpc"])

        return {
            "success": True,
            "seed": int(seed),
            "cn2": float(cn2),
            "equalizer": eq_method,
            "ber": float(metrics.get("ber", 1.0)),
            "coded_ber": coded_ber,
            "pre_info_ber": pre_info_ber,
            "cond_h": float(metrics.get("cond_H", np.linalg.cond(h_est))),
            "noise_var": float(metrics.get("noise_var", 0.0)),
            "bit_errors": int(metrics.get("bit_errors", 0)),
            "total_bits": int(metrics.get("total_bits", 0)),
            "elapsed_s": float(elapsed),
            "h_est_abs": np.abs(h_est).tolist(),
        }
    except Exception as exc:
        elapsed = time.time() - started
        return {
            "success": False,
            "seed": int(seed),
            "cn2": float(cn2),
            "equalizer": eq_method,
            "ber": 1.0,
            "coded_ber": 1.0,
            "pre_info_ber": 1.0,
            "cond_h": float("inf"),
            "noise_var": 0.0,
            "bit_errors": 0,
            "total_bits": 0,
            "elapsed_s": float(elapsed),
            "error": str(exc),
        }


def aggregate_runs(raw_runs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    aggregated: Dict[str, List[Dict[str, Any]]] = {}

    for eq_method, eq_runs in raw_runs.items():
        buckets: Dict[float, List[Dict[str, Any]]] = {}
        for item in eq_runs:
            buckets.setdefault(float(item["cn2"]), []).append(item)

        aggregated[eq_method] = []
        for cn2 in sorted(buckets.keys()):
            valid = [r for r in buckets[cn2] if r.get("success")]
            if not valid:
                aggregated[eq_method].append({
                    "cn2": cn2,
                    "n_success": 0,
                    "n_total": len(buckets[cn2]),
                    "ber_mean": 1.0,
                    "ber_std": 0.0,
                    "coded_ber_mean": 1.0,
                    "coded_ber_std": 0.0,
                    "pre_info_ber_mean": 1.0,
                    "pre_info_ber_std": 0.0,
                    "cond_h_mean": float("inf"),
                    "cond_h_std": 0.0,
                    "noise_var_mean": 0.0,
                    "noise_var_std": 0.0,
                })
                continue

            ber_vals = np.array([r["ber"] for r in valid], dtype=float)
            coded_vals = np.array([r["coded_ber"] for r in valid], dtype=float)
            pre_info_vals = np.array([r.get("pre_info_ber", r["coded_ber"]) for r in valid], dtype=float)
            cond_vals = np.array([r["cond_h"] for r in valid], dtype=float)
            noise_vals = np.array([r["noise_var"] for r in valid], dtype=float)

            aggregated[eq_method].append({
                "cn2": cn2,
                "n_success": len(valid),
                "n_total": len(buckets[cn2]),
                "ber_mean": float(np.mean(ber_vals)),
                "ber_std": float(np.std(ber_vals)),
                "coded_ber_mean": float(np.mean(coded_vals)),
                "coded_ber_std": float(np.std(coded_vals)),
                "pre_info_ber_mean": float(np.mean(pre_info_vals)),
                "pre_info_ber_std": float(np.std(pre_info_vals)),
                "cond_h_mean": float(np.mean(cond_vals)),
                "cond_h_std": float(np.std(cond_vals)),
                "noise_var_mean": float(np.mean(noise_vals)),
                "noise_var_std": float(np.std(noise_vals)),
            })

    return aggregated


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2)


def ieee_style() -> None:
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


def save_dual(fig: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)


def plot_cn2_vs_ber(
    aggregated: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    equalizers: List[str],
) -> None:
    ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    colors = {"mmse": "#1f77b4", "zf": "#d62728"}
    markers = {"mmse": "o", "zf": "s"}

    for eq_method in equalizers:
        if eq_method not in aggregated:
            continue
        rows = aggregated[eq_method]
        cn2 = np.array([r["cn2"] for r in rows], dtype=float)
        ber = np.array([r["ber_mean"] for r in rows], dtype=float)
        ber_std = np.array([r["ber_std"] for r in rows], dtype=float)

        ax.semilogy(cn2, ber, marker=markers[eq_method], color=colors[eq_method], label=eq_method.upper())
        lower = np.clip(ber - ber_std, 1e-6, None)
        upper = np.clip(ber + ber_std, 1e-6, 1.0)
        ax.fill_between(cn2, lower, upper, color=colors[eq_method], alpha=0.12, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"$C_n^2$ [$m^{-2/3}$]")
    ax.set_ylabel("BER")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.set_ylim(1e-5, 1.0)

    save_dual(fig, output_dir / "cn2_vs_ber.png")
    plt.close(fig)


def plot_pre_post_ber(
    aggregated: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    equalizers: List[str],
) -> None:
    ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    colors = {"mmse": "#1f77b4", "zf": "#d62728"}
    markers = {"mmse": "o", "zf": "s"}

    for eq_method in equalizers:
        if eq_method not in aggregated:
            continue
        rows = aggregated[eq_method]
        cn2 = np.array([r["cn2"] for r in rows], dtype=float)
        post_ber = np.array([r["ber_mean"] for r in rows], dtype=float)
        pre_ber = np.array(
            [r.get("pre_info_ber_mean", r.get("coded_ber_mean", 1.0)) for r in rows],
            dtype=float,
        )

        ax.semilogy(cn2, post_ber, marker=markers[eq_method], color=colors[eq_method],
                    label=f"{eq_method.upper()} post-LDPC")
        ax.semilogy(cn2, pre_ber, marker=markers[eq_method], color=colors[eq_method],
                    linestyle="--", label=f"{eq_method.upper()} pre-LDPC (info, hard)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$C_n^2$ [$m^{-2/3}$]")
    ax.set_ylabel("BER")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, ncol=1)
    ax.set_ylim(1e-5, 1.0)

    save_dual(fig, output_dir / "pre_post_ldpc_ber.png")
    plt.close(fig)


def plot_representative_channel(raw_runs: Dict[str, List[Dict[str, Any]]], args: argparse.Namespace, output_dir: Path) -> None:
    ieee_style()

    mmse_runs = raw_runs.get("mmse", [])
    if not mmse_runs:
        return

    target = float(args.representative_cn2)
    nearest = min(
        (r for r in mmse_runs if r.get("success") and "h_est_abs" in r),
        key=lambda r: abs(float(r["cn2"]) - target),
        default=None,
    )
    if nearest is None:
        return

    h_abs = np.array(nearest["h_est_abs"], dtype=float)
    fig, ax = plt.subplots(figsize=(3.5, 3.1), constrained_layout=True)
    im = ax.imshow(
        h_abs,
        cmap="viridis",
        interpolation="nearest",
        vmin=0.0,
        vmax=args.representative_vmax,
    )

    mode_labels = [f"({p},{l})" for p, l in SimulationConfig.SPATIAL_MODES]
    ax.set_xticks(np.arange(len(mode_labels)))
    ax.set_yticks(np.arange(len(mode_labels)))
    ax.set_xticklabels(mode_labels, rotation=45, ha="right")
    ax.set_yticklabels(mode_labels)
    ax.set_xlabel("TX mode")
    ax.set_ylabel("RX mode")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\hat{H}|$")

    save_dual(fig, output_dir / args.representative_filename)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_json is not None:
        with open(args.input_json, "r") as f:
            payload = json.load(f)

        aggregated = payload["aggregated"]
        raw_runs = payload["raw_runs"]

        plot_cn2_vs_ber(aggregated, args.output_dir, args.equalizers)
        plot_pre_post_ber(aggregated, args.output_dir, args.equalizers)
        plot_representative_channel(raw_runs, args, args.output_dir)

        print("\nReplotted from existing JSON:")
        print(f"  {args.output_dir / 'cn2_vs_ber.png'}")
        print(f"  {args.output_dir / 'pre_post_ldpc_ber.png'}")
        print(f"  {args.output_dir / 'representative_channel_matrix.png'}")
        return

    cn2_values = np.logspace(np.log10(args.cn2_min), np.log10(args.cn2_max), args.num_points)
    equalizers = tuple(args.equalizers)

    raw_runs: Dict[str, List[Dict[str, Any]]] = {eq: [] for eq in equalizers}
    started = time.time()

    total_runs = len(cn2_values) * len(equalizers) * args.repeats
    run_idx = 0

    print(f"\nRunning canonical sweep via pipeline.py")
    print(f"Cn^2 range: {args.cn2_min:.2e} -> {args.cn2_max:.2e}")
    print(f"Points: {args.num_points}, repeats: {args.repeats}, equalizers: {', '.join(equalizers)}")
    print(f"Frozen config: N_GRID={args.n_grid}, NUM_SCREENS={args.num_screens}, "
          f"LDPC_BLOCKS={args.ldpc_blocks}, SNR={args.snr_db:.1f} dB")

    for eq_method in equalizers:
        for cn2_idx, cn2 in enumerate(cn2_values):
            for rep in range(args.repeats):
                run_idx += 1
                seed = args.base_seed + rep + (cn2_idx * 1000) + (0 if eq_method == "mmse" else 100000)
                print(f"[{run_idx}/{total_runs}] {eq_method.upper()}  Cn^2={cn2:.2e}  repeat={rep+1}/{args.repeats}  seed={seed}")
                result = run_single_point(float(cn2), eq_method, seed, args)
                raw_runs[eq_method].append(result)

                # Checkpoint after every run to preserve progress if interrupted.
                checkpoint = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "args": vars(args),
                    "raw_runs": raw_runs,
                }
                save_json(args.output_dir / "baseline_sweep_raw.json", checkpoint)

    aggregated = aggregate_runs(raw_runs)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": float(time.time() - started),
        "args": vars(args),
        "cn2_values": cn2_values.tolist(),
        "equalizers": list(equalizers),
        "raw_runs": raw_runs,
        "aggregated": aggregated,
    }
    save_json(args.output_dir / "baseline_sweep_aggregated.json", payload)

    plot_cn2_vs_ber(aggregated, args.output_dir, args.equalizers)
    plot_pre_post_ber(aggregated, args.output_dir, args.equalizers)
    plot_representative_channel(raw_runs, args, args.output_dir)

    print("\nSaved:")
    print(f"  {args.output_dir / 'baseline_sweep_raw.json'}")
    print(f"  {args.output_dir / 'baseline_sweep_aggregated.json'}")
    print(f"  {args.output_dir / 'cn2_vs_ber.png'}")
    print(f"  {args.output_dir / 'pre_post_ldpc_ber.png'}")
    print(f"  {args.output_dir / 'representative_channel_matrix.png'}")


if __name__ == "__main__":
    main()
