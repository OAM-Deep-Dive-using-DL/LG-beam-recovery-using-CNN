import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def load_baseline_curve(path: Path):
    with open(path, 'r') as f:
        payload = json.load(f)
    rows = payload['aggregated']['mmse']
    cn2 = np.array([row['cn2'] for row in rows], dtype=float)
    ber = np.array([row['ber_mean'] for row in rows], dtype=float)
    return cn2, ber


def load_cnn_curve(path: Path):
    payload = np.load(path)
    return np.asarray(payload['cn2'], dtype=float), np.asarray(payload['ber'], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Compare CNN BER curve against MMSE baseline")
    parser.add_argument('--cnn-results', type=Path, required=True, help='Path to CNN results .npz')
    parser.add_argument('--baseline-json', type=Path, required=True, help='Path to MMSE baseline aggregated JSON')
    parser.add_argument('--cnn-label', type=str, default='CNN', help='Legend label for CNN curve')
    parser.add_argument('--output', type=Path, required=True, help='Output PNG/PDF path')
    args = parser.parse_args()

    cnn_cn2, cnn_ber = load_cnn_curve(args.cnn_results)
    mmse_cn2, mmse_ber = load_baseline_curve(args.baseline_json)

    ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    ax.semilogy(mmse_cn2, mmse_ber, marker='o', color='#1f77b4', label='MMSE baseline')
    ax.semilogy(cnn_cn2, cnn_ber, marker='s', color='#d62728', label=args.cnn_label)
    ax.set_xscale('log')
    ax.set_xlabel(r"$C_n^2$ [$m^{-2/3}$]")
    ax.set_ylabel("BER")
    ax.grid(True, which='major', alpha=0.25)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95)
    ax.set_ylim(1e-5, 1.0)
    save_dual(fig, args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
