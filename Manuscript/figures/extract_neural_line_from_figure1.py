#!/usr/bin/env python3
"""
Digitize the blue (Neural Receiver) curve from figure1_ber_comparison_IEEE.png and
regenerate a neural-only IEEE-style figure.

Calibration (no original CSV in repo):
  - x pixel range [x_lo, x_hi] maps linearly to log10(C_n^2) in [-18, -12] (semilog axis span).
  - y row range: bottom of flat region ~ y0 BER=0; top of curve y_min -> BER=0.5.

Outputs:
  - figure1_neural_receiver_ber_digitized.csv  (full resolution along x)
  - figure1_neural_receiver_ber_markers.csv    (~30 points at triangle markers, by peak detection)
  - figure1_ber_neural_only_IEEE.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FIG_PATH = Path(__file__).resolve().parent / "figure1_ber_comparison_IEEE.png"
OUT_CSV = Path(__file__).resolve().parent / "figure1_neural_receiver_ber_digitized.csv"
OUT_MARKERS = Path(__file__).resolve().parent / "figure1_neural_receiver_ber_markers.csv"
OUT_JSON = Path(__file__).resolve().parent / "figure1_neural_receiver_ber_digitized.json"
OUT_PNG = Path(__file__).resolve().parent / "figure1_ber_neural_only_IEEE.png"
META_JSON = Path(__file__).resolve().parent / "figure1_digitization_calibration.json"

TAB10_BLUE = (31, 119, 180)
TOL = 45


def mask_color(rgb: np.ndarray, target: tuple[int, int, int], tol: float = TOL) -> np.ndarray:
    t = np.array(target, dtype=np.float32)
    d = np.abs(rgb.astype(np.float32) - t.reshape(1, 1, 3))
    return np.all(d <= tol, axis=2)


def trace_blue_curve(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = b.shape
    xs: list[int] = []
    ys: list[float] = []
    y_prev: float | None = None
    for x in range(w):
        ys_col = np.where(b[:, x])[0]
        if ys_col.size == 0:
            continue
        if y_prev is None:
            y_prev = float(np.median(ys_col))
        else:
            near = ys_col[np.abs(ys_col - y_prev) <= 150]
            if near.size > 0:
                y_prev = float(np.median(near))
            else:
                y_prev = float(ys_col[np.argmin(np.abs(ys_col - y_prev))])
        xs.append(x)
        ys.append(y_prev)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def main() -> None:
    im = np.array(Image.open(FIG_PATH).convert("RGB"))
    b = mask_color(im, TAB10_BLUE)
    xs, ys = trace_blue_curve(b)

    # Axis calibration (see module docstring)
    x_lo, x_hi = 358.0, 3987.0
    log_lo, log_hi = -18.0, -12.0

    # BER=0 baseline from flat left segment
    n_flat = max(50, int(len(xs) * 0.55))
    y_ber0 = float(np.median(ys[:n_flat]))
    y_ber50 = float(np.min(ys))
    denom = y_ber0 - y_ber50
    if denom <= 1.0:
        raise RuntimeError("Bad y-axis calibration")

    log10_cn2 = log_lo + (xs - x_lo) / (x_hi - x_lo) * (log_hi - log_lo)
    cn2 = np.power(10.0, log10_cn2)
    ber = 0.5 * (y_ber0 - ys) / denom
    ber = np.clip(ber, 0.0, 0.5)
    # Suppress raster/linewidth noise in the error-free regime (stroke thickness + antialiasing).
    ber = np.where((cn2 < 4e-15) & (ber < 0.02), 0.0, ber)

    meta = {
        "source_image": str(FIG_PATH.name),
        "x_pixel_range": [x_lo, x_hi],
        "log10_cn2_range": [log_lo, log_hi],
        "y_ber0_row": y_ber0,
        "y_ber50_row": y_ber50,
        "note": (
            "Digitized from raster; original simulation CSV was not present in the repo. "
            "Cn2 uses linear mapping from x-pixel to log10(Cn2) over the figure width."
        ),
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Full-resolution CSV
    hdr = "x_pixel,y_row,Cn2,BER,BER_percent"
    lines = [hdr]
    for i in range(len(xs)):
        lines.append(f"{xs[i]:.1f},{ys[i]:.6f},{cn2[i]:.15e},{ber[i]:.10f},{100.0 * ber[i]:.6f}")
    OUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    arr = np.column_stack([xs, ys, cn2, ber])
    OUT_JSON.write_text(
        json.dumps(
            {
                "cn2": cn2.tolist(),
                "ber": ber.tolist(),
                "x_pixel": xs.tolist(),
                "y_row": ys.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Marker subsample: local maxima of |d^2 y / d (log cn2)^2| ~ curvature proxies on sparse grid
    # Simpler: take one point per ~120 columns (triangle spacing ~)
    step = max(1, int((x_hi - x_lo) / 29))
    idx = np.arange(0, len(xs), step)
    mlines = [hdr]
    for i in idx:
        mlines.append(f"{xs[i]:.1f},{ys[i]:.6f},{cn2[i]:.15e},{ber[i]:.10f},{100.0 * ber[i]:.6f}")
    OUT_MARKERS.write_text("\n".join(mlines) + "\n", encoding="utf-8")

    # Neural-only figure (IEEE-friendly)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.semilogx(cn2, ber, "^-", color="#1f77b4", linewidth=2.5, markersize=7, label="Neural Receiver (ConvNeXt)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(1e-15, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlim(1e-18, 1e-12)
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel(r"Turbulence Strength, $C_n^2$ ($m^{-2/3}$)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100.0 * y:.0f}%"))
    ax.grid(True, which="both", linestyle="-", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MARKERS}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {META_JSON}")


if __name__ == "__main__":
    main()
