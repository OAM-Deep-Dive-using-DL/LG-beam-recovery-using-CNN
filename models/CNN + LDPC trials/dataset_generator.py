from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
from tqdm import tqdm
import contextlib
import io

# Reuse physics / DSP components from the LDPC + Pilot + ZF trials
from encoding import encodingRunner
from fsplAtmAttenuation import calculate_geometric_loss, calculate_kim_attenuation
from lgBeam import LaguerreGaussianBeam
from turbulence import (
    AtmosphericTurbulence,
    apply_multi_layer_turbulence,
    create_multi_layer_screens,
)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    # Optical / link parameters
    wavelength_m: float = 1550e-9
    w0_m: float = 25e-3
    distance_m: float = 800.0
    receiver_diameter_m: float = 0.3
    total_tx_power_w: float = 1.0

    # Grid / propagation
    n_grid: int = 512
    oversampling: int = 2
    num_screens: int = 10
    outer_scale_m: float = 10.0
    inner_scale_m: float = 0.005

    # Digital parameters
    spatial_modes: Sequence[Tuple[int, int]] = (
        (0, -1),
        (0, 1),
        (0, -3),
        (0, 3),
        (0, -4),
        (0, 4),
        (1, -1),
        (1, 1),
    )
    fec_rate: float = 0.5
    pilot_ratio: float = 0.2
    symbol_time_ps: float = 1000.0
    ldpc_blocks: int = 4

    # Dataset sweep parameters
    cn2_values: Sequence[float] = (
        0.0,
        5e-19,
        1e-18,
        5e-18,
        1e-17,
        5e-17,
        1e-16,
    )
    turbulence_seeds_per_cn2: int = 5
    payload_seeds_per_cn2: int = 5

    # Output / storage
    downsample_size: int = 128
    chunk_size: int = 512  # samples per chunk in HDF5
    dtype: str = "float32"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def stride_downsample(field: np.ndarray, target_size: int) -> np.ndarray:
    """
    Downsample a complex field to target_size x target_size using stride pooling.
    """
    if field.shape[0] % target_size != 0 or field.shape[1] % target_size != 0:
        raise ValueError(
            f"Field shape {field.shape} not divisible by target size {target_size}."
        )
    factor = field.shape[0] // target_size
    reshaped = field.reshape(
        target_size, factor, target_size, factor
    )  # (target, f, target, f)
    downsampled = reshaped.mean(axis=(1, 3))
    return downsampled


def stack_real_imag(field: np.ndarray) -> np.ndarray:
    """Stack real and imaginary components into channel-last format."""
    stacked = np.stack([field.real, field.imag], axis=-1)
    return stacked.astype(np.float32)


def compute_aperture_mask(X: np.ndarray, Y: np.ndarray, radius: float) -> np.ndarray:
    """Create a circular aperture mask."""
    return (np.sqrt(X**2 + Y**2) <= radius).astype(float)


def initialise_grid(cfg: DatasetConfig) -> Tuple[Dict[str, np.ndarray], float]:
    """Build spatial grid and return dictionary + pixel area."""
    max_mode = max(cfg.spatial_modes, key=lambda m: abs(m[1]))
    dummy_beam = LaguerreGaussianBeam(
        max_mode[0], max_mode[1], cfg.wavelength_m, cfg.w0_m
    )
    beam_size_at_rx = dummy_beam.beam_waist(cfg.distance_m)
    grid_extent = cfg.oversampling * 6 * beam_size_at_rx

    x = np.linspace(-grid_extent / 2, grid_extent / 2, cfg.n_grid)
    y = np.linspace(-grid_extent / 2, grid_extent / 2, cfg.n_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    delta = grid_extent / cfg.n_grid
    grid_info = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "R": R,
        "PHI": PHI,
        "extent": grid_extent,
        "delta": delta,
    }
    return grid_info, delta**2


def initialise_transmitter(cfg: DatasetConfig) -> encodingRunner:
    """Create the encoding runner with shared LDPC/pilot configuration."""
    with contextlib.redirect_stdout(io.StringIO()):
        runner = encodingRunner(
            spatial_modes=list(cfg.spatial_modes),
            wavelength=cfg.wavelength_m,
            w0=cfg.w0_m,
            fec_rate=cfg.fec_rate,
            pilot_ratio=cfg.pilot_ratio,
            symbol_time_s=cfg.symbol_time_ps * 1e-12,
        )
    # Adjust info bits to an integer number of LDPC blocks
    ldpc_k = runner.ldpc.k
    runner.n_info_bits = int(ldpc_k * cfg.ldpc_blocks)
    return runner


def generate_frame_samples(
    cfg: DatasetConfig,
    transmitter: encodingRunner,
    grid_info: Dict[str, np.ndarray],
    dA: float,
    cn2: float,
    turb_seed: int,
    payload_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate samples for a single frame (all symbols) at a given CN² and seeds.

    Returns:
        fields_ds: (num_symbols, H, W, 2) downsampled complex fields
        symbols_true: (num_symbols, n_modes, 2) true per-mode complex symbols
        metadata: dict containing arrays of metadata per sample
    """
    rng = np.random.default_rng(payload_seed)
    info_bits = rng.integers(
        0, 2, size=transmitter.n_info_bits, dtype=np.int8
    )
    tx_frame = transmitter.transmit(info_bits, verbose=False)

    symbols_per_mode = {
        mode: sig["symbols"]
        for mode, sig in tx_frame.tx_signals.items()
    }
    symbol_count = min(len(s) for s in symbols_per_mode.values())
    mode_list = list(cfg.spatial_modes)
    n_modes = len(mode_list)

    # Basis fields at transmitter plane (scaled)
    delta = grid_info["delta"]
    basis_fields = {}
    base_beam = None
    for mode_key, beam in transmitter.lg_beams.items():
        if beam is None:
            continue
        if base_beam is None:
            base_beam = beam
        E_basis = beam.generate_beam_field(grid_info["R"], grid_info["PHI"], 0.0)
        energy = np.sum(np.abs(E_basis) ** 2) * dA
        scale = np.sqrt(cfg.total_tx_power_w / (n_modes * energy))
        basis_fields[mode_key] = E_basis * scale
    if base_beam is None:
        raise RuntimeError("No valid Laguerre-Gaussian beam available for propagation.")

    # Turbulence layers
    np.random.seed(turb_seed)
    layers = create_multi_layer_screens(
        cfg.distance_m,
        cfg.num_screens,
        cfg.wavelength_m,
        cn2,
        cfg.outer_scale_m,
        cfg.inner_scale_m,
        cn2_model="uniform",
        verbose=False,
    )

    # Atmospheric attenuation
    receiver_radius = cfg.receiver_diameter_m / 2.0
    eta_modes = []
    for mode_key, beam in transmitter.lg_beams.items():
        _, eta_mode = calculate_geometric_loss(
            beam, cfg.distance_m, receiver_radius
        )
        eta_modes.append(eta_mode)
    eta_mean = np.mean(eta_modes)
    alpha_dBkm = calculate_kim_attenuation(
        cfg.wavelength_m * 1e9, visibility_km=23.0
    )
    L_atm_dB = alpha_dBkm * (cfg.distance_m / 1000.0)
    amplitude_loss = 10 ** (-L_atm_dB / 20.0)

    aperture_mask = compute_aperture_mask(
        grid_info["X"], grid_info["Y"], receiver_radius
    )

    fields_ds = []
    symbols_true = []
    metadata = {
        "cn2": [],
        "turb_seed": [],
        "payload_seed": [],
        "symbol_index": [],
    }

    X = grid_info["X"]
    Y = grid_info["Y"]

    for sym_idx in range(symbol_count):
        E_tx = np.zeros_like(X, dtype=np.complex128)
        symbol_vec = np.zeros((n_modes,), dtype=np.complex64)
        for mode_idx, mode_key in enumerate(mode_list):
            sym_val = symbols_per_mode[mode_key][sym_idx]
            E_tx += basis_fields[mode_key] * sym_val
            symbol_vec[mode_idx] = sym_val

        result = apply_multi_layer_turbulence(
            initial_field=E_tx,
            base_beam=base_beam,
            layers=layers,
            total_distance=cfg.distance_m,
            N=cfg.n_grid,
            oversampling=cfg.oversampling,
            L0=cfg.outer_scale_m,
            l0=cfg.inner_scale_m,
        )
        E_rx = result["final_field"] * amplitude_loss
        E_rx *= aperture_mask

        field_ds = stride_downsample(E_rx, cfg.downsample_size)
        fields_ds.append(stack_real_imag(field_ds))

        symbol_stack = np.stack(
            [symbol_vec.real, symbol_vec.imag], axis=-1
        ).astype(np.float32)
        symbols_true.append(symbol_stack)

        metadata["cn2"].append(cn2)
        metadata["turb_seed"].append(turb_seed)
        metadata["payload_seed"].append(payload_seed)
        metadata["symbol_index"].append(sym_idx)

    return (
        np.asarray(fields_ds, dtype=np.float32),
        np.asarray(symbols_true, dtype=np.float32),
        {k: np.asarray(v) for k, v in metadata.items()},
    )


def write_hdf5(
    output_path: str,
    config: DatasetConfig,
    fields_iter: Iterable[np.ndarray],
    symbols_iter: Iterable[np.ndarray],
    metadata_iter: Iterable[Dict[str, np.ndarray]],
) -> None:
    """Stream batches into an HDF5 file with extendable datasets."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        hf.attrs["config"] = config.to_json()
        hf.attrs["dataset_uuid"] = str(uuid.uuid4())

        X_ds = hf.create_dataset(
            "fields",
            shape=(0, config.downsample_size, config.downsample_size, 2),
            maxshape=(None, config.downsample_size, config.downsample_size, 2),
            chunks=(config.chunk_size, config.downsample_size, config.downsample_size, 2),
            dtype=config.dtype,
        )
        Y_ds = hf.create_dataset(
            "symbols",
            shape=(0, len(config.spatial_modes), 2),
            maxshape=(None, len(config.spatial_modes), 2),
            chunks=(config.chunk_size, len(config.spatial_modes), 2),
            dtype=config.dtype,
        )
        meta_group = hf.create_group("metadata")
        meta_buffers: Dict[str, h5py.Dataset] = {}

        total_samples = 0
        for fields, symbols, meta in zip(fields_iter, symbols_iter, metadata_iter):
            batch_size = fields.shape[0]
            if batch_size == 0:
                continue
            new_size = total_samples + batch_size
            X_ds.resize(new_size, axis=0)
            Y_ds.resize(new_size, axis=0)
            X_ds[total_samples:new_size] = fields
            Y_ds[total_samples:new_size] = symbols

            for key, values in meta.items():
                if key not in meta_buffers:
                    meta_buffers[key] = meta_group.create_dataset(
                        key,
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(config.chunk_size,),
                        dtype=values.dtype,
                    )
                ds = meta_buffers[key]
                ds.resize(new_size, axis=0)
                ds[total_samples:new_size] = values

            total_samples = new_size

        hf.attrs["num_samples"] = total_samples
        print(f"✓ Saved {total_samples} samples to {output_path}")


def generate_dataset(config: DatasetConfig, output_path: str) -> None:
    """Main dataset generation routine."""
    grid_info, dA = initialise_grid(config)
    transmitter = initialise_transmitter(config)

    fields_batches = []
    symbols_batches = []
    metadata_batches = []

    for cn2 in config.cn2_values:
        turb_seeds = range(config.turbulence_seeds_per_cn2)
        payload_seeds = range(config.payload_seeds_per_cn2)
        for turb_seed in turb_seeds:
            for payload_seed in payload_seeds:
                fields, symbols, meta = generate_frame_samples(
                    config,
                    transmitter,
                    grid_info,
                    dA,
                    cn2,
                    turb_seed=int(turb_seed),
                    payload_seed=int(payload_seed),
                )
                fields_batches.append(fields)
                symbols_batches.append(symbols)
                metadata_batches.append(meta)

    write_hdf5(output_path, config, fields_batches, symbols_batches, metadata_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CNN + LDPC training dataset.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output HDF5 dataset file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to JSON config file overriding defaults.",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> DatasetConfig:
    if config_path is None:
        return DatasetConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetConfig(**data)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("Dataset configuration:")
    print(cfg.to_json())
    generate_dataset(cfg, args.output)


if __name__ == "__main__":
    main()

