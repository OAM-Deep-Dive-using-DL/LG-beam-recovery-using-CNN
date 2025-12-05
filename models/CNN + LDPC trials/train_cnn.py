"""Training script for the neural OAM demultiplexer.
The script parses the simulation configuration stored alongside the dataset in
order to reconstruct the physical grid and LG mode basis.  Training targets are
cross-entropy labels over the Gray-mapped QPSK constellation together with a
small regression term that keeps the predicted complex symbols close to the
analytical values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lgBeam import LaguerreGaussianBeam
from encoding import PilotHandler

from cnn_model import NeuralDemuxConfig, OAMNeuralDemultiplexer
from torch.nn.utils import clip_grad_norm_


@dataclass
class DatasetMeta:
    spatial_modes: Tuple[Tuple[int, int], ...]
    wavelength: float
    w0: float
    distance: float
    grid_extent: float
    grid_size: int
    pilot_ratio: float
    frame_length: int
    fec_rate: float


_QPSK_POINTS = torch.tensor(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
    ],
    dtype=torch.float32,
) / math.sqrt(2.0)
_QPSK_BITS = torch.tensor(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ],
    dtype=torch.float32,
)


def _qpsk_symbol_to_index(symbol: torch.Tensor) -> torch.Tensor:
    diff = torch.cdist(symbol, _QPSK_POINTS.to(symbol.device))
    return diff.argmin(dim=-1)


def _qpsk_index_to_bits(index: torch.Tensor) -> torch.Tensor:
    return _QPSK_BITS.to(index.device)[index]


def _symbol_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "cosine":
        eps = 1e-8
        target_norm = target.norm(dim=-1)
        pred_norm = pred.norm(dim=-1)
        if torch.any(target_norm < eps) or torch.any(pred_norm < eps):
            return F.mse_loss(pred, target)
        cos = F.cosine_similarity(pred, target, dim=-1, eps=eps)
        return (1.0 - cos).mean()
    raise ValueError(f"Unsupported symbol loss type: {loss_type}")


class OAMH5Dataset(Dataset):
    def __init__(self, path: Path, meta: DatasetMeta) -> None:
        self.path = Path(path)
        self.meta = meta
        with h5py.File(self.path, "r") as hf:
            self.length = hf["fields"].shape[0]
            self.n_modes = hf["symbols"].shape[1]
            self.symbol_indices = hf["metadata"]["symbol_index"][:]
        self._file = None
        self.pilot_lookup = self._build_pilot_lookup()

    def _build_pilot_lookup(self) -> torch.Tensor:
        pilot_handler = PilotHandler(self.meta.pilot_ratio)
        spacing = max(1, int(round(1.0 / self.meta.pilot_ratio)))
        # Infer number of data symbols (n_total = n_data + preamble + ceil(n_data/spacing))
        n_total = self.meta.frame_length
        preamble = 64
        for n_data in range(1, n_total):
            n_comb = math.ceil(n_data / spacing)
            if n_data + preamble + n_comb == n_total:
                data_len = n_data
                break
        else:
            raise RuntimeError("Unable to reconcile pilot pattern with frame length")
        frame, pilot_pos, _ = pilot_handler.insert_pilots_per_mode(np.zeros(data_len, dtype=np.complex64), (0, 1))
        mask = torch.zeros(len(frame), dtype=torch.float32)
        mask[pilot_pos] = 1.0
        return mask

    def __len__(self) -> int:
        return self.length

    def _ensure_file(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.path, "r")

    def __getstate__(self):
        self.close()
        state = self.__dict__.copy()
        # File handles cannot be pickled; make sure each worker opens its own handle.
        state["_file"] = None
        return state

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_file()
        field = self._file["fields"][idx]
        symbols = self._file["symbols"][idx]
        sym_idx = int(self.symbol_indices[idx])
        field_t = torch.from_numpy(field.transpose(2, 0, 1))  # (2,H,W)
        symbols_t = torch.from_numpy(symbols)
        class_index = _qpsk_symbol_to_index(symbols_t)
        bits = _qpsk_index_to_bits(class_index)
        pilot_mask = torch.full((1, self.meta.grid_size, self.meta.grid_size), self.pilot_lookup[sym_idx])
        return {
            "field": field_t.float(),
            "symbols": symbols_t.float(),
            "class_index": class_index.long(),
            "bits": bits.float(),
            "pilot_mask": pilot_mask.float(),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _load_meta(dataset_path: Path) -> DatasetMeta:
    with h5py.File(dataset_path, "r") as hf:
        cfg = json.loads(hf.attrs["config"])
        spatial_modes = tuple(tuple(mode) for mode in cfg["spatial_modes"])
        wavelength = float(cfg["wavelength_m"])
        w0 = float(cfg["w0_m"])
        distance = float(cfg["distance_m"])
        oversampling = float(cfg["oversampling"])
        downsample = int(cfg["downsample_size"])
        pilot_ratio = float(cfg["pilot_ratio"])
        # Replicate grid extent logic from dataset generator
        max_mode = max(spatial_modes, key=lambda m: abs(m[1]))
        beam = LaguerreGaussianBeam(max_mode[0], max_mode[1], wavelength, w0)
        beam_size = beam.beam_waist(distance)
        grid_extent = oversampling * 6.0 * beam_size
        symbol_index = hf["metadata"]["symbol_index"][:]
        frame_length = int(np.max(symbol_index) + 1)
    return DatasetMeta(
        spatial_modes=spatial_modes,
        wavelength=wavelength,
        w0=w0,
        distance=distance,
        grid_extent=grid_extent,
        grid_size=downsample,
        pilot_ratio=pilot_ratio,
        frame_length=frame_length,
        fec_rate=float(cfg["fec_rate"]),
    )


def _validate_dataset_finite(dataset_path: Path, chunk_size: int = 256) -> None:
    with h5py.File(dataset_path, "r") as hf:
        for key in ("fields", "symbols"):
            if key not in hf:
                continue
            ds = hf[key]
            total = ds.shape[0]
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                batch = ds[start:end]
                if not np.isfinite(batch).all():
                    bad = np.argwhere(~np.isfinite(batch))
                    rel_idx = bad[0]
                    raise ValueError(
                        f"Non-finite value detected in '{key}' at sample {start + rel_idx[0]}, "
                        f"element indices {tuple(rel_idx[1:])}"
                    )
    print(f"[validate] No NaN/Inf detected in '{dataset_path}' (chunk_size={chunk_size}).")


def train_epoch(
    model,
    loader,
    optimiser,
    device: torch.device,
    symbol_weight: float,
    scaler: GradScaler,
    use_amp: bool,
    accum_steps: int,
    ce_loss_fn: nn.Module,
    symbol_loss_type: str,
    bit_loss_weight: float,
    grad_clip: float,
) -> Dict[str, float]:
    if accum_steps < 1:
        raise ValueError("accum_steps must be >= 1")
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_sym = 0.0
    total_bit = 0.0
    count = 0
    progress = tqdm(loader, desc="train", leave=False)
    autocast_enabled = use_amp and device.type in ("cuda", "mps")
    autocast_device = device.type if device.type in ("cuda", "mps") else "cpu"
    num_batches = len(loader)
    optimiser.zero_grad(set_to_none=True)
    nan_detected = False
    for batch_idx, batch in enumerate(progress):
        field = batch["field"].to(device, non_blocking=True)
        pilot_mask = batch["pilot_mask"].to(device, non_blocking=True)
        class_index = batch["class_index"].to(device, non_blocking=True)
        target_symbols = batch["symbols"].to(device, non_blocking=True)
        target_symbols = batch["symbols"].to(device, non_blocking=True)
        target_bits = batch["bits"].to(device, non_blocking=True)
        
        if batch_idx == 0:
             # Log stats for first batch of epoch
             print(f"    [Epoch Start] Field min/max: {field.min():.2e}/{field.max():.2e}")

        with autocast(
            device_type=autocast_device,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            out = model(field, pilot_mask=pilot_mask)
            logits = out["class_logits"].view(-1, model.n_modes, 4)
            ce_loss = ce_loss_fn(logits.view(-1, 4), class_index.view(-1))
            pred_symbol = out["symbol"]
            sym_loss = _symbol_loss(pred_symbol, target_symbols, symbol_loss_type)
            loss = ce_loss + symbol_weight * sym_loss
            if bit_loss_weight > 0:
                bit_loss = F.binary_cross_entropy_with_logits(out["llr"], target_bits)
                loss = loss + bit_loss_weight * bit_loss
            else:
                bit_loss = loss.new_tensor(0.0)

        activations = {
            "class_logits": logits,
            "symbol": pred_symbol,
            "llr": out["llr"],
        }
        for name, tensor in activations.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"[nan] {name} produced NaN/Inf at batch {batch_idx}")
                nan_detected = True
                break
        if nan_detected:
            break

        batch_size = field.size(0)
        total_loss += loss.detach().item() * batch_size
        total_ce += ce_loss.detach().item() * batch_size
        total_sym += sym_loss.detach().item() * batch_size
        total_bit += bit_loss.detach().item() * batch_size
        count += batch_size

        loss_to_backward = loss / accum_steps
        if scaler.is_enabled():
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[nan] loss became NaN/Inf at batch {batch_idx}")
            nan_detected = True
        if nan_detected:
            break

        should_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == num_batches)
        if should_step:
            if grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimiser)
                clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimiser)
                scaler.update()
            else:
                optimiser.step()
            optimiser.zero_grad(set_to_none=True)

        if (batch_idx + 1) % 50 == 0:
            progress.set_postfix(
                loss=total_loss / count,
                ce=total_ce / count,
                sym=total_sym / count,
                bit=total_bit / count if bit_loss_weight > 0 else 0.0,
            )
        if nan_detected:
            break
    if count == 0:
        raise RuntimeError("No samples processed during training epoch (possible early NaN).")
    return {
        "loss": total_loss / count,
        "ce": total_ce / count,
        "symbol": total_sym / count,
        "bit": total_bit / count if bit_loss_weight > 0 else 0.0,
    }


def evaluate(
    model,
    loader,
    device: torch.device,
    symbol_weight: float,
    use_amp: bool,
    ce_loss_fn: nn.Module,
    symbol_loss_type: str,
    bit_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_sym = 0.0
    total_bit = 0.0
    count = 0
    autocast_enabled = use_amp and device.type in ("cuda", "mps")
    autocast_device = device.type if device.type in ("cuda", "mps") else "cpu"
    with torch.no_grad():
        progress = tqdm(loader, desc="val", leave=False)
        for batch in progress:
            field = batch["field"].to(device, non_blocking=True)
            pilot_mask = batch["pilot_mask"].to(device, non_blocking=True)
            class_index = batch["class_index"].to(device, non_blocking=True)
            target_symbols = batch["symbols"].to(device, non_blocking=True)
            target_bits = batch["bits"].to(device, non_blocking=True)
            with autocast(
                device_type=autocast_device,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                out = model(field, pilot_mask=pilot_mask)
                logits = out["class_logits"].view(-1, model.n_modes, 4)
                ce_loss = ce_loss_fn(logits.view(-1, 4), class_index.view(-1))
                sym_loss = _symbol_loss(out["symbol"], target_symbols, symbol_loss_type)
                loss = ce_loss + symbol_weight * sym_loss
                if bit_loss_weight > 0:
                    bit_loss = F.binary_cross_entropy_with_logits(out["llr"], target_bits)
                    loss = loss + bit_loss_weight * bit_loss
                else:
                    bit_loss = loss.new_tensor(0.0)
            batch_size = field.size(0)
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_sym += sym_loss.item() * batch_size
            total_bit += bit_loss.item() * batch_size
            count += batch_size
    return {
        "loss": total_loss / count,
        "ce": total_ce / count,
        "symbol": total_sym / count,
        "bit": total_bit / count if bit_loss_weight > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neural OAM demultiplexer")
    cpu_count = os.cpu_count() or 4
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.h5")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--symbol-weight", type=float, default=0.5, help="Weight for symbol regression term")
    parser.add_argument(
        "--symbol-loss",
        type=str,
        default="mse",
        choices=("mse", "cosine"),
        help="Type of auxiliary symbol loss to use.",
    )
    parser.add_argument(
        "--bit-loss-weight",
        type=float,
        default=0.0,
        help="Optional weight for binary cross-entropy on predicted LLRs.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing applied to the CE loss.",
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    default_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--feature-channels", type=int, default=48)
    parser.add_argument("--transformer-dim", type=int, default=192)
    parser.add_argument("--transformer-heads", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout applied within CNN/MLP heads.")
    parser.add_argument("--transformer-dropout", type=float, default=0.1, help="Dropout inside the transformer encoder.")
    parser.add_argument(
        "--stochastic-depth",
        type=float,
        default=0.1,
        help="Final stochastic depth probability across CNN residual blocks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision (CUDA/MPS).")
    parser.add_argument("--compile", action="store_true", help="Compile the model for graph optimisations.")
    parser.add_argument("--num-workers", type=int, default=max(2, cpu_count - 2))
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Samples prefetched per worker.")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing in the CNN backbone.")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Clip gradients to this max norm before each optimiser step (0 disables).",
    )
    parser.add_argument(
        "--grad-clip-scale-accum",
        action="store_true",
        help="Automatically scale the gradient clip threshold by 1/sqrt(accum-steps).",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch.autograd anomaly detection to pinpoint NaN sources.",
    )
    parser.add_argument(
        "--validate-dataset",
        action="store_true",
        help="Scan dataset.h5 for NaN/Inf before training.",
    )
    parser.add_argument(
        "--validate-chunk-size",
        type=int,
        default=256,
        help="Row chunk size used when scanning dataset for NaN/Inf.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of initial epochs to linearly warm up the learning rate.",
    )
    args = parser.parse_args()
    if args.accum_steps < 1:
        raise ValueError("--accum-steps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.set_float32_matmul_precision("high")

    meta = _load_meta(args.dataset)
    if args.validate_dataset:
        _validate_dataset_finite(args.dataset, chunk_size=args.validate_chunk_size)
    dataset = OAMH5Dataset(args.dataset, meta)
    val_len = int(len(dataset) * args.val_split)
    if args.val_split > 0 and val_len == 0:
        val_len = 1
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError("Validation split too large; no samples left for training.")
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    # --- Sanity Check: Validate one batch ---
    print("[info] Running sanity check on one batch...")
    temp_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(temp_loader))
    f_check = batch["field"]
    print(f"    Input stats: min={f_check.min():.2e}, max={f_check.max():.2e}, mean={f_check.mean():.2e}, std={f_check.std():.2e}")
    if torch.isnan(f_check).any() or torch.isinf(f_check).any():
        raise ValueError("Sanity check failed: Input fields contain NaN/Inf!")
    if f_check.abs().max() < 1e-6:
        print("    [WARNING] Input fields are extremely small. Normalization might be missing!")
    print("[info] Sanity check passed.")

    pin_memory = args.device.startswith("cuda")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    config = NeuralDemuxConfig(
        spatial_modes=meta.spatial_modes,
        wavelength=meta.wavelength,
        w0=meta.w0,
        distance=meta.distance,
        grid_extent=meta.grid_extent,
        grid_size=meta.grid_size,
        pilot_ratio=meta.pilot_ratio,
        feature_channels=args.feature_channels,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        transformer_dropout=args.transformer_dropout,
        stochastic_depth_prob=args.stochastic_depth,
        use_checkpoint=args.grad_checkpoint,
    )
    device = torch.device(args.device)
    if device.type == "mps":
        torch.set_default_dtype(torch.float32)
    model = OAMNeuralDemultiplexer(config).to(device)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print(f"[warn] torch.compile failed: {exc}. Continuing without compilation.")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    main_T = max(1, args.epochs - args.warmup_epochs)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=main_T)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs
        )
        if args.epochs > args.warmup_epochs:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimiser,
                schedulers=[warmup, main_scheduler],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = warmup
    else:
        scheduler = main_scheduler
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, args.epochs + 1):
        grad_clip_value = args.grad_clip
        if grad_clip_value > 0 and args.grad_clip_scale_accum:
            grad_clip_value = grad_clip_value / math.sqrt(args.accum_steps)
            if epoch == 1:
                print(
                    f"[info] Gradient clip scaled by 1/sqrt(accum_steps): original={args.grad_clip:.4f}, "
                    f"effective={grad_clip_value:.4f}"
                )
        train_metrics = train_epoch(
            model,
            train_loader,
            optimiser,
            device,
            args.symbol_weight,
            scaler,
            args.use_amp,
            args.accum_steps,
            ce_loss_fn,
            args.symbol_loss,
            args.bit_loss_weight,
            grad_clip_value,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.symbol_weight,
            args.use_amp,
            ce_loss_fn,
            args.symbol_loss,
            args.bit_loss_weight,
        )
        scheduler.step()
        train_bit = f" bit={train_metrics['bit']:.4f}" if args.bit_loss_weight > 0 else ""
        val_bit = f" bit={val_metrics['bit']:.4f}" if args.bit_loss_weight > 0 else ""
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} ce={train_metrics['ce']:.4f} sym={train_metrics['symbol']:.4f}"
            f"{train_bit} | "
            f"val loss={val_metrics['loss']:.4f} ce={val_metrics['ce']:.4f} sym={val_metrics['symbol']:.4f}"
            f"{val_bit}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt = {
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "config": config,
                "meta": meta.__dict__,
            }
            torch.save(ckpt, args.save_dir / "best_model.pt")

    dataset.close()


if __name__ == "__main__":
    main()
