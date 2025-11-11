from __future__ import annotations

import argparse
import json
import math
import typing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from cnn_model import NeuralDemuxConfig, OAMNeuralDemultiplexer
from train_cnn import OAMH5Dataset, _load_meta, _qpsk_index_to_bits


matplotlib.use("Agg")
add_safe_globals([NeuralDemuxConfig])


def _format_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _prepare_loaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    lengths = [val_split, test_split, 1.0 - val_split - test_split]
    total = len(dataset)
    split_lengths = [math.floor(total * frac) for frac in lengths[:2]]
    split_lengths.append(total - sum(split_lengths))

    generator = torch.Generator().manual_seed(seed)
    val_subset, test_subset, _ = random_split(dataset, split_lengths, generator=generator)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = min(4, batch_size)

    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)
    return val_loader, test_loader


@torch.no_grad()
def _evaluate_split(
    model: OAMNeuralDemultiplexer,
    loader: DataLoader,
    device: torch.device,
    symbol_weight: float,
) -> Dict[str, object]:
    metrics: Dict[str, float] = {
        "loss": 0.0,
        "ce": 0.0,
        "symbol_mse": 0.0,
        "accuracy": 0.0,
    }
    total_samples = 0
    total_bit_errors = 0
    total_bits = 0
    preds_per_mode: List[List[int]] = []
    targets_per_mode: List[List[int]] = []
    pred_symbols: List[List[np.ndarray]] = []
    true_symbols: List[List[np.ndarray]] = []
    per_mode_bit_errors: typing.Optional[np.ndarray] = None
    per_mode_bit_totals: typing.Optional[np.ndarray] = None

    progress = tqdm(loader, desc="eval", leave=False)
    for batch in progress:
        field = batch["field"].to(device)
        pilot_mask = batch["pilot_mask"].to(device)
        target_indices = batch["class_index"].to(device)
        target_symbols = batch["symbols"].to(device)
        target_bits = batch["bits"].to(device)

        out = model(field, pilot_mask=pilot_mask)
        logits = out["class_logits"].view(field.size(0), model.n_modes, 4)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 4), target_indices.view(-1)
        )
        mse_loss = torch.nn.functional.mse_loss(out["symbol"], target_symbols)
        loss = ce_loss + mse_loss * symbol_weight

        pred_idx = logits.argmax(dim=-1)
        correct = (pred_idx == target_indices).float().sum()
        pred_bits = _qpsk_index_to_bits(pred_idx).to(device)
        pred_bits_int = pred_bits.to(torch.int64)
        target_bits_int = target_bits.to(torch.int64)
        bit_errors = (pred_bits_int != target_bits_int)

        batch_size = field.size(0)
        metrics["loss"] += loss.item() * batch_size
        metrics["ce"] += ce_loss.item() * batch_size
        metrics["symbol_mse"] += mse_loss.item() * batch_size
        metrics["accuracy"] += correct.item()
        total_samples += batch_size
        total_bit_errors += bit_errors.sum().item()
        total_bits += target_bits_int.numel()

        if not preds_per_mode:
            preds_per_mode = [[] for _ in range(model.n_modes)]
            targets_per_mode = [[] for _ in range(model.n_modes)]
            pred_symbols = [[] for _ in range(model.n_modes)]
            true_symbols = [[] for _ in range(model.n_modes)]
            per_mode_bit_errors = np.zeros(model.n_modes, dtype=np.float64)
            per_mode_bit_totals = np.zeros(model.n_modes, dtype=np.float64)

        pred_cpu = pred_idx.cpu().numpy()
        target_cpu = target_indices.cpu().numpy()
        sym_pred_cpu = out["symbol"].cpu().numpy()
        sym_true_cpu = target_symbols.cpu().numpy()
        bit_errors_per_mode = bit_errors.sum(dim=2).cpu().numpy()

        for mode in range(model.n_modes):
            preds_per_mode[mode].extend(pred_cpu[:, mode].tolist())
            targets_per_mode[mode].extend(target_cpu[:, mode].tolist())
            pred_symbols[mode].append(sym_pred_cpu[:, mode, :])
            true_symbols[mode].append(sym_true_cpu[:, mode, :])
            if per_mode_bit_errors is not None and per_mode_bit_totals is not None:
                per_mode_bit_errors[mode] += float(bit_errors_per_mode[:, mode].sum())
                per_mode_bit_totals[mode] += float(batch_size * bit_errors.size(2))

    if total_samples == 0:
        raise RuntimeError("Empty split encountered during evaluation.")

    for key in ("loss", "ce", "symbol_mse"):
        metrics[key] /= total_samples
    metrics["accuracy"] /= (total_samples * model.n_modes)
    if total_bits > 0:
        metrics["ber"] = total_bit_errors / total_bits
    else:
        metrics["ber"] = float("nan")

    confusion: List[np.ndarray] = []
    per_mode_accuracy: List[float] = []
    per_mode_mse: List[float] = []
    per_mode_ber: List[float] = []
    symbol_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for mode in range(model.n_modes):
        preds = np.asarray(preds_per_mode[mode], dtype=np.int64)
        targets = np.asarray(targets_per_mode[mode], dtype=np.int64)
        confusion.append(confusion_matrix(targets, preds, labels=np.arange(4)))
        per_mode_accuracy.append(float((preds == targets).mean()))
        pred_sym = np.concatenate(pred_symbols[mode], axis=0)
        true_sym = np.concatenate(true_symbols[mode], axis=0)
        per_mode_mse.append(float(np.mean((pred_sym - true_sym) ** 2)))
        symbol_pairs.append((pred_sym, true_sym))
        if per_mode_bit_errors is not None and per_mode_bit_totals is not None and per_mode_bit_totals[mode] > 0:
            per_mode_ber.append(float(per_mode_bit_errors[mode] / per_mode_bit_totals[mode]))
        else:
            per_mode_ber.append(float("nan"))

    return {
        "metrics": metrics,
        "confusion": confusion,
        "per_mode_accuracy": per_mode_accuracy,
        "per_mode_symbol_mse": per_mode_mse,
        "per_mode_bit_ber": per_mode_ber,
        "symbols": symbol_pairs,
    }


def _plot_confusion(
    matrix: np.ndarray,
    mode: Tuple[int, int],
    split: str,
    out_dir: Path,
) -> None:
    norm = matrix.sum(axis=1, keepdims=True)
    norm[norm == 0] = 1
    matrix_norm = matrix / norm
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(matrix_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Predicted index")
    ax.set_ylabel("True index")
    ax.set_title(f"{split} confusion | mode {mode}")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:d}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalised")
    fig.tight_layout()
    out_path = out_dir / f"{split}_confusion_mode_{mode[0]}_{mode[1]}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_symbol_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    mode: Tuple[int, int],
    split: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(true[:, 0], true[:, 1], s=8, alpha=0.6, label="True")
    ax.scatter(pred[:, 0], pred[:, 1], s=8, alpha=0.6, label="Pred")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(f"{split} symbols | mode {mode}")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = out_dir / f"{split}_symbols_mode_{mode[0]}_{mode[1]}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_encoder) + "\n", encoding="utf-8")


def _json_encoder(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained neural demultiplexer.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.h5")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Directory where evaluation artifacts are stored.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where PNG plots are written.",
    )
    parser.add_argument(
        "--symbol-weight",
        type=float,
        default=0.2,
        help="Symbol regression weight used during training.",
    )
    parser.add_argument(
        "--commit-hash",
        type=str,
        default="unknown",
        help="Optional commit hash recorded in the evaluation metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    meta = _load_meta(args.dataset)
    dataset = OAMH5Dataset(args.dataset, meta)
    pin_memory = device.type == "cuda"
    val_loader, test_loader = _prepare_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model = OAMNeuralDemultiplexer(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    run_id = f"eval_{_format_timestamp()}"
    run_dir = args.output_root / run_id
    plots_dir = args.plots_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as hf:
        dataset_uuid = hf.attrs.get("dataset_uuid", "unknown")

    splits = {
        "val": _evaluate_split(model, val_loader, device, args.symbol_weight),
        "test": _evaluate_split(model, test_loader, device, args.symbol_weight),
    }

    summary = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": str(args.dataset),
        "dataset_uuid": dataset_uuid,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "val_loss_best": float(ckpt.get("val_loss", float("nan"))),
        "commit_hash": args.commit_hash,
        "symbol_weight": args.symbol_weight,
        "device": device.type,
        "meta": meta.__dict__,
        "config": ckpt["config"].__dict__,
        "splits": {},
    }

    for split_name, result in splits.items():
        metrics = result["metrics"]
        metrics["per_mode_accuracy"] = result["per_mode_accuracy"]
        metrics["per_mode_symbol_mse"] = result["per_mode_symbol_mse"]
        metrics["per_mode_bit_ber"] = result["per_mode_bit_ber"]
        summary["splits"][split_name] = metrics

        for mode_idx, mode in enumerate(meta.spatial_modes):
            _plot_confusion(result["confusion"][mode_idx], mode, split_name, plots_dir)
            pred_sym, true_sym = result["symbols"][mode_idx]
            _plot_symbol_scatter(pred_sym, true_sym, mode, split_name, plots_dir)

    _save_json(run_dir / "metrics.json", summary)
    dataset.close()


if __name__ == "__main__":
    main()
