from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.serialization import add_safe_globals

from cnn_model import NeuralDemuxConfig, OAMNeuralDemultiplexer
from train_cnn import OAMH5Dataset, _load_meta


add_safe_globals([NeuralDemuxConfig])


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _load_weights(path: Path, device: torch.device) -> Dict[str, Any]:
    bundle = torch.load(path, map_location=device, weights_only=False)
    if "model_state" in bundle and "config" in bundle:
        return bundle
    raise ValueError(f"Unrecognised checkpoint format: {path}")


def _select_indices(length: int, indices: Sequence[int] | None, num_samples: int, seed: int) -> List[int]:
    if indices:
        unique = sorted({idx for idx in indices if 0 <= idx < length})
        if not unique:
            raise ValueError("Provided indices are out of range after filtering.")
        return unique
    num_samples = min(num_samples, length)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(length, size=num_samples, replace=False).tolist())


def _tensor_to_list(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity-check inference on selected samples.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.h5")
    parser.add_argument("--weights", type=Path, required=True, help="Path to bundle.pt or checkpoint .pt")
    parser.add_argument("--output-dir", type=Path, default=Path("inference"), help="Root directory for outputs.")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit sample indices to evaluate.")
    parser.add_argument("--num-samples", type=int, default=5, help="Random samples to draw when --indices is not set.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed used for sampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    meta = _load_meta(args.dataset)
    dataset = OAMH5Dataset(args.dataset, meta)
    indices = _select_indices(len(dataset), args.indices, args.num_samples, args.seed)

    weights = _load_weights(args.weights, device)
    model = OAMNeuralDemultiplexer(weights["config"]).to(device)
    model.eval()
    model.load_state_dict(weights["model_state"])

    run_id = f"infer_{_timestamp()}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    correct_count = 0
    total = 0

    for idx in indices:
        sample = dataset[idx]
        field = sample["field"].unsqueeze(0).to(device)
        pilot_mask = sample["pilot_mask"].unsqueeze(0).to(device)
        true_indices = sample["class_index"]
        true_symbols = sample["symbols"]

        with torch.no_grad():
            output = model(field, pilot_mask=pilot_mask)
            logits = output["class_logits"].view(model.n_modes, 4)
            probs = torch.softmax(logits, dim=-1)
            pred_indices = probs.argmax(dim=-1).cpu()

        match = (pred_indices == true_indices).sum().item()
        correct_count += match
        total += model.n_modes

        result = {
            "sample_index": idx,
            "pred_class_index": pred_indices.tolist(),
            "true_class_index": true_indices.tolist(),
            "per_mode_match": (pred_indices == true_indices).tolist(),
            "pred_symbol": _tensor_to_list(output["symbol"].squeeze(0)),
            "true_symbol": _tensor_to_list(true_symbols),
            "logits": _tensor_to_list(logits),
            "probabilities": _tensor_to_list(probs),
        }
        results.append(result)

    summary = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "indices": indices,
        "n_samples": len(indices),
        "modes_per_sample": model.n_modes,
        "overall_accuracy": correct_count / total if total > 0 else 0.0,
    }

    report = {
        "summary": summary,
        "results": results,
    }

    output_path = run_dir / "predictions.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Inference complete. Saved predictions to {output_path}")

    dataset.close()


if __name__ == "__main__":
    main()

