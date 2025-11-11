from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import h5py
import torch
from torch.serialization import add_safe_globals

from cnn_model import NeuralDemuxConfig
from train_cnn import _load_meta


add_safe_globals([NeuralDemuxConfig])


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _normalise(obj: Any) -> Any:
    if is_dataclass(obj):
        return _normalise(asdict(obj))
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _sha256_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained neural demultiplexer checkpoint with metadata."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.h5 used for training.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory where the versioned bundle will be written.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional tag to include in the bundle name.")
    parser.add_argument(
        "--commit-hash",
        type=str,
        default=os.environ.get("GIT_COMMIT", "unknown"),
        help="Commit hash to embed in the metadata.",
    )
    parser.add_argument(
        "--symbol-weight",
        type=float,
        default=0.2,
        help="Symbol regression weight used during training.",
    )
    parser.add_argument(
        "--extra-note",
        type=str,
        default="",
        help="Optional free-form note recorded alongside the export.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a .zip archive of the export directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict: Dict[str, torch.Tensor] = checkpoint["model_state"]
    config: NeuralDemuxConfig = checkpoint["config"]
    dataset_meta = _load_meta(args.dataset)

    with h5py.File(args.dataset, "r") as hf:
        dataset_uuid = hf.attrs.get("dataset_uuid", "unknown")
        dataset_samples = int(hf["fields"].shape[0])

    bundle_id = args.tag or _timestamp()
    export_root = args.output_dir / f"bundle_{bundle_id}"
    export_root.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "bundle_id": bundle_id,
        "export_timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint_source": str(args.checkpoint),
        "dataset": {
            "path": str(args.dataset),
            "uuid": dataset_uuid,
            "num_samples": dataset_samples,
            "meta": _normalise(dataset_meta),
        },
        "model": {
            "config": _normalise(config),
            "parameter_count": int(sum(t.numel() for t in state_dict.values())),
            "state_sha256": _sha256_state_dict(state_dict),
        },
        "training": {
            "epoch": int(checkpoint.get("epoch", -1)),
            "val_loss": float(checkpoint.get("val_loss", float("nan"))),
            "symbol_weight": args.symbol_weight,
        },
        "environment": {
            "torch_version": torch.__version__,
            "python_version": tuple(os.sys.version_info[:3]),
        },
        "commit_hash": args.commit_hash,
        "extra_note": args.extra_note,
    }

    metadata_path = export_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    bundle = {
        "model_state": state_dict,
        "config": config,
        "meta": checkpoint.get("meta", {}),
        "training": metadata["training"],
        "dataset_uuid": dataset_uuid,
    }
    torch.save(bundle, export_root / "bundle.pt")

    if args.zip:
        import shutil

        archive_path = shutil.make_archive(str(export_root), "zip", root_dir=export_root)
        print(f"Created archive: {archive_path}")

    print(f"Exported bundle to {export_root}")
    print(f"  metadata : {metadata_path}")
    print(f"  weights  : {export_root / 'bundle.pt'}")


if __name__ == "__main__":
    main()

