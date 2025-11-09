"""Neural receiver drop-in replacement for :mod:`receiver`.

The class defined here mirrors the interface of ``FSORx`` enough to be swapped
into ``pipeline.py``.  It loads a trained checkpoint produced by
``train_cnn.py`` and uses the neural demultiplexer to generate LLRs which are
then fed through the familiar LDPC decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, List

import contextlib
import io
import numpy as np
import torch

from encoding import PilotHandler, PyLDPCWrapper
from cnn_model import NeuralDemuxConfig, OAMNeuralDemultiplexer


@dataclass
class CheckpointMeta:
    spatial_modes: Tuple[Tuple[int, int], ...]
    wavelength: float
    w0: float
    distance: float
    grid_extent: float
    grid_size: int
    pilot_ratio: float
    frame_length: int
    fec_rate: float

    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointMeta":
        return cls(
            spatial_modes=tuple(tuple(m) for m in data["spatial_modes"]),
            wavelength=float(data["wavelength"]),
            w0=float(data["w0"]),
            distance=float(data["distance"]),
            grid_extent=float(data["grid_extent"]),
            grid_size=int(data["grid_size"]),
            pilot_ratio=float(data["pilot_ratio"]),
            frame_length=int(data["frame_length"]),
            fec_rate=float(data["fec_rate"]),
        )


def _stride_downsample(field: np.ndarray, target_size: int) -> np.ndarray:
    if field.shape[0] % target_size != 0:
        raise ValueError("input field size must be divisible by target size")
    factor = field.shape[0] // target_size
    reshaped = field.reshape(target_size, factor, target_size, factor)
    return reshaped.mean(axis=(1, 3))


class NeuralFSORx:
    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[torch.device] = None,
        ldpc: Optional[PyLDPCWrapper] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config: NeuralDemuxConfig = ckpt["config"]
        self.meta = CheckpointMeta.from_dict(ckpt["meta"])
        self.model = OAMNeuralDemultiplexer(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.n_modes = len(self.config.spatial_modes)
        self.pilot_handler = PilotHandler(self.meta.pilot_ratio)
        self.pilot_mask = self._build_pilot_mask()
        if ldpc is not None:
            self.ldpc = ldpc
        else:
            # Construct a matching LDPC wrapper using encodingRunner parameters.
            from encoding import encodingRunner
            with contextlib.redirect_stdout(io.StringIO()):
                runner = encodingRunner(
                    spatial_modes=list(self.meta.spatial_modes),
                    wavelength=self.meta.wavelength,
                    w0=self.meta.w0,
                    fec_rate=self.meta.fec_rate,
                    pilot_ratio=self.meta.pilot_ratio,
                )
            self.ldpc = runner.ldpc

    def _build_pilot_mask(self) -> torch.Tensor:
        spacing = max(1, int(round(1.0 / self.meta.pilot_ratio)))
        frame_length = self.meta.frame_length
        preamble = 64
        for n_data in range(1, frame_length):
            n_comb = int(np.ceil(n_data / spacing))
            if n_data + preamble + n_comb == frame_length:
                data_len = n_data
                break
        else:
            raise RuntimeError("Unable to match pilot structure in checkpoint meta")
        frame, pilot_pos, _ = self.pilot_handler.insert_pilots_per_mode(
            np.zeros(data_len, dtype=np.complex64), (0, 1)
        )
        mask = torch.zeros(len(frame), dtype=torch.float32)
        mask[pilot_pos] = 1.0
        return mask

    def _prepare_field(self, field: np.ndarray) -> torch.Tensor:
        field_ds = _stride_downsample(field, self.meta.grid_size)
        stacked = np.stack([field_ds.real, field_ds.imag], axis=0)
        return torch.from_numpy(stacked).float()

    def receive_sequence(
        self,
        E_rx_sequence: Iterable[np.ndarray],
        original_data_bits: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        fields = torch.stack([self._prepare_field(np.asarray(f)) for f in E_rx_sequence])
        pilot_mask_values = self.pilot_mask[: fields.size(0)]
        pilot_mask = torch.stack(
            [
                torch.full((1, self.meta.grid_size, self.meta.grid_size), pilot_mask_values[i])
                for i in range(fields.size(0))
            ]
        )
        with torch.no_grad():
            outputs = self.model(
                fields.to(self.device),
                pilot_mask=pilot_mask.to(self.device),
            )
        llr = outputs["llr"].cpu()  # (B, modes, 2)
        data_indices = (pilot_mask_values == 0).nonzero(as_tuple=False).squeeze(-1)
        llr_data = llr[data_indices]
        llr_np = llr_data.reshape(-1, self.n_modes * 2).numpy().reshape(-1)
        decoded = self.ldpc.decode_bp(llr_np)
        decoded = decoded[: len(original_data_bits)]
        errors = np.count_nonzero(decoded != original_data_bits[: len(decoded)])
        ber = errors / max(1, len(original_data_bits))
        metrics = {
            "ber": float(ber),
            "symbols": int(len(data_indices)),
        }
        return decoded, metrics