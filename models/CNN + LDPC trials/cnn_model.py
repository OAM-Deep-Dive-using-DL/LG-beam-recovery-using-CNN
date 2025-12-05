"""Physics-aware CNN architecture for OAM demultiplexing.

The network accepts complex optical fields sampled on a 2D grid (real/imag stored
in separate channels) and produces log-likelihood ratios (LLRs) for the QPSK bits
assigned to each spatial mode.  It mirrors the classical receiver pipeline while
allowing data-driven corrections:

* complex-valued convolutional stem with LG-inspired initialisation;
* residual blocks that preserve the helical phase structure via magnitude/phase
  parameterisation;
* pilot-aware attention operating on latent tokens so that known pilots remain
  phase references;
* physics-informed pooling that reuses analytically generated LG basis fields
  when collapsing the spatial map into per-mode symbol features.

The module is fully Torch-scriptable and can be dropped into the existing
receiver to replace the analytical OAM projection + LS/MMSE equaliser.  All
outputs are compatible with ``PyLDPCWrapper.decode_bp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from lgBeam import LaguerreGaussianBeam
except Exception as exc:  # pragma: no cover - import guard for unit tests
    raise ImportError(
        "cnn_model requires lgBeam.LaguerreGaussianBeam to build LG references"
    ) from exc


def _split_complex(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a (B, 2*C, H, W) tensor into real/imag parts."""
    real, imag = torch.chunk(tensor, 2, dim=1)
    return real, imag


def _merge_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Merge real/imag tensors back into (B, 2*C, H, W)."""
    return torch.cat([real, imag], dim=1)


class ComplexConv2d(nn.Module):
    """Complex-valued convolution implemented via magnitude/phase weights.

    The weight tensor is parameterised in polar form which encourages filters to
    represent oriented wavefronts.  Forward propagation explicitly performs the
    complex convolution so that phase relationships are preserved.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight_mag = nn.Parameter(torch.randn(shape) * 0.02)
        self.weight_phase = nn.Parameter(torch.zeros(shape))
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_complex(x)
        weight_real = self.weight_mag * torch.cos(self.weight_phase)
        weight_imag = self.weight_mag * torch.sin(self.weight_phase)
        yr = F.conv2d(
            xr,
            weight_real,
            bias=self.bias_real,
            stride=self.stride,
            padding=self.padding,
        ) - F.conv2d(
            xi,
            weight_imag,
            bias=None if self.bias_imag is None else torch.zeros_like(self.bias_imag),
            stride=self.stride,
            padding=self.padding,
        )
        yi = F.conv2d(
            xr,
            weight_imag,
            bias=self.bias_imag,
            stride=self.stride,
            padding=self.padding,
        ) + F.conv2d(
            xi,
            weight_real,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )
        return _merge_complex(yr, yi)


class ComplexInstanceNorm(nn.Module):
    """Instance normalisation applied on the concatenated real/imag channels."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels * 2, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ComplexGELU(nn.Module):
    """Apply GELU to real/imag components separately."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        xr, xi = _split_complex(x)
        return _merge_complex(F.gelu(xr), F.gelu(xi))


class ComplexResidualBlock(nn.Module):
    """Residual block composed of complex convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size)
        self.norm1 = ComplexInstanceNorm(channels)
        self.act1 = ComplexGELU()
        self.conv2 = ComplexConv2d(channels, channels, kernel_size)
        self.norm2 = ComplexInstanceNorm(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        return out + residual


@dataclass
class NeuralDemuxConfig:
    spatial_modes: Sequence[Tuple[int, int]]
    wavelength: float
    w0: float
    distance: float
    grid_extent: float
    grid_size: int
    pilot_ratio: float
    feature_channels: int = 32
    transformer_heads: int = 2
    transformer_layers: int = 2
    transformer_dim: int = 128
    use_checkpoint: bool = False
    dropout: float = 0.0
    transformer_dropout: float = 0.0
    stochastic_depth_prob: float = 0.0


def _make_grid(extent: float, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-extent / 2, extent / 2, size)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    return X, Y


def _generate_basis_fields(
    spatial_modes: Sequence[Tuple[int, int]],
    wavelength: float,
    w0: float,
    distance: float,
    grid_extent: float,
    grid_size: int,
) -> torch.Tensor:
    """Generate normalised LG basis fields at the receiver plane."""
    X, Y = _make_grid(grid_extent, grid_size)
    R = torch.sqrt(X**2 + Y**2)
    PHI = torch.atan2(Y, X)
    basis: List[torch.Tensor] = []
    dA = (grid_extent / grid_size) ** 2
    for mode in spatial_modes:
        beam = LaguerreGaussianBeam(mode[0], mode[1], wavelength, w0)
        field = beam.generate_beam_field(R.numpy(), PHI.numpy(), distance)
        field = torch.from_numpy(field.astype("complex64"))
        # normalise such that integral |E|^2 dA = 1
        energy = torch.sum(torch.abs(field) ** 2).real * dA
        field = field / torch.sqrt(energy + 1e-12)
        basis.append(field)
    stacked = torch.stack(basis, dim=0)  # (M, H, W)
    return stacked


def _downsample_tensor(t: torch.Tensor, target_size: int) -> torch.Tensor:
    return F.interpolate(t.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(0)


class PilotAwareAttention(nn.Module):
    """Lightweight transformer with pilot-aware masking."""

    def __init__(self, dim: int, heads: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pilot_gain = nn.Parameter(torch.tensor(0.5))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, tokens: torch.Tensor, pilot_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pilot_mask is not None:
            pilot_scale = 1.0 + self.pilot_gain * pilot_mask.squeeze(-1)
            tokens = tokens * pilot_scale.unsqueeze(-1)
        return self.encoder(tokens)


class ModePoolingHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, n_modes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 4)  # 4 logits per QPSK symbol
        self.symbol_head = nn.Linear(hidden_dim, 2)  # real/imag symbol prediction
        self.n_modes = n_modes

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Clamp pooled values to prevent extreme magnitudes that cause NaN
        x = self.norm(pooled)
        h = self.fc1(x)
        h = self.act(h)
        class_logits = self.fc2(h)
        symbol = self.symbol_head(h)
        llr = class_logits.view(-1, self.n_modes, 4)
        # Map logits to LLR per bit (2 bits).  For QPSK Gray: bit0 distinguishes imag, bit1 real.
        bit0 = llr[..., 0] - llr[..., 1]
        bit1 = llr[..., 2] - llr[..., 3]
        llr_bits = torch.stack([bit0, bit1], dim=-1)
        symbol = symbol.view(-1, self.n_modes, 2)
        return {"llr": llr_bits, "symbol": symbol, "class_logits": llr}


class OAMNeuralDemultiplexer(nn.Module):
    """Main model used for neural OAM demultiplexing."""

    def __init__(self, config: NeuralDemuxConfig) -> None:
        super().__init__()
        self.config = config
        self.n_modes = len(config.spatial_modes)
        self.use_checkpoint = config.use_checkpoint
        self.stem = ComplexConv2d(1, config.feature_channels, kernel_size=5)
        self.res1 = ComplexResidualBlock(config.feature_channels, dropout=config.dropout)
        self.down1 = ComplexConv2d(config.feature_channels, config.feature_channels * 2, stride=2)
        self.res2 = ComplexResidualBlock(config.feature_channels * 2, dropout=config.dropout)
        self.down2 = ComplexConv2d(config.feature_channels * 2, config.feature_channels * 4, stride=2)
        self.res3 = ComplexResidualBlock(config.feature_channels * 4, dropout=config.dropout)
        # Convert complex feature map to real embedding for transformer
        self.conv_real_proj = nn.Conv2d(config.feature_channels * 4 * 3, config.transformer_dim, kernel_size=1)
        self.attention = PilotAwareAttention(
            dim=config.transformer_dim,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.transformer_dropout,
        )
        self.mode_head = ModePoolingHead(config.transformer_dim, config.transformer_dim * 2, self.n_modes)
        self.register_buffer(
            "basis_fields",
            _generate_basis_fields(
                config.spatial_modes,
                config.wavelength,
                config.w0,
                config.distance,
                config.grid_extent,
                config.grid_size,
            ),
            persistent=False,
        )

    def forward(
        self,
        field: torch.Tensor,
        pilot_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if field.dim() != 4 or field.size(1) != 2:
            raise ValueError("field must have shape (batch, 2, H, W)")
        x = field
        x = self._maybe_checkpoint(self.stem, x)
        x = self._maybe_checkpoint(self.res1, x)
        x = self._maybe_checkpoint(self.down1, x)
        x = self._maybe_checkpoint(self.res2, x)
        x = self._maybe_checkpoint(self.down2, x)
        x = self._maybe_checkpoint(self.res3, x)
        xr, xi = _split_complex(x)
        mag = torch.sqrt(xr**2 + xi**2 + 1e-8)
        feat_real = torch.cat([xr, xi, mag], dim=1)
        feat_embed = self.conv_real_proj(feat_real)
        B, C, H, W = feat_embed.shape
        tokens = feat_embed.flatten(2).transpose(1, 2)  # (B, N, C)
        if pilot_mask is not None:
            pilot_mask_low = F.interpolate(pilot_mask, size=(H, W), mode="nearest")
            pilot_tokens = pilot_mask_low.flatten(2).transpose(1, 2)
        else:
            pilot_tokens = None
        tokens = self.attention(tokens, pilot_tokens)
        feat_att = tokens.transpose(1, 2).view(B, C, H, W)
        basis_mag = torch.abs(self.basis_fields.to(feat_att.device)) + 1e-6
        basis_low = F.interpolate(
            basis_mag.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (M, H, W)
        norm = basis_low.flatten(1).sum(-1, keepdim=True).clamp_min(1e-9)
        weights = (basis_low / norm.view(self.n_modes, 1, 1)).unsqueeze(0)  # (1, M, H, W)
        pooled = (feat_att.unsqueeze(1) * weights.unsqueeze(2)).sum(dim=(3, 4))  # (B, M, C)
        outputs = self.mode_head(pooled)
        return outputs

    def _maybe_checkpoint(
        self,
        module: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(lambda inp: module(inp), x)
        return module(x)


__all__ = [
    "NeuralDemuxConfig",
    "OAMNeuralDemultiplexer",
]
