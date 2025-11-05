"""
encoding.py -- FSO-MDM Encoding & Transmitter (rectified)

Requirements (recommended):
    pip install pyldpc matplotlib numpy scipy

Notes:
 - Requires lgBeam.py (LaguerreGaussianBeam) for spatial field generation.
 - If pyldpc is missing, the module raises a clear error when trying to construct the LDPC wrapper.
"""
import os
import json
import gc
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List

import numpy as np
import matplotlib
# If running headless uncomment:
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import erfc
import ast

# Attempt lgBeam import (user-supplied)
try:
    from lgBeam import LaguerreGaussianBeam
except Exception as e:
    LaguerreGaussianBeam = None
    warnings.warn(f"Could not import lgBeam.LaguerreGaussianBeam: {e}. Spatial field generation will fail if attempted.")

# pyldpc optional import
_HAS_PYLDPC = True
try:
    from pyldpc import make_ldpc, encode as pyldpc_encode, decode as pyldpc_decode, get_message as pyldpc_get_message
except Exception:
    _HAS_PYLDPC = False
    warnings.warn("pyldpc not found. Install via `pip install pyldpc` to enable LDPC FEC.")


# ---------- Utilities ----------
def _closest_divisor_leq(n: int, target: int) -> int:
    if target <= 1:
        return 1
    for d in range(min(target, n), 1, -1):
        if n % d == 0:
            return d
    return 1

def _sha32_seed_from_tuple(t: Tuple[int, int]) -> int:
    h = hashlib.sha1(f"{t[0]},{t[1]}".encode("utf-8")).digest()[:4]
    return int.from_bytes(h, "big") % (2 ** 32)

def normalize_bits(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits)
    arr = np.mod(arr.astype(int), 2)
    return arr


# ---------- Frame dataclass ----------
@dataclass
class FSO_MDM_Frame:
    tx_signals: Dict[Tuple[int, int], Dict[str, Any]]
    multiplexed_field: Optional[np.ndarray] = None
    grid_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.tx_signals or len(self.tx_signals) == 0:
            raise ValueError("tx_signals must be a non-empty dict")
        if self.metadata is None:
            self.metadata = {}
        self.metadata.setdefault("n_modes", len(self.tx_signals))
        self.metadata.setdefault("spatial_modes", list(self.tx_signals.keys()))
        lengths = [v.get("n_symbols", 0) for v in self.tx_signals.values()]
        self.metadata.setdefault("total_symbols_per_mode", int(np.max(lengths)) if lengths else 0)
        first = next(iter(self.tx_signals.values()))
        self.metadata.setdefault("pilot_positions", first.get("pilot_positions", []))

    def to_dict(self, save_fields: bool = False):
        def convert(obj):
            if obj is None:
                return None
            if isinstance(obj, complex):
                return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
            if isinstance(obj, np.ndarray):
                if save_fields:
                    return obj.tolist()
                return None
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            return obj

        serial = {"tx_signals": {}, "multiplexed_field": convert(self.multiplexed_field) if save_fields else None,
                  "grid_info": convert(self.grid_info), "metadata": convert(self.metadata)}
        for k, v in self.tx_signals.items():
            serial["tx_signals"][str(k)] = {sk: convert(sv) for sk, sv in v.items() if sk != "beam" or save_fields}
        return serial

    @classmethod
    def from_dict(cls, d):
        def recon(obj):
            if isinstance(obj, dict) and "real" in obj and "imag" in obj:
                return complex(obj["real"], obj["imag"])
            if isinstance(obj, dict):
                return {k: recon(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return np.array([recon(x) for x in obj])
            return obj

        tx = {}
        for ks, v in d.get("tx_signals", {}).items():
            try:
                key = ast.literal_eval(ks)
                key = tuple(key) if not isinstance(key, tuple) else key
            except Exception:
                parts = ks.strip("()[] ").split(",")
                key = (int(parts[0].strip()), int(parts[1].strip()))
            tx[key] = recon(v)
        mf = recon(d.get("multiplexed_field")) if d.get("multiplexed_field") is not None else None
        grid = recon(d.get("grid_info")) if d.get("grid_info") is not None else None
        meta = recon(d.get("metadata", {}))
        return cls(tx_signals=tx, multiplexed_field=mf, grid_info=grid, metadata=meta)


# ---------- QPSK ----------
class QPSKModulator:
    def __init__(self, symbol_energy=1.0):
        self.Es = float(symbol_energy)
        self.A = np.sqrt(self.Es)
        self.constellation_map = {
            (0, 0): self.A * (1 + 1j) / np.sqrt(2),
            (0, 1): self.A * (-1 + 1j) / np.sqrt(2),
            (1, 1): self.A * (-1 - 1j) / np.sqrt(2),
            (1, 0): self.A * (1 - 1j) / np.sqrt(2),
        }
        self.constellation_points = np.array(list(self.constellation_map.values()))
        self.bits_list = list(self.constellation_map.keys())

    def modulate(self, bits):
        bits = normalize_bits(np.asarray(bits, dtype=int))
        if bits.size % 2 != 0:
            bits = np.concatenate([bits, np.array([0], dtype=int)])
        pairs = bits.reshape(-1, 2)
        symbols = np.array([self.constellation_map[tuple(p)] for p in pairs], dtype=complex)
        return symbols

    def demodulate_hard(self, rx_symbols):
        s = np.asarray(rx_symbols, dtype=complex)
        bits = []
        for x in s:
            d = np.abs(self.constellation_points - x)
            idx = np.argmin(d)
            bits.extend(self.bits_list[idx])
        return normalize_bits(np.array(bits, dtype=int))

    def demodulate_soft(self, rx_symbols, noise_var):
        rx = np.asarray(rx_symbols, dtype=complex)
        a_scale = 2.0 * np.sqrt(2.0 * self.Es) / (noise_var + 1e-12)
        llr_I = a_scale * np.real(rx)
        llr_Q = a_scale * np.imag(rx)
        llrs = np.empty(2 * len(rx), dtype=float)
        llrs[0::2] = llr_I
        llrs[1::2] = llr_Q
        return llrs

    def plot_constellation(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        for bits, s in self.constellation_map.items():
            ax.plot(s.real, s.imag, "ro")
            ax.annotate(f"{bits[0]}{bits[1]}", (s.real, s.imag), fontsize=12)
        ax.axhline(0, color="grey"); ax.axvline(0, color="grey")
        ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_title("QPSK (Gray)")
        ax.grid(True); ax.axis("equal")
        return ax


# ---------- pyldpc wrapper ----------
class PyLDPCWrapper:
    def __init__(self, n=2048, rate=0.8, dv=2, dc=8, seed=42):
        if not _HAS_PYLDPC:
            raise RuntimeError("pyldpc not installed. Install: pip install pyldpc")
        self._n_req = int(n)
        self._requested_rate = float(rate)
        self.dv = int(dv)
        # adjust dc to divide n
        dc_actual = _closest_divisor_leq(self._n_req, int(dc))
        if dc_actual < 2:
            dc_actual = 2
        if dc_actual != int(dc):
            warnings.warn(f"Adjusted dc {dc} -> {dc_actual} so that n%dc==0 for n={self._n_req}.")
        self.dc = dc_actual
        self.seed = int(seed)

        # create H,G robustly
        try:
            H, G = make_ldpc(self._n_req, self.dv, self.dc, systematic=True, sparse=True, seed=self.seed)
        except TypeError:
            H, G = make_ldpc(self._n_req, self.dv, self.dc, systematic=True, seed=self.seed)

        self.H = H
        self.G = G

        # infer dimensions robustly
        try:
            gshape = np.array(self.G.shape)
            if gshape.size == 2:
                # If G is (k, n) -> rows=k, cols=n
                # If G is (n, k) for some versions, infer and transpose for consistency
                if gshape[0] <= gshape[1]:
                    self.k = int(gshape[0])
                    self.n = int(gshape[1])
                else:
                    # G is (n,k) -> transpose to (k,n)
                    self.G = self.G.T
                    self.k = int(self.G.shape[0])
                    self.n = int(self.G.shape[1])
                self.m = self.n - self.k
            else:
                raise ValueError("Unexpected G shape returned by pyldpc")
        except Exception as e:
            warnings.warn(f"Failed to infer k/n from G: {e}; falling back to requested rate.")
            self.n = self._n_req
            self.k = int(round(self.n * self._requested_rate))
            self.m = self.n - self.k

        if not (0 < self.k < self.n):
            raise ValueError(f"Invalid inferred LDPC dimensions: k={self.k}, n={self.n}")

    @property
    def rate(self):
        return float(self.k) / float(self.n)

    def encode(self, info_bits):
        info_bits = normalize_bits(np.asarray(info_bits, dtype=int))
        if info_bits.size == 0:
            return np.array([], dtype=int)
        num_blocks = int(np.ceil(info_bits.size / self.k))
        pad_len = num_blocks * self.k - info_bits.size
        if pad_len > 0:
            info_p = np.concatenate([info_bits, np.zeros(pad_len, dtype=int)])
        else:
            info_p = info_bits.copy()
        codewords = []
        for b in range(num_blocks):
            u = info_p[b * self.k : (b + 1) * self.k]
            # pyldpc_encode may require different signatures; try common forms
            cw = None
            try:
                cw = pyldpc_encode(self.G.T, u, snr=1e6)
            except TypeError:
                try:
                    cw = pyldpc_encode(self.G.T, u)
                except Exception as e:
                    warnings.warn(f"pyldpc_encode failed for block {b}: {e}; using zeros.")
                    cw = np.zeros(self.n, dtype=int)
            cw = normalize_bits(np.asarray(cw, dtype=int))
            codewords.append(cw)
        coded = np.concatenate(codewords).astype(int)
        # If padding was added, coded length must be num_blocks * n; that's fine.
        return coded

    def decode_bp(self, llrs, max_iter=50):
        llrs = np.asarray(llrs, dtype=float)
        if llrs.size == 0:
            return np.array([], dtype=int)
        num_blocks = llrs.size // self.n
        recovered = []
        for b in range(num_blocks):
            block_llr = llrs[b * self.n : (b + 1) * self.n]
            try:
                x_hat = pyldpc_decode(self.H, block_llr, maxiter=max_iter)
            except TypeError:
                x_hat = pyldpc_decode(self.H, block_llr, maxiter=max_iter, decode="sumproduct")
            try:
                u_hat = pyldpc_get_message(self.G, x_hat)
            except Exception:
                u_hat = x_hat[: self.k]
            recovered.append(normalize_bits(np.asarray(u_hat, dtype=int)))
        if recovered:
            return np.concatenate(recovered)
        return np.array([], dtype=int)

    def decode_hard(self, received_bits):
        r = normalize_bits(np.asarray(received_bits, dtype=int))
        if r.size == 0:
            return np.array([], dtype=int)
        num_blocks = r.size // self.n
        recovered = []
        for b in range(num_blocks):
            block = r[b * self.n : (b + 1) * self.n]
            llr = (1 - 2 * block) * 1e6
            try:
                x_hat = pyldpc_decode(self.H, llr, maxiter=10)
            except TypeError:
                x_hat = pyldpc_decode(self.H, llr, maxiter=10, decode="sumproduct")
            try:
                u_hat = pyldpc_get_message(self.G, x_hat)
            except Exception:
                u_hat = x_hat[: self.k]
            recovered.append(normalize_bits(np.asarray(u_hat, dtype=int)))
        if recovered:
            return np.concatenate(recovered)
        return np.array([], dtype=int)


# ---------- Pilot Handler ----------
class PilotHandler:
    def __init__(self, pilot_ratio=0.1, pattern="uniform"):
        if not (0.0 < pilot_ratio < 1.0):
            raise ValueError("pilot_ratio must be in (0,1)")
        self.pilot_ratio = float(pilot_ratio)
        self.pattern = pattern
        self.qpsk_constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        self.pilot_sequence = None
        self.pilot_positions = None

    def insert_pilots_per_mode(self, data_symbols, mode_key):
        rng = np.random.default_rng(_sha32_seed_from_tuple(mode_key))
        n_data = int(len(data_symbols))
        if n_data == 0:
            return np.array([], dtype=complex), np.array([], dtype=int)
        pilot_spacing = max(1, int(round(1.0 / self.pilot_ratio)))
        n_comb = int(np.ceil(n_data / pilot_spacing))
        preamble = 64
        n_pilots_total = preamble + n_comb
        pilot_idx = rng.integers(0, 4, size=n_pilots_total)
        pilot_seq = self.qpsk_constellation[pilot_idx]
        preamble_pos = np.arange(preamble)
        comb_pos = preamble + np.arange(0, n_data, pilot_spacing)[:n_comb]
        all_pilots = np.sort(np.unique(np.concatenate([preamble_pos, comb_pos])))
        n_total = preamble + n_data + n_comb
        all_pilots = all_pilots[all_pilots < n_total]
        frame = np.zeros(n_total, dtype=complex)
        is_pilot = np.isin(np.arange(n_total), all_pilots)
        pilot_locs = np.where(is_pilot)[0]
        data_locs = np.where(~is_pilot)[0]
        frame[pilot_locs] = pilot_seq[: len(pilot_locs)]
        if len(data_locs) > 0:
            frame[data_locs[:n_data]] = data_symbols
        self.pilot_sequence = pilot_seq
        self.pilot_positions = all_pilots
        return frame, all_pilots

    def extract_pilots(self, received_frame, pilot_positions):
        rx_pil = received_frame[pilot_positions]
        mask = np.ones(len(received_frame), dtype=bool)
        mask[pilot_positions] = False
        data = received_frame[mask]
        return data, rx_pil

    def estimate_channel(self, rx_pilots, method="MMSE", turbulence_var=0.0, noise_var=1.0):
        if len(rx_pilots) == 0 or self.pilot_sequence is None:
            return 1.0 + 0j
        tx = self.pilot_sequence[: len(rx_pilots)]
        ratios = np.divide(rx_pilots, tx, out=np.zeros_like(rx_pilots, dtype=complex), where=tx != 0)
        mask = (tx != 0) & np.isfinite(ratios)
        if np.sum(mask) == 0:
            return 1.0 + 0j
        ratios_valid = ratios[mask]
        tx_valid = tx[mask]
        if method.upper() == "LS":
            h = np.mean(ratios_valid)
        else:
            weights = np.abs(tx_valid) ** 2 / (noise_var + turbulence_var * np.abs(tx_valid) ** 2)
            h = np.average(ratios_valid, weights=weights)
        return h if np.isfinite(h) else (1.0 + 0j)


# ---------- encodingRunner ----------
class encodingRunner:
    def __init__(
        self,
        spatial_modes: Optional[List[Tuple[int, int]]] = None,
        wavelength: float = 1550e-9,
        w0: float = 25e-3,
        fec_rate: float = 0.8,
        pilot_ratio: float = 0.1,
        symbol_time_s: float = 1e-9,
        P_tx_watts: float = 1.0,
        laser_linewidth_kHz: Optional[float] = None,
        timing_jitter_ps: Optional[float] = None,
        tx_aperture_radius: Optional[float] = None,
        beam_tilt_x_rad: float = 0.0,
        beam_tilt_y_rad: float = 0.0,
    ):
        if spatial_modes is None:
            spatial_modes = [(0, -1), (0, 1)]
        self.spatial_modes = spatial_modes
        self.n_modes = len(spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.symbol_time_s = symbol_time_s
        self.qpsk = QPSKModulator(symbol_energy=1.0)

        # pyldpc wrapper (raises informative error if pyldpc missing)
        if not _HAS_PYLDPC:
            raise RuntimeError("pyldpc is required for encodingRunner. Install via `pip install pyldpc`.")
        self.ldpc = PyLDPCWrapper(n=2048, rate=fec_rate, dv=2, dc=8, seed=42)

        self.pilot_handler = PilotHandler(pilot_ratio=pilot_ratio)
        self.P_tx_watts = float(P_tx_watts)
        self.power_per_mode = self.P_tx_watts / max(1, self.n_modes)
        self.laser_linewidth_kHz = laser_linewidth_kHz
        self.timing_jitter_ps = timing_jitter_ps
        self.tx_aperture_radius = tx_aperture_radius
        self.beam_tilt_x_rad = beam_tilt_x_rad
        self.beam_tilt_y_rad = beam_tilt_y_rad

        # instantiate LG beams (if lgBeam available)
        self.lg_beams = {}
        for p, l in self.spatial_modes:
            if LaguerreGaussianBeam is None:
                self.lg_beams[(p, l)] = None
            else:
                try:
                    self.lg_beams[(p, l)] = LaguerreGaussianBeam(p, l, wavelength, w0)
                except Exception as e:
                    warnings.warn(f"Could not construct LG beam for mode {(p,l)}: {e}")
                    self.lg_beams[(p, l)] = None

        print(f"Spatial Modes: {self.spatial_modes}")
        print(f"Wavelength (nm): {wavelength * 1e9:.1f}, w0 (mm): {w0 * 1e3:.2f}")
        print(f"LDPC: n={self.ldpc.n}, k={self.ldpc.k}, rate={self.ldpc.rate:.4f}")
        print(f"Pilot ratio: {pilot_ratio}, symbol_time (ps): {self.symbol_time_s * 1e12:.1f}")
        print(f"Tx power total: {self.P_tx_watts:.3f} W, per-mode: {self.power_per_mode:.5f} W")

    def transmit(self, data_bits, generate_3d_field=False, z_field=500.0, grid_size=256, verbose=True, single_slice_if_large=True):
        data_bits = normalize_bits(np.asarray(data_bits, dtype=int))
        if verbose:
            print(f"Input info bits: {len(data_bits)}")
        coded = self.ldpc.encode(data_bits)
        if verbose:
            rate_eff = (len(data_bits) / (len(coded) + 1e-12)) if len(coded) > 0 else 0.0
            print(f"Encoded bits: {len(coded)} (effective rate ~{rate_eff:.4f})")

        # Modulate
        symbols = self.qpsk.modulate(coded)
        if symbols.size == 0:
            return FSO_MDM_Frame(tx_signals={})

        symbols_per_mode = len(symbols) // self.n_modes
        total_symbols = symbols_per_mode * self.n_modes
        symbols = symbols[:total_symbols]
        if verbose:
            print(f"Total symbols {total_symbols} ({symbols_per_mode} per mode)")

        tx_signals = {}
        idx = 0
        for mode_key in self.spatial_modes:
            syms = symbols[idx : idx + symbols_per_mode].copy()
            frame_sym, pilot_pos = self.pilot_handler.insert_pilots_per_mode(syms, mode_key)
            n_mode = len(frame_sym)

            seed = _sha32_seed_from_tuple(mode_key)
            pn = np.zeros(n_mode, dtype=float)
            beam = self.lg_beams.get(mode_key)
            if beam is not None and self.laser_linewidth_kHz is not None and self.laser_linewidth_kHz > 0:
                pn += beam.generate_phase_noise_sequence(n_mode, self.symbol_time_s, self.laser_linewidth_kHz, seed=seed)

            if self.timing_jitter_ps is not None and self.timing_jitter_ps > 0:
                f_c = 3e8 / self.wavelength
                jitter_rad = 2 * np.pi * f_c * self.timing_jitter_ps * 1e-12
                rng = np.random.default_rng(seed + 123456)
                pn += np.clip(jitter_rad * rng.standard_normal(n_mode), -np.pi, np.pi)

            frame_sym = frame_sym * np.exp(1j * pn)
            if np.mean(np.abs(frame_sym) ** 2) > 0:
                frame_sym = frame_sym / np.sqrt(np.mean(np.abs(frame_sym) ** 2))

            tx_signals[mode_key] = {
                "symbols": frame_sym,
                "frame": frame_sym,
                "beam": beam,
                "n_symbols": n_mode,
                "pilot_positions": pilot_pos,
                "phase_noise": pn,
            }
            if verbose:
                print(f" Mode {mode_key}: n_symbols={n_mode}, pilots={len(pilot_pos)}")
            idx += symbols_per_mode

        frame = FSO_MDM_Frame(tx_signals=tx_signals, metadata={"pilot_ratio": self.pilot_handler.pilot_ratio})

        if generate_3d_field:
            if any(b is None for b in self.lg_beams.values()):
                raise RuntimeError("generate_3d_field requested but some LG beams are missing (lgBeam import?).")
            field3d, grid = self._generate_spatial_field(tx_signals, z_field, grid_size, single_slice_if_large=single_slice_if_large)
            frame.multiplexed_field = field3d
            frame.grid_info = grid

        return frame

    def _generate_spatial_field(self, tx_signals, z=500.0, grid_size=256, extent_factor=3.0, single_slice_if_large=True):
        beams = [s["beam"] for s in tx_signals.values() if s.get("beam") is not None]
        if not beams:
            raise RuntimeError("No LaguerreGaussianBeam instances available for field generation.")
        max_w = max(b.beam_waist(z) for b in beams)
        extent_m = extent_factor * max_w
        x = np.linspace(-extent_m, extent_m, grid_size)
        y = np.linspace(-extent_m, extent_m, grid_size)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)

        # Precompute base fields
        mode_fields = {}
        for mode_key, sig in tx_signals.items():
            beam = sig.get("beam")
            if beam is None:
                continue
            f = beam.generate_beam_field(R, PHI, z, P_tx_watts=self.power_per_mode, laser_linewidth_kHz=None,
                                         timing_jitter_ps=None, tx_aperture_radius=self.tx_aperture_radius,
                                         beam_tilt_x_rad=self.beam_tilt_x_rad, beam_tilt_y_rad=self.beam_tilt_y_rad,
                                         phase_noise_samples=None)
            mode_fields[mode_key] = f.astype(np.complex64)

        n_symbols = max(s["n_symbols"] for s in tx_signals.values())
        if single_slice_if_large and n_symbols > 500:
            warnings.warn(f"n_symbols={n_symbols} > 500; returning single-slice (first symbol) to save memory.")
            n_symbols = 1

        if n_symbols <= 1:
            total = np.zeros((grid_size, grid_size), dtype=np.complex64)
            for mode_key, sig in tx_signals.items():
                if mode_key in mode_fields and sig["n_symbols"] > 0:
                    total += mode_fields[mode_key] * sig["symbols"][0]
            intensity = np.abs(total) ** 2
            grid = {"x": x, "y": y, "extent_m": extent_m, "grid_size": grid_size}
            return intensity.astype(np.float32), grid

        total_field_3d = np.zeros((n_symbols, grid_size, grid_size), dtype=np.complex64)
        for t in range(n_symbols):
            sl = np.zeros((grid_size, grid_size), dtype=np.complex64)
            for mode_key, sig in tx_signals.items():
                if t < sig["n_symbols"] and mode_key in mode_fields:
                    sl += mode_fields[mode_key] * sig["symbols"][t]
            p = np.mean(np.abs(sl) ** 2)
            if p > 0:
                total_field_3d[t] = (sl / np.sqrt(p)).astype(np.complex64)
            else:
                total_field_3d[t] = sl
            # keep memory pressure low
            if (t % 20) == 0:
                gc.collect()
        intensity_3d = np.abs(total_field_3d) ** 2
        grid = {"x": x, "y": y, "extent_m": extent_m, "grid_size": grid_size}
        return intensity_3d.astype(np.float32), grid

    def plot_system_summary(self, data_bits, frame, plot_dir="plots", save_name="encoding_summary.png"):
        os.makedirs(plot_dir, exist_ok=True)
        n_modes = len(self.spatial_modes)
        rows_modes = int(np.ceil(n_modes / 4)) if n_modes > 0 else 1
        fig = plt.figure(figsize=(16, 5 + 3 * rows_modes))
        fig.suptitle("FSO-MDM Transmitter Summary", fontsize=16, fontweight="bold")

        ax1 = plt.subplot2grid((2 + rows_modes, 4), (0, 0))
        self.qpsk.plot_constellation(ax=ax1)

        ax2 = plt.subplot2grid((2 + rows_modes, 4), (0, 1))
        first = self.spatial_modes[0]
        if first in frame.tx_signals:
            syms = frame.tx_signals[first]["symbols"][:200]
            ax2.plot(np.real(syms), np.imag(syms), ".", alpha=0.6)
            ax2.set_title(f"Tx Symbols (mode {first})")
            ax2.axis("equal"); ax2.grid(True)

        per_row = 4
        extent_m = 3 * self.w0
        gsize = 200
        # Precompute R,PHI for plotting per-beam intensities (coarse)
        x = np.linspace(-extent_m, extent_m, gsize)
        y = np.linspace(-extent_m, extent_m, gsize)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)

        # collect per-beam maxima to help consistent colormap scaling if desired
        per_beam_max = []
        for mode_key in self.spatial_modes:
            beam = self.lg_beams.get(mode_key)
            if beam is None:
                continue
            try:
                Itest = beam.calculate_intensity(R, PHI, 0,
                                                 P_tx_watts=self.power_per_mode,
                                                 laser_linewidth_kHz=None,
                                                 timing_jitter_ps=None,
                                                 tx_aperture_radius=self.tx_aperture_radius,
                                                 beam_tilt_x_rad=self.beam_tilt_x_rad,
                                                 beam_tilt_y_rad=self.beam_tilt_y_rad,
                                                 phase_noise_samples=None, symbol_time_s=None)
                per_beam_max.append(float(np.max(Itest)))
            except Exception:
                per_beam_max.append(1.0)

        global_maxI = max(per_beam_max) if per_beam_max else 1.0
        # safe vmin/vmax for LogNorm
        vmin_global = max(global_maxI * 1e-8, 1e-12)
        vmax_global = max(global_maxI, vmin_global * 10.0)

        for idx, mode_key in enumerate(self.spatial_modes):
            row = idx // per_row
            col = idx % per_row
            ax = plt.subplot2grid((2 + rows_modes, 4), (1 + row, col))
            beam = self.lg_beams.get(mode_key)
            if beam is None:
                ax.text(0.5, 0.5, f"No beam\n{mode_key}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"LG {mode_key}")
                continue
            I = beam.calculate_intensity(R, PHI, 0, P_tx_watts=self.power_per_mode, laser_linewidth_kHz=None,
                                         timing_jitter_ps=None, tx_aperture_radius=self.tx_aperture_radius,
                                         beam_tilt_x_rad=self.beam_tilt_x_rad, beam_tilt_y_rad=self.beam_tilt_y_rad,
                                         phase_noise_samples=None, symbol_time_s=None)
            # use global vmin/vmax for consistent scaling across mode panels
            vmin = vmin_global
            vmax = vmax_global if vmax_global > vmin else vmin * 10.0
            im = ax.imshow(I, extent=[-extent_m*1e3, extent_m*1e3, -extent_m*1e3, extent_m*1e3],
                           origin="lower", cmap="hot", norm=LogNorm(vmin=vmin, vmax=vmax))
            ax.set_title(f"LG p={beam.p} l={beam.l} M²={beam.M_squared:.1f}")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(plot_dir, save_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
        return save_path

    def validate_transmitter(self, frame, snr_db=10, max_modes=4):
        print("\n--- Validation ---")
        tx = frame.tx_signals
        if not tx:
            print("No tx signals to validate.")
            return
        keys = list(tx.keys())[:max_modes]
        print("Checking orthogonality using lgBeam.overlap_with (if available)...")
        ortho_ok = True
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                b1, b2 = self.lg_beams.get(k1), self.lg_beams.get(k2)
                if b1 is None or b2 is None:
                    print(f"  Missing beam for {k1} or {k2}")
                    ortho_ok = False
                    continue
                ov = b1.overlap_with(b2, r_max_factor=8.0, n_r=512, n_phi=360)
                print(f"  <{k1}|{k2}> = {abs(ov):.4e}")
                if abs(ov) > 0.05:
                    ortho_ok = False
        print("Orthogonality:", "PASS" if ortho_ok else "FAIL")

        print("Checking M^2 values...")
        m2_ok = True
        for k, beam in self.lg_beams.items():
            if beam is None:
                print(f"  {k}: missing")
                m2_ok = False
                continue
            expected = 2 * beam.p + abs(beam.l) + 1
            print(f"  {k}: M²={beam.M_squared:.3f} expected={expected:.3f}")
            if abs(beam.M_squared - expected) > 1e-6:
                m2_ok = False
        print("M²:", "PASS" if m2_ok else "FAIL")

        # AWGN BER on first mode (data symbols)
        first = next(iter(tx.keys()))
        syms = tx[first]["symbols"]
        pilot_pos = tx[first].get("pilot_positions", [])
        mask = np.ones(len(syms), dtype=bool)
        if len(pilot_pos) > 0:
            mask[pilot_pos] = False
        data_syms = syms[mask]
        if len(data_syms) == 0:
            print("No data symbols available for BER test.")
            return
        snr_lin = 10 ** (snr_db / 10.0)
        noise_var = 1.0 / max(1e-12, snr_lin)
        rng = np.random.default_rng(12345)
        noise = np.sqrt(noise_var / 2.0) * (rng.normal(size=len(data_syms)) + 1j * rng.normal(size=len(data_syms)))
        rx = data_syms + noise
        rx_bits = self.qpsk.demodulate_hard(rx)
        tx_bits = self.qpsk.demodulate_hard(data_syms)
        rx_bits = normalize_bits(rx_bits)
        tx_bits = normalize_bits(tx_bits)
        minlen = min(len(rx_bits), len(tx_bits))
        ber_emp = np.mean(rx_bits[:minlen] != tx_bits[:minlen]) if minlen > 0 else 1.0
        ber_th = 0.5 * erfc(np.sqrt(snr_lin))
        print(f"AWGN BER (empirical) = {ber_emp:.2e}, theoretical QPSK = {ber_th:.2e} @ Es/N0={snr_db} dB")
        print("Validation finished.")


# ---------- Demo ----------
if __name__ == "__main__":
    WAVELENGTH = 1550e-9
    W0 = 25e-3
    SPATIAL_MODES = [(0, 1), (0, -1), (0, 2), (0, -2), (0, 3), (0, -3)]
    FEC_RATE = 0.8
    PILOT_RATIO = 0.1
    N_INFO_BITS = 4096
    GRID_SIZE = 128
    Z_PROP = 500.0
    PLOT_DIR = os.path.join(os.getcwd(), "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    runner = encodingRunner(spatial_modes=SPATIAL_MODES, wavelength=WAVELENGTH, w0=W0,
                            fec_rate=FEC_RATE, pilot_ratio=PILOT_RATIO, symbol_time_s=1e-9,
                            P_tx_watts=1.0, laser_linewidth_kHz=10.0, timing_jitter_ps=5.0)

    data = np.random.default_rng(123).integers(0, 2, N_INFO_BITS)
    frame = runner.transmit(data, generate_3d_field=True, z_field=Z_PROP, grid_size=GRID_SIZE, single_slice_if_large=True)
    runner.validate_transmitter(frame)
    p = runner.plot_system_summary(data, frame, plot_dir=PLOT_DIR, save_name="encoding_summary_fixed.png")
    if p:
        print("Saved summary at:", p)
