import os
import json
import gc
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List, Union

import numpy as np
import matplotlib
# If running headless: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import erfc
import ast
from scipy import sparse
from scipy.sparse import issparse  # For H/G handling

# Attempt lgBeam import
try:
    from lgBeam import LaguerreGaussianBeam
except Exception as e:
    LaguerreGaussianBeam = None
    warnings.warn(f"Could not import lgBeam.LaguerreGaussianBeam: {e}. Skipping spatial fields.")

_HAS_PYLDPC = False
pyldpc_encode = None
pyldpc_decode = None
pyldpc_get_message = None
try:
    # Preferred import set (0.7.x and similar)
    from pyldpc import make_ldpc, encode as pyldpc_encode, decode as pyldpc_decode, get_message as pyldpc_get_message
    # Some pyldpc versions also expose an LDPC class for BP decoding
    try:
        from pyldpc import LDPC as pyldpc_LDPC_class
    except Exception:
        pyldpc_LDPC_class = None
    _HAS_PYLDPC = True
except Exception:
    # try minimal import (older/different packaging) — at least make_ldpc
    try:
        from pyldpc import make_ldpc
        _HAS_PYLDPC = True
    except Exception:
        warnings.warn("pyldpc not found. Install: pip install pyldpc for LDPC.")
        _HAS_PYLDPC = False

# ---------- Utilities ----------
def _closest_divisor_leq(n: int, target: int) -> int:
    if target <= 1:
        return 1
    for d in range(target, 1, -1):
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
        llr_Q = a_scale * np.imag(rx)
        llr_I = a_scale * np.real(rx)
        llrs = np.empty(2 * len(rx), dtype=float)
        llrs[0::2] = llr_Q  # bit-0 follows Q (matching constellation_map ordering)
        llrs[1::2] = llr_I  # bit-1 follows I component
        return llrs

    def plot_constellation(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        for bits, s in self.constellation_map.items():
            ax.plot(s.real, s.imag, "ro")
            ax.annotate(f"{bits[0]}{bits[1]}", (s.real, s.imag), fontsize=12)
        ax.axhline(0, color="grey")
        ax.axvline(0, color="grey")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("QPSK (Gray)")
        ax.grid(True)
        ax.axis("equal")
        return ax

# ---------- PyLDPCWrapper ----------
class PyLDPCWrapper:
    def __init__(self, n=2048, rate=0.8, dv=3, dc=None, seed=42):
        """
        Robust initialization for PyLDPCWrapper.
        - Ensures pyldpc present.
        - Chooses d_c that divides n (pyldpc requirement) and is close to theoretical dv/(1-rate).
        - Falls back gracefully if make_ldpc fails.
        """
        if not _HAS_PYLDPC:
            raise RuntimeError("pyldpc required. Install: pip install pyldpc")

        # Basic params (assign before use)
        self._n_req = int(n)
        self._requested_rate = float(rate)
        self.dv = int(dv)
        self.seed = int(seed)
        np.random.seed(self.seed)
        self._pyldpc = bool(_HAS_PYLDPC and (pyldpc_encode is not None and pyldpc_decode is not None and pyldpc_get_message is not None))
    
        # compute theoretical dc (may not divide n)
        theory_dc = max(3, int(round(self.dv / (1.0 - self._requested_rate))))
    
        # helper: get divisors of n (efficient enough for n ~ few thousand)
        def divisors_of(val):
            divs = []
            i = 1
            while i * i <= val:
                if val % i == 0:
                    divs.append(i)
                    if i != val // i:
                        divs.append(val // i)
                i += 1
            return sorted(divs)

        n_val = self._n_req
        n_divs = divisors_of(n_val)
        # require d_c > dv and >= 3 (practical constraint)
        candidates = [d for d in n_divs if (d >= 3 and d > self.dv)]
        if len(candidates) == 0:
            # extremely unlikely for reasonable dv/n; raise clear error
            raise ValueError(f"No valid d_c divisors found for n={n_val} with dv={self.dv}")

        # if user provided dc explicitly, prefer it but ensure it divides n (otherwise pick nearest)
        if dc is not None:
            if n_val % int(dc) == 0 and int(dc) > self.dv:
                self.dc = int(dc)
            else:
                warnings.warn(f"Requested d_c={dc} is invalid for n={n_val} (must divide n and be > dv). Falling back to auto-select.")
                # fall through to auto-pick
                self.dc = None
        else:
            self.dc = None

        # Auto-pick d_c (if not set or invalid) as divisor of n closest to theoretical (tie -> pick larger)
        if self.dc is None:
            self.dc = int(min(candidates, key=lambda x: (abs(x - theory_dc), -x)))
            if self.dc != theory_dc:
                warnings.warn(f"Adjusted d_c from theoretical {theory_dc} -> {self.dc} (must divide n={n_val}).")

        # Now call make_ldpc with chosen dv, dc
        try:
            H, G = make_ldpc(self._n_req, self.dv, self.dc, seed=self.seed, systematic=True, sparse=True)
        except Exception as e:
            # try one more safe fallback: pick the next valid candidate divisor (larger one) and retry
            warnings.warn(f"make_ldpc failed for dv={self.dv}, dc={self.dc}: {e}. Trying alternative divisors of n.")
            tried = set([self.dc])
            success = False
            for alt in sorted(candidates, key=lambda x: (abs(x - theory_dc), -x)):
                if alt in tried:
                    continue
                if alt <= self.dv:
                    continue
                try:
                    H, G = make_ldpc(self._n_req, self.dv, int(alt), seed=self.seed, systematic=True, sparse=True)
                    self.dc = int(alt)
                    warnings.warn(f"make_ldpc succeeded with alternative dc={self.dc}.")
                    success = True
                    break
                except Exception:
                    tried.add(alt)
                    continue
            if not success:
                # Last resort: fallback to a stable small regular (dv=3, dc= (choose divisor>dv)) attempt
                fallback_dc = next((d for d in n_divs if d > self.dv and d >= 3), None)
                if fallback_dc is None:
                    raise RuntimeError("Failed to find any workable d_c for LDPC generation.")
                warnings.warn(f"All attempts failed; using fallback dv=3, dc={fallback_dc} (may change rate).")
                H, G = make_ldpc(self._n_req, 3, int(fallback_dc), seed=self.seed, systematic=True, sparse=True)
                self.dv = 3
                self.dc = int(fallback_dc)

        # Store H,G (sparse)
        self.H = H if issparse(H) else sparse.csr_matrix(H)
        self.G = G if issparse(G) else sparse.csr_matrix(G)
    
        # infer k,n,m from G (handle sparse shapes)
        gshape = self.G.shape if issparse(self.G) else np.array(self.G.shape)
        if gshape[0] < gshape[1]:
            self.k, self.n = int(gshape[0]), int(gshape[1])
        else:
            self.G = self.G.T
            self.k, self.n = int(self.G.shape[0]), int(self.G.shape[1])
        self.m = self.n - self.k
        actual_rate = float(self.k) / float(self.n)
        if abs(actual_rate - self._requested_rate) > 0.05:
            warnings.warn(f"Actual rate={actual_rate:.4f} != requested {self._requested_rate} (dv/dc rounding).")
        if not (0 < self.k < self.n):
            raise ValueError(f"Invalid LDPC dims inferred: k={self.k}, n={self.n}")

    @property
    def rate(self):
        return float(self.k) / float(self.n)

    @rate.setter
    def rate(self, value):
        self._requested_rate = float(value)  # Re-init for new rate

    def encode(self, info_bits):

        info_bits = normalize_bits(np.asarray(info_bits, dtype=int))
        if info_bits.size == 0:
            return np.array([], dtype=int)

        num_blocks = int(np.ceil(info_bits.size / self.k))
        pad_len = num_blocks * self.k - info_bits.size
        info_p = np.concatenate([info_bits, np.zeros(pad_len, dtype=int)]) if pad_len > 0 else info_bits.copy()

        # prepare dense G for safe fallback (if convertible)
        G_dense = None
        try:
            G_dense = self.G.toarray() if issparse(self.G) else np.asarray(self.G)
        except Exception:
            G_dense = None

        codewords = []
        for b in range(num_blocks):
            u = info_p[b * self.k : (b + 1) * self.k].astype(int)
            block_cw = None

            # Try pyldpc.encode if present
            if getattr(self, "_pyldpc", False) and (pyldpc_encode is not None):
                # determine likely good candidate orientations (preferring (n,k) tG)
                try:
                    gshape = self.G.shape if issparse(self.G) else np.asarray(self.G).shape
                    if gshape[0] == self.k and gshape[1] == self.n:
                        candidates = [self.G.T, self.G]
                    elif gshape[0] == self.n and gshape[1] == self.k:
                        candidates = [self.G, self.G.T]
                    else:
                        candidates = [self.G.T, self.G]
                except Exception:
                    candidates = [self.G.T, self.G]

                for cand in candidates:
                    try:
                        # try with large snr kwarg first (deterministic BPSK-like result)
                        try:
                            cw = pyldpc_encode(cand, u, snr=1e6)
                        except TypeError:
                            # signature may not accept snr
                            cw = pyldpc_encode(cand, u)
                        cw_arr = np.asarray(cw)
                        # if encode returned floats (BPSK-like), threshold to bits
                        if np.issubdtype(cw_arr.dtype, np.floating):
                            cw_bits = (cw_arr < 0).astype(int)
                        else:
                            cw_bits = normalize_bits(cw_arr.astype(int))
                        # ensure length exactly n
                        if cw_bits.size < self.n:
                            cw_bits = np.pad(cw_bits, (0, self.n - cw_bits.size), constant_values=0)
                        elif cw_bits.size > self.n:
                            cw_bits = cw_bits[: self.n]
                        block_cw = cw_bits
                        break
                    except Exception:
                        block_cw = None
                        continue

            # Fallback: binary multiply with dense generator matrix
            if block_cw is None:
                if G_dense is None:
                    raise RuntimeError("pyldpc.encode failed and generator matrix is not available for fallback.")
                gd = np.asarray(G_dense)
                # handle gd shape being (k,n) or (n,k)
                if gd.shape[0] == self.k and gd.shape[1] == self.n:
                    # (k,n) -> u @ G  (1 x k) @ (k x n) => (n,)
                    try:
                        cw_bits = (u @ gd) % 2
                    except Exception:
                        cw_bits = np.mod(np.dot(u, gd), 2)
                elif gd.shape[0] == self.n and gd.shape[1] == self.k:
                    # (n,k) -> (gd.T @ u) % 2
                    cw_bits = (gd.T @ u) % 2
                else:
                    # last-ditch attempt
                    try:
                        cw_bits = (gd @ u) % 2
                    except Exception as e:
                        raise RuntimeError(f"Cannot fallback-encode block with generator matrix: {e}")
                block_cw = normalize_bits(np.asarray(cw_bits, dtype=int))
                if block_cw.size < self.n:
                    block_cw = np.pad(block_cw, (0, self.n - block_cw.size), constant_values=0)
                elif block_cw.size > self.n:
                    block_cw = block_cw[: self.n]

            codewords.append(block_cw)

        coded = np.concatenate(codewords).astype(int)
        # Ensure exact returned length
        expected_len = num_blocks * self.n
        if coded.size < expected_len:
            coded = np.pad(coded, (0, expected_len - coded.size), constant_values=0)
        elif coded.size > expected_len:
            coded = coded[:expected_len]
        return coded

    def encode_hard(self, info_bits: np.ndarray) -> np.ndarray:
        return self.encode(info_bits)

    def decode_hard(self, received_bits, max_iters=50):
        r = normalize_bits(np.asarray(received_bits, dtype=int))
        if r.size == 0:
            return np.array([], dtype=int)
        # only full codeword blocks
        num_blocks = r.size // self.n
        recovered = []
        for b in range(num_blocks):
            block = r[b * self.n : (b + 1) * self.n].astype(int)
            if getattr(self, "_pyldpc", False) and (pyldpc_decode is not None and pyldpc_get_message is not None):
                try:
                    # pyldpc.decode expects LLRs/soft values. Convert hard bits -> large-magnitude LLR:
                    # bit 0 -> large positive LLR, bit 1 -> large negative LLR
                    llr = (1.0 - 2.0 * block) * 1e6
                    x_hat = pyldpc_decode(self.H, llr, maxiter=max_iters)
                    # x_hat may be floats; convert to bits if needed
                    x_hat_arr = np.asarray(x_hat)
                    if np.issubdtype(x_hat_arr.dtype, np.floating):
                        x_bits = (x_hat_arr < 0).astype(int)
                    else:
                        x_bits = normalize_bits(x_hat_arr.astype(int))
                    # Extract message bits using get_message (if available)
                    try:
                        u_hat = pyldpc_get_message(self.G, x_bits)
                        u_hat = normalize_bits(np.asarray(u_hat, dtype=int))
                    except Exception:
                        # fallback: if systematic, first k bits are the message
                        u_hat = x_bits[: self.k]
                    recovered.append(u_hat)
                except Exception:
                    recovered.append(block[: self.k])
            else:
                recovered.append(block[: self.k])

        if recovered:
            return np.concatenate(recovered)
        return np.array([], dtype=int)

    def decode_bp(self, llrs, max_iter=50):
        llrs = np.asarray(llrs, dtype=float)
        if llrs.size == 0:
            return np.array([], dtype=int)
        if not getattr(self, "_pyldpc", False) or pyldpc_decode is None or pyldpc_get_message is None:
            warnings.warn("pyldpc soft decoder unavailable; falling back to hard decisions.")
            hard_bits = (llrs < 0).astype(int)
            return self.decode_hard(hard_bits, max_iters=max_iter // 5)

        if llrs.size % self.n != 0:
            raise ValueError(f"LLR length {llrs.size} is not a multiple of code length n={self.n}.")

        num_blocks = llrs.size // self.n
        recovered = []
        for b in range(num_blocks):
            block_llr = llrs[b * self.n : (b + 1) * self.n]
            try:
                x_hat = pyldpc_decode(self.H, block_llr, maxiter=max_iter)
                x_arr = np.asarray(x_hat)
                if np.issubdtype(x_arr.dtype, np.floating):
                    x_bits = (x_arr < 0).astype(int)
                else:
                    x_bits = normalize_bits(x_arr.astype(int))
                try:
                    u_hat = pyldpc_get_message(self.G, x_bits)
                    u_hat = normalize_bits(np.asarray(u_hat, dtype=int))
                except Exception:
                    u_hat = x_bits[: self.k]
                recovered.append(u_hat)
            except Exception:
                warnings.warn("Soft decode block failed; using hard decisions for that block.")
                hard_block = (block_llr < 0).astype(int)
                recovered.append(self.decode_hard(hard_block, max_iters=max_iter // 5)[: self.k])

        return np.concatenate(recovered) if recovered else np.array([], dtype=int)

# ---------- Pilot Handler ----------
class PilotHandler:
    def __init__(self, pilot_ratio=0.1, pattern="uniform"):
        if not (0.0 < pilot_ratio < 1.0):
            raise ValueError("pilot_ratio in (0,1)")
        self.pilot_ratio = float(pilot_ratio)
        self.pattern = pattern
        self.qpsk_constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
        # NOTE: intentionally stateless regarding per-mode pilot sequences/positions

    def insert_pilots_per_mode(self, data_symbols, mode_key):
        """
        Insert pilots into `data_symbols` for mode `mode_key`.
        Pattern: 64-symbol preamble + comb pattern (pilots inserted before data symbols).
        Returns (frame_with_pilots, pilot_positions, pilot_sequence).
        This method is stateless with respect to per-mode pilots (caller must store results).
        
        Note: Comb pilots are inserted without dropping payload symbols, so frame length =
        preamble + n_data + n_comb. Actual pilot ratio = (preamble + n_comb) / (frame length),
        which may differ from self.pilot_ratio due to the fixed preamble length.
        """
        rng = np.random.default_rng(_sha32_seed_from_tuple(mode_key))
        n_data = int(len(data_symbols))
        if n_data == 0:
            return np.array([], dtype=complex), np.array([], dtype=int), np.array([], dtype=complex)
        
        # Calculate comb pilot pattern
        pilot_spacing = max(1, int(round(1.0 / self.pilot_ratio)))
        n_comb = int(np.ceil(n_data / pilot_spacing))
        preamble = 64
        
        # Generate pilot sequence (preamble + comb)
        n_pilots_total = preamble + n_comb
        pilot_idx = rng.integers(0, 4, size=n_pilots_total)
        pilot_seq = self.qpsk_constellation[pilot_idx]
        
        comb_data_indices = np.arange(0, n_data, pilot_spacing)[:n_comb]

        frame_list = []
        pilot_positions = []

        # Insert preamble pilots at the start
        for idx in range(preamble):
            frame_list.append(pilot_seq[idx])
            pilot_positions.append(idx)

        comb_idx = 0
        for data_idx in range(n_data):
            while comb_idx < len(comb_data_indices) and comb_data_indices[comb_idx] == data_idx:
                frame_list.append(pilot_seq[preamble + comb_idx])
                pilot_positions.append(len(frame_list) - 1)
                comb_idx += 1
            frame_list.append(data_symbols[data_idx])

        # In rare cases comb_idx may lag (shouldn't, but guard for safety)
        while comb_idx < len(comb_data_indices):
            frame_list.append(pilot_seq[preamble + comb_idx])
            pilot_positions.append(len(frame_list) - 1)
            comb_idx += 1

        frame = np.asarray(frame_list, dtype=complex)
        pilot_positions = np.asarray(pilot_positions, dtype=int)

        # return pilot_seq (already sized preamble + n_comb) and pilot positions
        return frame, pilot_positions, pilot_seq

    def extract_pilots(self, received_frame, pilot_positions):
        """
        Stateless extraction: returns (data_symbols, rx_pilots)
        """
        rx_pil = received_frame[pilot_positions]
        mask = np.ones(len(received_frame), dtype=bool)
        mask[pilot_positions] = False
        data = received_frame[mask]
        return data, rx_pil

    def estimate_channel(self, rx_pilots, tx_pilot_sequence=None, method="MMSE", turbulence_var=0.0, noise_var=1.0):
        """
        Estimate channel given rx_pilots and the *tx_pilot_sequence* (must be provided per-mode by caller).
        If tx_pilot_sequence is None, fall back to unity scalar.
        """
        if tx_pilot_sequence is None or len(rx_pilots) == 0:
            return 1.0 + 0j
        tx = tx_pilot_sequence[:len(rx_pilots)]
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

        if not _HAS_PYLDPC:
            raise RuntimeError("pyldpc required for encodingRunner. Install: pip install pyldpc.")
        self.ldpc = PyLDPCWrapper(n=2048, rate=fec_rate, dv=3, dc=None, seed=42)  # FIX: dc=None for auto-tune (avoids hardcode/adjust warn)

        self.pilot_handler = PilotHandler(pilot_ratio=pilot_ratio)
        self.P_tx_watts = float(P_tx_watts)
        self.power_per_mode = self.P_tx_watts / max(1, self.n_modes)
        self.laser_linewidth_kHz = laser_linewidth_kHz
        self.timing_jitter_ps = timing_jitter_ps
        self.tx_aperture_radius = tx_aperture_radius
        self.beam_tilt_x_rad = beam_tilt_x_rad
        self.beam_tilt_y_rad = beam_tilt_y_rad

        self.lg_beams = {}
        for p, l in self.spatial_modes:
            if LaguerreGaussianBeam is None:
                self.lg_beams[(p, l)] = None
                continue
            try:
                self.lg_beams[(p, l)] = LaguerreGaussianBeam(p, l, wavelength, w0)
            except Exception as e:
                warnings.warn(f"LG beam {(p,l)} failed: {e}")
                self.lg_beams[(p, l)] = None

        print(f"Spatial Modes: {self.spatial_modes}")
        print(f"Wavelength (nm): {wavelength * 1e9:.1f}, w0 (mm): {w0 * 1e3:.2f}")
        print(f"LDPC: dv={self.ldpc.dv}, dc={self.ldpc.dc}, n={self.ldpc.n}, k={self.ldpc.k}, rate={self.ldpc.rate:.4f}")  # FIX: Add dv/dc for debug
        print(f"Pilot ratio: {pilot_ratio}, symbol_time (ps): {self.symbol_time_s * 1e12:.1f}")
        print(f"Tx power total: {self.P_tx_watts:.3f} W, per-mode: {self.power_per_mode:.5f} W")

    def transmit(self, data_bits, generate_3d_field=False, z_field=500.0, grid_size=256, verbose=True, single_slice_if_large=True):
        data_bits = normalize_bits(np.asarray(data_bits, dtype=int))
        if verbose:
            print(f"Input info bits: {len(data_bits)}")
        coded = self.ldpc.encode(data_bits)
        if verbose:
            rate_eff = len(data_bits) / (len(coded) + 1e-12) if len(coded) > 0 else 0.0
            print(f"Encoded bits: {len(coded)} (effective rate ~{rate_eff:.4f})")

        symbols = self.qpsk.modulate(coded)
        if symbols.size == 0:
            return FSO_MDM_Frame(tx_signals={})

        # Even distribution per mode (pad with zeros if needed)
        # Note: Padding with zeros (0+0j) is better than padding with a constellation point
        # to avoid biasing the decoder. Zeros will be demodulated as (0,0) → (1+j)/√2,
        # but this is more neutral than repeating a specific symbol.
        symbols_per_mode = len(symbols) // self.n_modes
        total_symbols = symbols_per_mode * self.n_modes
        if len(symbols) > total_symbols:
            symbols = symbols[:total_symbols]
        else:
            # Pad with zeros (0+0j) instead of first constellation point
            # This avoids biasing the decoder with repeated symbols
            pad_len = total_symbols - len(symbols)
            pad = np.zeros(pad_len, dtype=complex)
            symbols = np.concatenate([symbols, pad])
        if verbose:
            print(f"Total symbols {total_symbols} ({symbols_per_mode} per mode)")

        tx_signals = {}
        idx = 0
        for mode_key in self.spatial_modes:
            syms = symbols[idx : idx + symbols_per_mode].copy()
            # IMPORTANT: pilot insertion now returns pilot sequence explicitly
            frame_sym, pilot_pos, pilot_seq = self.pilot_handler.insert_pilots_per_mode(syms, mode_key)
            n_mode = len(frame_sym)

            seed = _sha32_seed_from_tuple(mode_key)
            pn = np.zeros(n_mode, dtype=float)
            beam = self.lg_beams.get(mode_key)
            
            # Generate phase noise sequence (will apply only to data symbols)
            if beam is not None and self.laser_linewidth_kHz and self.laser_linewidth_kHz > 0:
                try:
                    pn += beam.generate_phase_noise_sequence(n_mode, self.symbol_time_s, self.laser_linewidth_kHz, seed=seed)
                except:
                    pass  # Skip if method missing

            if self.timing_jitter_ps and self.timing_jitter_ps > 0:
                f_c = 3e8 / self.wavelength
                jitter_rad = 2 * np.pi * f_c * self.timing_jitter_ps * 1e-12
                rng = np.random.default_rng(seed + 123456)
                pn += np.clip(jitter_rad * rng.standard_normal(n_mode), -np.pi, np.pi)

            # Apply phase noise only to DATA symbols (not pilots)
            # Pilots must remain known for channel estimation
            # Create mask: True for data symbols, False for pilots
            is_pilot_mask = np.isin(np.arange(n_mode), pilot_pos)
            pn_data_only = pn.copy()
            pn_data_only[is_pilot_mask] = 0.0  # Zero phase noise on pilots
            
            frame_sym *= np.exp(1j * pn_data_only)
            
            # Power normalization: Normalize to unit average power per symbol
            # This ensures consistent power across modes, but note that the actual
            # optical power is determined by the beam field scaling (P_tx_watts parameter).
            # This normalization is for digital symbol consistency, not optical power budget.
            if np.mean(np.abs(frame_sym)**2) > 0:
                frame_sym /= np.sqrt(np.mean(np.abs(frame_sym)**2))

            tx_signals[mode_key] = {
                "symbols": frame_sym,
                "frame": frame_sym,
                "beam": beam,
                "n_symbols": n_mode,
                "pilot_positions": pilot_pos,
                "pilot_sequence": pilot_seq,   # STORE per-mode pilot sequence explicitly
                "phase_noise": pn,
            }
            if verbose:
                print(f" Mode {mode_key}: n_symbols={n_mode}, pilots={len(pilot_pos)}")
            idx += symbols_per_mode

        frame = FSO_MDM_Frame(tx_signals=tx_signals, metadata={"pilot_ratio": self.pilot_handler.pilot_ratio})

        if generate_3d_field:
            if all(b is None for b in self.lg_beams.values()):
                warnings.warn("No LG beams; skipping 3D field.")
                frame.multiplexed_field = None
                frame.grid_info = None
            else:
                field3d, grid = self._generate_spatial_field(tx_signals, z_field, grid_size, single_slice_if_large)
                frame.multiplexed_field = field3d
                frame.grid_info = grid

        return frame

    def _generate_spatial_field(self, tx_signals, z=500.0, grid_size=256, extent_factor=3.0, single_slice_if_large=True):
        beams = [s["beam"] for s in tx_signals.values() if s.get("beam") is not None]
        if not beams:
            raise RuntimeError("No LG beams for field gen.")
        max_w = max((b.beam_waist(z) for b in beams if b), default=self.w0)
        extent_m = extent_factor * max_w
        x = np.linspace(-extent_m, extent_m, grid_size)
        y = np.linspace(-extent_m, extent_m, grid_size)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)

        mode_fields = {}
        for mode_key, sig in tx_signals.items():
            beam = sig.get("beam")
            if beam is None:
                continue
            try:
                f = beam.generate_beam_field(R, PHI, z, P_tx_watts=self.power_per_mode)
                mode_fields[mode_key] = f.astype(np.complex64)
            except Exception as e:
                warnings.warn(f"Field for {mode_key} failed: {e}")
                mode_fields[mode_key] = np.zeros((grid_size, grid_size), dtype=np.complex64)

        n_symbols = max(s["n_symbols"] for s in tx_signals.values())
        if single_slice_if_large and n_symbols > 500:
            warnings.warn(f"n_symbols={n_symbols}>500; single slice.")
            n_symbols = 1

        if n_symbols <= 1:
            total = np.zeros((grid_size, grid_size), dtype=np.complex64)
            for mode_key, sig in tx_signals.items():
                if mode_key in mode_fields and sig["n_symbols"] > 0:
                    total += mode_fields[mode_key] * sig["symbols"][0]
            intensity = np.abs(total)**2
            grid = {"x": x, "y": y, "extent_m": extent_m, "grid_size": grid_size}
            return intensity.astype(np.float32), grid

        total_field_3d = np.zeros((n_symbols, grid_size, grid_size), dtype=np.complex64)
        for t in range(n_symbols):
            sl = np.zeros((grid_size, grid_size), dtype=np.complex64)
            for mode_key, sig in tx_signals.items():
                if t < sig["n_symbols"] and mode_key in mode_fields:
                    sl += mode_fields[mode_key] * sig["symbols"][t]
            p = np.mean(np.abs(sl)**2)
            total_field_3d[t] = sl / np.sqrt(p) if p > 0 else sl
            if t % 20 == 0:
                gc.collect()
        intensity_3d = np.abs(total_field_3d)**2
        grid = {"x": x, "y": y, "extent_m": extent_m, "grid_size": grid_size}
        return intensity_3d.astype(np.float32), grid

    def plot_system_summary(self, data_bits, frame, plot_dir="plots", save_name="encoding_summary.png"):
        os.makedirs(plot_dir, exist_ok=True)
        n_modes = len(self.spatial_modes)
        rows = int(np.ceil(n_modes / 4))
        fig = plt.figure(figsize=(16, 5 + 3 * rows))
        fig.suptitle("FSO-MDM Transmitter Summary", fontsize=16, fontweight="bold")

        ax1 = plt.subplot2grid((2 + rows, 4), (0, 0))
        self.qpsk.plot_constellation(ax=ax1)

        ax2 = plt.subplot2grid((2 + rows, 4), (0, 1))
        first = self.spatial_modes[0]
        if first in frame.tx_signals:
            syms = frame.tx_signals[first]["symbols"][:200]
            ax2.plot(np.real(syms), np.imag(syms), ".", alpha=0.6)
            ax2.set_title(f"Tx Symbols (mode {first})")
            ax2.axis("equal")
            ax2.grid(True)

        per_row = 4
        extent_m = 3 * self.w0
        gsize = 200
        x = np.linspace(-extent_m, extent_m, gsize)
        y = np.linspace(-extent_m, extent_m, gsize)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)

        for idx, mode_key in enumerate(self.spatial_modes):
            row = idx // per_row
            col = idx % per_row
            ax = plt.subplot2grid((2 + rows, 4), (1 + row, col))
            beam = self.lg_beams.get(mode_key)
            if beam is None:
                ax.text(0.5, 0.5, f"No beam\n{mode_key}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"LG {mode_key}")
                continue
            try:
                I = beam.calculate_intensity(R, PHI, 0, P_tx_watts=self.power_per_mode)
            except:
                I = np.zeros_like(R)
            maxI = np.max(I) if np.max(I) > 0 else 1.0
            vmin = max(maxI * 1e-8, 1e-12)
            im = ax.imshow(I, extent=[-extent_m*1e3, extent_m*1e3, -extent_m*1e3, extent_m*1e3], origin="lower",
                           cmap="hot", norm=LogNorm(vmin=vmin, vmax=maxI))
            ax.set_title(f"LG p={beam.p} l={beam.l} M²={beam.M_squared:.1f}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
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
            print("No tx signals.")
            return
        keys = list(tx.keys())[:max_modes]
        print("Orthogonality check (lgBeam.overlap_with):")
        ortho_ok = True
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                b1, b2 = self.lg_beams.get(k1), self.lg_beams.get(k2)
                if b1 is None or b2 is None:
                    print(f"  Missing beam {k1} or {k2}")
                    ortho_ok = False
                    continue
                try:
                    ov = b1.overlap_with(b2, r_max_factor=8.0, n_r=512, n_phi=360)
                    print(f"  <{k1}|{k2}> = {abs(ov):.4e}")
                    if abs(ov) > 0.05:
                        ortho_ok = False
                except:
                    print(f"  overlap_with missing for {k1}-{k2}")
                    ortho_ok = False
        print("Orthogonality: " + ("PASS" if ortho_ok else "FAIL"))

        print("M^2 check:")
        m2_ok = True
        for k, beam in self.lg_beams.items():
            if beam is None:
                print(f"  {k}: missing")
                m2_ok = False
                continue
            expected = 2 * beam.p + abs(beam.l) + 1
            print(f"  {k}: M²={beam.M_squared:.3f} (expected {expected:.3f})")
            if abs(beam.M_squared - expected) > 1e-6:
                m2_ok = False
        print("M^2: " + ("PASS" if m2_ok else "FAIL"))

        print("LDPC round-trip:")
        test_info = np.random.randint(0, 2, self.ldpc.k)
        test_coded = self.ldpc.encode(test_info)
        test_decoded = self.ldpc.decode_hard(test_coded)
        ber_ldpc = np.mean(test_info != test_decoded[:len(test_info)])
        print(f"LDPC BER (ideal): {ber_ldpc:.2e}")

        first = next(iter(tx.keys()))
        syms = tx[first]["symbols"]
        pilot_pos = tx[first].get("pilot_positions", [])
        mask = np.ones(len(syms), dtype=bool)
        mask[pilot_pos] = False
        data_syms = syms[mask]
        if len(data_syms) == 0:
            print("No data for BER.")
            return
        snr_lin = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_lin
        rng = np.random.default_rng(12345)
        noise = np.sqrt(noise_var / 2) * (rng.normal(size=len(data_syms)) + 1j * rng.normal(size=len(data_syms)))
        rx = data_syms + noise
        rx_bits = self.qpsk.demodulate_hard(rx)
        tx_bits = self.qpsk.demodulate_hard(data_syms)
        minlen = min(len(rx_bits), len(tx_bits))
        ber_emp = np.mean(rx_bits[:minlen] != tx_bits[:minlen]) if minlen > 0 else 1.0
        ber_th = 0.5 * erfc(np.sqrt(snr_lin))
        print(f"AWGN BER emp={ber_emp:.2e}, th={ber_th:.2e} @ {snr_db}dB")
        print("Validation done.")

# ---------- Demo ----------
if __name__ == "__main__":
    WAVELENGTH = 1550e-9
    W0 = 25e-3
    SPATIAL_MODES = [(0, 1), (0, -1), (0, 2), (0, -2), (0, 3), (0, -3)]
    FEC_RATE = 0.8
    PILOT_RATIO = 0.1
    N_INFO_BITS = 4096
    GRID_SIZE = 128
    Z_PROP = 5000.0
    PLOT_DIR = os.path.join(os.getcwd(), "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    runner = encodingRunner(spatial_modes=SPATIAL_MODES, wavelength=WAVELENGTH, w0=W0,
                            fec_rate=FEC_RATE, pilot_ratio=PILOT_RATIO, symbol_time_s=1e-9,
                            P_tx_watts=1.0, laser_linewidth_kHz=10.0, timing_jitter_ps=5.0)

    data = np.random.default_rng(123).integers(0, 2, N_INFO_BITS)
    frame = runner.transmit(data, generate_3d_field=False, z_field=Z_PROP, grid_size=GRID_SIZE)
    runner.validate_transmitter(frame)
    p = runner.plot_system_summary(data, frame, plot_dir=PLOT_DIR, save_name="encoding_summary.png")
    if p:
        print(f"Saved: {p}")