# receiver.py -- Receiver for FSO-MDM OAM system (aligned to your rectified encoder & turbulence)
# Requirements: numpy, scipy, matplotlib, encoding.py, turbulence.py, lgBeam.py (optional but preferred)
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, pinv
from scipy.fft import fft2, ifft2
from typing import Dict, Tuple, Any, Optional

# script dir resolution (same pattern used in other modules)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

# imports from your rectified modules
try:
    from encoding import QPSKModulator, PilotHandler, PyLDPCWrapper, FSO_MDM_Frame
except Exception as e:
    raise ImportError(f"receiver.py requires encoding.py in the same directory: {e}")

try:
    from turbulence import angular_spectrum_propagation
except Exception as e:
    angular_spectrum_propagation = None
    warnings.warn(f"turbulence.angular_spectrum_propagation not available: {e}")

# lgBeam may or may not be needed (we prefer using beam instances attached to frame)
try:
    from lgBeam import LaguerreGaussianBeam
except Exception:
    LaguerreGaussianBeam = None

warnings.filterwarnings("ignore")


# -------------------------------------------------------
# Utilities: grid reconstruction, normalization helpers
# -------------------------------------------------------
def reconstruct_grid_from_gridinfo(grid_info: Dict[str, Any]):
    """
    grid_info expected keys (from encoding._generate_spatial_field):
      - x: 1D array
      - y: 1D array
      - grid_size or extent_m (not mandatory)
    Returns X,Y,delta (float), x,y arrays
    """
    if grid_info is None:
        raise ValueError("grid_info is required to reconstruct spatial grid.")
    x = np.asarray(grid_info.get("x"))
    y = np.asarray(grid_info.get("y"))
    if x.size == 0 or y.size == 0:
        raise ValueError("grid_info.x/y empty or missing.")
    X, Y = np.meshgrid(x, y, indexing="ij")
    # sampling interval (assume uniform)
    delta_x = np.mean(np.diff(x))
    delta_y = np.mean(np.diff(y))
    if not np.isclose(delta_x, delta_y, rtol=1e-6, atol=0.0):
        warnings.warn("Non-square sampling intervals detected; using delta_x as delta.")
    delta = float(delta_x)
    return X, Y, delta, x, y


def energy_normalize_field(field: np.ndarray, delta: float):
    """
    Normalize field so that total power (sum |E|^2 * delta^2) == 1
    """
    p = np.sum(np.abs(field) ** 2) * (delta ** 2)
    if p > 0:
        return field / np.sqrt(p)
    return field


# -------------------------------------------------------
# OAM Demultiplexer (mode projection)
# -------------------------------------------------------
class OAMDemultiplexer:
    """
    Project received complex field onto reference LG modes (uses transmitter's beam objects when present).
    """

    def __init__(self, spatial_modes, wavelength, w0, z_distance, angular_prop_func=angular_spectrum_propagation):
        self.spatial_modes = list(spatial_modes)
        self.n_modes = len(self.spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
        self.angular_prop = angular_prop_func
        # cache for reference fields per grid size/delta
        self._ref_cache = {}

    def _make_ref_key(self, mode_key, N, delta, X_shape):
        return (mode_key, int(N), float(delta), X_shape)

    def reference_field(self, mode_key: Tuple[int, int], X, Y, delta, grid_z, tx_beam_obj=None):
        """
        Construct (or retrieve) propagated reference field for mode_key on grid X,Y (at z=grid_z).
        If tx_beam_obj is provided (the beam instance saved by transmitter), use it to generate reference at z=0 then propagate.
        """
        N = X.shape[0]
        key = self._make_ref_key(mode_key, N, delta, X.shape)
        if key in self._ref_cache:
            return self._ref_cache[key].copy()

        # compute R,PHI and generate reference field at z=0 via provided beam or by constructing LaguerreGaussianBeam
        R = np.sqrt(X ** 2 + Y ** 2)
        PHI = np.arctan2(Y, X)

        beam = tx_beam_obj
        if beam is None:
            # try to instantiate if lgBeam available (fallback)
            p, l = mode_key
            if LaguerreGaussianBeam is None:
                raise RuntimeError("No beam instance available and lgBeam import missing.")
            beam = LaguerreGaussianBeam(p, l, self.wavelength, self.w0)

        ref_z0 = beam.generate_beam_field(R, PHI, 0.0)
        # propagate numerically to grid z if requested (+ uses angular spectrum function if available)
        if self.angular_prop is None or grid_z == 0.0:
            ref = ref_z0
        else:
            ref = self.angular_prop(ref_z0.copy(), delta, self.wavelength, grid_z)

        # store (aperture-unmasked) reference in cache
        self._ref_cache[key] = ref.copy()
        return ref

    def project_field(self, E_rx, grid_info, receiver_radius=None, tx_frame=None):
        """
        Project single complex field (E_rx) onto modes in self.spatial_modes.
        - grid_info: from frame.grid_info
        - tx_frame: optional FSO_MDM_Frame to pull beam instances / pilot positions
        Returns dict mapping mode_key -> complex projection symbol (per-slice).
        """
        X, Y, delta, x, y = reconstruct_grid_from_gridinfo(grid_info)
        R = np.sqrt(X ** 2 + Y ** 2)

        # if E_rx is intensity-only (real, >=0) attempt to construct amplitude; warn user
        if not np.iscomplexobj(E_rx):
            warnings.warn("E_rx appears to be real (intensity). Assuming zero-phase sqrt(I) field for projection.")
            E_rx = np.sqrt(np.abs(E_rx)).astype(np.complex128)

        dA = float(delta ** 2)
        if receiver_radius is not None:
            aperture_mask = (R <= receiver_radius).astype(float)
        else:
            aperture_mask = np.ones_like(R, dtype=float)

        symbols = {}
        N = X.shape[0]

        for mode_key in self.spatial_modes:
            tx_beam_obj = None
            if tx_frame is not None:
                # try to get beam instance stored in frame.tx_signals
                sig = tx_frame.tx_signals.get(mode_key)
                if sig is not None:
                    tx_beam_obj = sig.get("beam", None)

            ref = self.reference_field(mode_key, X, Y, delta, grid_z=self.z_distance, tx_beam_obj=tx_beam_obj)
            ref_ap = ref * aperture_mask
            ref_energy = np.sum(np.abs(ref_ap) ** 2) * dA
            projection = np.sum(E_rx * np.conj(ref_ap)) * dA
            if ref_energy > 1e-20:
                symbols[mode_key] = projection / ref_energy
            else:
                symbols[mode_key] = 0.0 + 0.0j
        return symbols

    def extract_symbols_sequence(self, E_rx_sequence, grid_info, receiver_radius=None, tx_frame=None):
        """
        Accepts E_rx_sequence as:
          - list/np.ndarray of 2D complex fields (n_frames x N x N)
          - or single 2D field -> returns single-column arrays
        Returns symbols_per_mode: dict mode_key -> complex array (len = n_frames)
        """
        # convert to array-like
        seq = np.asarray(E_rx_sequence)
        if seq.ndim == 2:
            seq = seq[np.newaxis, ...]  # shape (1, N, N)
        n_frames = seq.shape[0]
        symbols_per_mode = {mode: np.zeros(n_frames, dtype=complex) for mode in self.spatial_modes}
        for i in range(n_frames):
            symbols_snapshot = self.project_field(seq[i], grid_info, receiver_radius, tx_frame=tx_frame)
            for mode in self.spatial_modes:
                symbols_per_mode[mode][i] = symbols_snapshot.get(mode, 0.0 + 0.0j)
        return symbols_per_mode


# -------------------------------------------------------
# Channel estimator (LS + optional MMSE fallback)
# -------------------------------------------------------
class ChannelEstimator:
    """
    LS channel estimator using pilot symbols. Expects tx_signals shaped per encodingRunner frame.
    """

    def __init__(self, pilot_handler: PilotHandler, spatial_modes):
        self.pilot_handler = pilot_handler
        self.spatial_modes = list(spatial_modes)
        self.M = len(self.spatial_modes)
        self.H_est = None
        self.noise_var_est = None

    def _gather_pilots(self, rx_symbols_per_mode: Dict[Tuple[int,int], np.ndarray],
                       tx_frame: FSO_MDM_Frame):
        """
        Construct Y_pilot (M x n_p) and P_pilot (M x n_p) aligned on valid pilot indices within frame length.
        Uses pilot positions from tx_frame.tx_signals[mode]['pilot_positions'] if present; else uses self.pilot_handler.pilot_positions.
        """
        # verify we have pilots on TX frame if provided
        first_mode = self.spatial_modes[0]
        tx_signals = tx_frame.tx_signals if tx_frame is not None else {}
        # determine pilot positions (global): try tx_signals first
        pilot_positions = None
        for mode_key in self.spatial_modes:
            sig = tx_signals.get(mode_key)
            if sig is not None and "pilot_positions" in sig:
                pilot_positions = np.asarray(sig["pilot_positions"])
                break
        if pilot_positions is None:
            # fallback to pilot_handler's last-known positions
            pilot_positions = np.asarray(self.pilot_handler.pilot_positions) if self.pilot_handler.pilot_positions is not None else np.array([], dtype=int)

        # ensure integer positions and valid length
        if pilot_positions is None or len(pilot_positions) == 0:
            return None, None, np.array([], dtype=int)

        # gather min rx frame length
        min_rx_len = min([len(rx_symbols_per_mode[mk]) for mk in self.spatial_modes])
        valid_pos = pilot_positions[pilot_positions < min_rx_len]
        if len(valid_pos) == 0:
            return None, None, np.array([], dtype=int)

        n_p = len(valid_pos)
        Y_p = np.zeros((self.M, n_p), dtype=complex)
        P_p = np.zeros((self.M, n_p), dtype=complex)

        for idx, mk in enumerate(self.spatial_modes):
            Y_p[idx, :] = rx_symbols_per_mode[mk][valid_pos]
            # TX symbols must be retrievable from tx_frame
            if tx_frame is None or mk not in tx_frame.tx_signals:
                raise ValueError("tx_frame with tx_signals required for LS channel estimation (to provide pilot symbols).")
            tx_syms = tx_frame.tx_signals[mk]["symbols"]
            P_p[idx, :] = tx_syms[valid_pos]

        return Y_p, P_p, valid_pos

    def estimate_channel_ls(self, rx_symbols_per_mode: Dict[Tuple[int, int], np.ndarray], tx_frame: FSO_MDM_Frame):
        Y_p, P_p, pilot_pos = self._gather_pilots(rx_symbols_per_mode, tx_frame)
        if Y_p is None or P_p is None or P_p.size == 0:
            warnings.warn("No valid pilots found for LS channel estimation. Returning identity H.")
            self.H_est = np.eye(self.M, dtype=complex)
            return self.H_est

        # LS: H = Y_p * P_p^H * (P_p P_p^H)^{-1}
        try:
            PPH = P_p @ P_p.conj().T
            cond = np.linalg.cond(PPH)
            if cond > 1e6:
                warnings.warn(f"Pilot Gram matrix ill-conditioned (cond={cond:.2e}), using pseudo-inverse.")
                H = Y_p @ pinv(P_p)
            else:
                H = Y_p @ P_p.conj().T @ inv(PPH)
        except np.linalg.LinAlgError:
            warnings.warn("Matrix inversion failed; using pseudo-inverse for channel estimate.")
            H = Y_p @ pinv(P_p)

        # regularize if extremely small determinant (avoid unstable ZF)
        if np.linalg.cond(H) > 1e8:
            reg = 1e-6
            H = H @ inv(H + reg * np.eye(self.M))

        self.H_est = H
        return H

    def estimate_noise_variance(self, rx_symbols_per_mode: Dict[Tuple[int,int], np.ndarray],
                                tx_frame: FSO_MDM_Frame, H_est: np.ndarray):
        # compute residual on pilot positions
        Y_p, P_p, pilot_pos = self._gather_pilots(rx_symbols_per_mode, tx_frame)
        if Y_p is None or P_p is None or P_p.size == 0:
            self.noise_var_est = 1e-6
            return self.noise_var_est
        residual = Y_p - H_est @ P_p
        noise_var = np.mean(np.abs(residual) ** 2)
        noise_var = max(noise_var, 1e-12)
        self.noise_var_est = noise_var
        return noise_var


# -------------------------------------------------------
# FSORx: Full receiver pipeline (ZF/MMSE + LDPC decode)
# -------------------------------------------------------
class FSORx:
    def __init__(self, spatial_modes, wavelength, w0, z_distance,
                 pilot_handler: PilotHandler,
                 ldpc_instance: Optional[PyLDPCWrapper] = None,
                 eq_method: str = "zf", receiver_radius: Optional[float] = None):
        self.spatial_modes = list(spatial_modes)
        self.n_modes = len(self.spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
        self.pilot_handler = pilot_handler
        self.eq_method = eq_method.lower()
        self.receiver_radius = receiver_radius

        # QPSK mapping must match encoding.QPSKModulator
        self.qpsk = QPSKModulator(symbol_energy=1.0)

        # LDPC: prefer shared instance from transmitter for exact parity matrix
        if ldpc_instance is not None:
            self.ldpc = ldpc_instance
        else:
            # try to construct default wrapper (requires pyldpc)
            try:
                self.ldpc = PyLDPCWrapper(n=2048, rate=0.8, dv=2, dc=8, seed=42)
                warnings.warn("No LDPC instance provided; receiver created local PyLDPCWrapper that may not match TX.")
            except Exception as e:
                raise RuntimeError(f"Cannot construct LDPC wrapper. Provide ldpc_instance from transmitter. Error: {e}")

        self.demux = OAMDemultiplexer(self.spatial_modes, self.wavelength, self.w0, self.z_distance)
        self.chan_est = ChannelEstimator(self.pilot_handler, self.spatial_modes)
        self.metrics = {}

    def receive_frame(self, rx_field_sequence, tx_frame: FSO_MDM_Frame,
                      original_data_bits: np.ndarray, verbose: bool = True):
        """
        Main receiver routine:
          - rx_field_sequence: array-like of complex fields (n_frames x N x N) or single 2D field
          - tx_frame: FSO_MDM_Frame produced by encodingRunner.transmit(...)
          - original_data_bits: ground-truth info bits for final BER calculation
        Returns: decoded_bits (1D np.array int), metrics dict
        """
        if verbose:
            print("\n" + "=" * 72)
            print("FSO-OAM Receiver: Start")
            print("=" * 72)

        # grid info from tx_frame (required)
        grid_info = tx_frame.grid_info
        if grid_info is None:
            raise ValueError("tx_frame.grid_info required for demux/projection.")

        # 1) Demultiplex: projection
        if verbose: print("1) OAM demultiplexing (projection)...")
        rx_symbols_per_mode = self.demux.extract_symbols_sequence(rx_field_sequence, grid_info,
                                                                   receiver_radius=self.receiver_radius,
                                                                   tx_frame=tx_frame)
        if verbose:
            first_mode = self.spatial_modes[0]
            print(f"   Extracted {len(rx_symbols_per_mode[first_mode])} symbols per mode (incl. pilots).")

        # 2) Channel estimation (LS using pilots)
        if verbose: print("2) Channel estimation (LS using pilots)...")
        H_est = self.chan_est.estimate_channel_ls(rx_symbols_per_mode, tx_frame)
        if verbose:
            print("   H_est magnitude (rows):")
            for row in np.abs(H_est):
                print("     [" + " ".join(f"{v:.3f}" for v in row) + "]")
            print(f"   cond(H_est) = {np.linalg.cond(H_est):.2e}")

        # 3) Noise estimate from pilot residuals
        if verbose: print("3) Noise variance estimation...")
        noise_var = self.chan_est.estimate_noise_variance(rx_symbols_per_mode, tx_frame, H_est)
        if verbose:
            print(f"   Estimated noise variance σ² = {noise_var:.3e}")

        # 4) Separate pilots and data symbols (use pilot positions from tx_frame)
        if verbose: print("4) Separate pilots and data")
        # determine pilot positions array from tx_frame
        pilot_positions = None
        for mk in self.spatial_modes:
            sig = tx_frame.tx_signals.get(mk)
            if sig is not None and "pilot_positions" in sig:
                pilot_positions = np.asarray(sig["pilot_positions"])
                break
        if pilot_positions is None:
            pilot_positions = np.asarray(self.pilot_handler.pilot_positions) if self.pilot_handler.pilot_positions is not None else np.array([], dtype=int)

        # length check and data mask
        first_mode = self.spatial_modes[0]
        total_rx_symbols = len(rx_symbols_per_mode[first_mode])
        data_mask = np.ones(total_rx_symbols, dtype=bool)
        if pilot_positions is not None and pilot_positions.size > 0:
            valid_pilots = pilot_positions[pilot_positions < total_rx_symbols]
            data_mask[valid_pilots] = False

        # Build Y_data matrix: M x Ndata (truncate to common min)
        rx_data_per_mode = {mk: rx_symbols_per_mode[mk][data_mask] for mk in self.spatial_modes}
        data_lengths = [len(v) for v in rx_data_per_mode.values()]
        if len(set(data_lengths)) > 1:
            warnings.warn("Uneven data counts across modes; truncating to minimum length.")
            min_len = min(data_lengths)
            for mk in self.spatial_modes:
                rx_data_per_mode[mk] = rx_data_per_mode[mk][:min_len]
        if data_lengths and data_lengths[0] == 0:
            raise ValueError("No data symbols available after removing pilots.")

        Y_data = np.vstack([rx_data_per_mode[mk] for mk in self.spatial_modes])  # shape M x Ndata
        N_data = Y_data.shape[1]
        if verbose:
            print(f"   Data symbols per mode: {N_data}")

        # 5) Equalize (ZF or MMSE). Use reg if ill-conditioned.
        if verbose: print("5) Equalization")
        H = H_est.copy()
        # if nearly singular, do MMSE instead
        try:
            cond_H = np.linalg.cond(H)
        except Exception:
            cond_H = np.inf
        if self.eq_method == "auto":
            use_mmse = cond_H > 1e4
        else:
            use_mmse = (self.eq_method == "mmse")

        if not use_mmse:
            try:
                W_zf = inv(H)
                S_est = W_zf @ Y_data
            except np.linalg.LinAlgError:
                warnings.warn("ZF inversion failed; switching to pseudo-inverse.")
                W_zf = pinv(H)
                S_est = W_zf @ Y_data
        else:
            # MMSE: W = (H^H H + σ² I)^{-1} H^H
            sigma2 = max(noise_var, 1e-12)
            try:
                W_mmse = inv(H.conj().T @ H + sigma2 * np.eye(self.n_modes)) @ H.conj().T
                S_est = W_mmse @ Y_data
            except np.linalg.LinAlgError:
                warnings.warn("MMSE matrix inversion failed; fallback to pinv(H).")
                W_mmse = pinv(H).conj().T
                S_est = W_mmse @ Y_data

        if verbose:
            print(f"   Equalized symbols shape: {S_est.shape} (modes x symbols)")
            print(f"   Sample post-eq symbol (mode 0, first 5): {S_est[0, :5]}")

        # 6) Demodulate: choose hard vs soft depending on noise variance
        if verbose: print("6) Demodulation (QPSK)")
        s_est_flat = S_est.flatten()
        IDEAL_THRESHOLD = 1e-4
        if noise_var < IDEAL_THRESHOLD:
            if verbose: print("   Low noise: hard decisions.")
            received_bits = self.qpsk.demodulate_hard(s_est_flat)
        else:
            if verbose: print("   Using soft LLRs for demodulation.")
            llrs = self.qpsk.demodulate_soft(s_est_flat, noise_var)
            # convert LLR->hard: sign convention in encoding uses llr<0 -> bit=1 in FSORx earlier;
            # encodingRunner used QPSK.demodulate_hard for generating tx coded bits, so we'll use llr threshold to produce bits for hard-LDPC decode path.
            received_bits = (llrs < 0).astype(int)

        if verbose:
            print(f"   Demodulated coded bits: {len(received_bits)}")

        # 7) LDPC decode (BP if LLRs available else hard)
        if verbose: print("7) LDPC decoding")
        decoded_info_bits = np.array([], dtype=int)
        # prefer BP decode if using soft LLRs
        if noise_var >= IDEAL_THRESHOLD:
            # Use decode_bp on llrs if available (we have llrs above)
            try:
                decoded_info_bits = self.ldpc.decode_bp(llrs)
                if verbose:
                    print(f"   Decoded info bits (BP): {len(decoded_info_bits)}")
            except Exception as e:
                warnings.warn(f"BP decode failed: {e}; falling back to hard decode.")
                decoded_info_bits = self.ldpc.decode_hard(received_bits)
        else:
            try:
                decoded_info_bits = self.ldpc.decode_hard(received_bits)
                if verbose:
                    print(f"   Decoded info bits (hard): {len(decoded_info_bits)}")
            except Exception as e:
                warnings.warn(f"Hard LDPC decode failed: {e}")
                decoded_info_bits = np.array([], dtype=int)

        # 8) BER and metrics
        if verbose: print("8) Performance metrics (BER)")
        orig = np.asarray(original_data_bits, dtype=int)
        L_orig = len(orig)
        L_rec = len(decoded_info_bits)
        compare_len = min(L_orig, L_rec)
        if compare_len == 0 and L_orig > 0:
            bit_errors = L_orig
            ber = 1.0
        else:
            trimmed_orig = orig[:compare_len]
            trimmed_rec = decoded_info_bits[:compare_len]
            bit_errors_common = np.sum(trimmed_orig != trimmed_rec) if compare_len > 0 else 0
            # count length mismatch as errors
            len_mismatch = abs(L_orig - L_rec)
            bit_errors = bit_errors_common + len_mismatch
            ber = bit_errors / L_orig if L_orig > 0 else 0.0

        # store metrics
        self.metrics = {
            "H_est": H_est,
            "noise_var": noise_var,
            "bit_errors": int(bit_errors),
            "total_bits": int(L_orig),
            "ber": float(ber),
            "n_data_symbols": int(N_data),
            "n_modes": int(self.n_modes),
            "cond_H": float(np.linalg.cond(H_est))
        }

        if verbose:
            print(f"   Original bits: {L_orig}, Decoded bits: {L_rec}, Errors: {bit_errors}, BER={ber:.3e}")
            print("=" * 72)

        return decoded_info_bits, self.metrics


# -------------------------------------------------------
# Example helper (diagnostic plotting)
# -------------------------------------------------------
def plot_constellation(rx_symbols, title="Received Constellation"):
    plt.figure(figsize=(5, 5))
    plt.plot(rx_symbols.real, rx_symbols.imag, ".", alpha=0.6)
    plt.axhline(0, color="grey"); plt.axvline(0, color="grey")
    plt.title(title); plt.xlabel("I"); plt.ylabel("Q"); plt.axis("equal")
    plt.grid(True)
    plt.show()


# ---------------------------
# If module run directly, provide a small sanity check sketch (no I/O)
# ---------------------------
if __name__ == "__main__":
    print("")