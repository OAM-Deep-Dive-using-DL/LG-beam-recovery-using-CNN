# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.linalg import inv, pinv
# import warnings

# # --- Imports from other files ---
# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     SCRIPT_DIR = os.getcwd()
# sys.path.insert(0, SCRIPT_DIR)

# try:
#     from lgBeam import LaguerreGaussianBeam
#     # Imports are now known to be correct
#     from encoding import QPSKModulator, SimplifiedLDPC, PilotHandler
# except ImportError as e:
#     print(f"Error importing required modules: {e}")
#     print(f"Please ensure lgBeam.py and encoding.py are in the directory: {SCRIPT_DIR}")
#     sys.exit(1)
# # --- End Imports ---

# warnings.filterwarnings('ignore')

# class OAMDemultiplexer:
#     """
#     OAM Demultiplexer using mode projection.
#     """
    
#     def __init__(self, spatial_modes, wavelength, w0, z_distance):
#         self.spatial_modes = spatial_modes
#         self.n_modes = len(spatial_modes)
#         self.wavelength = wavelength
#         self.w0 = w0
#         self.z_distance = z_distance
#         self.reference_beams = {}
        
#         for mode_key in self.spatial_modes:
#             p, l = mode_key
#             self.reference_beams[mode_key] = LaguerreGaussianBeam(
#                 p=p, l=l, wavelength=wavelength, w0=w0
#             )

#     def project_field(self, E_rx, grid_info, receiver_radius=None):
#         """
#         Project received field onto OAM modes.
#         """
#         X = grid_info['X']
#         Y = grid_info['Y']
#         delta = grid_info['delta']
#         R = np.sqrt(X**2 + Y**2)
#         PHI = np.arctan2(Y, X)
#         dA = delta**2

#         if receiver_radius is not None:
#             aperture_mask = (R <= receiver_radius).astype(float)
#         else:
#             aperture_mask = np.ones_like(R)

#         symbols = {}
        
#         for mode_key in self.spatial_modes:
#             beam = self.reference_beams[mode_key]
#             LG_ref_tx = beam.generate_beam_field(R, PHI, z=0)
#             LG_ref_apertured = LG_ref_tx * aperture_mask
#             ref_energy = np.sum(np.abs(LG_ref_apertured)**2) * dA
#             projection = np.sum(E_rx * np.conj(LG_ref_apertured)) * dA

#             if ref_energy > 1e-20:
#                 # Correct normalization
#                 symbols[mode_key] = projection / ref_energy
#             else:
#                 symbols[mode_key] = 0.0 + 0.0j
                
#         return symbols

#     def extract_symbols_sequence(self, E_rx_sequence, grid_info, receiver_radius=None):
#         """
#         Projects a sequence of fields.
#         """
#         symbols_per_mode = {mode_key: [] for mode_key in self.spatial_modes}
#         num_fields = len(E_rx_sequence)

#         for i in range(num_fields):
#             symbols_snapshot = self.project_field(E_rx_sequence[i], grid_info, receiver_radius)
#             for mode_key in self.spatial_modes:
#                 symbols_per_mode[mode_key].append(symbols_snapshot[mode_key])

#         for mode_key in self.spatial_modes:
#             symbols_per_mode[mode_key] = np.array(symbols_per_mode[mode_key])

#         return symbols_per_mode


# class ChannelEstimator:
#     """
#     Channel estimator using LS method with pilots.
#     """
    
#     def __init__(self, pilot_handler, spatial_modes):
#         self.pilot_handler = pilot_handler
#         self.spatial_modes = spatial_modes
#         self.n_modes = len(spatial_modes)
#         self.H_est = None
#         self.noise_var_est = None

#     def estimate_channel_ls(self, rx_symbols_per_mode, tx_symbols_per_mode):
#         """
#         LS channel estimation from pilots.
#         """
#         M = self.n_modes
#         if self.pilot_handler.pilot_positions is None:
#              raise ValueError("Pilot positions not set in handler.")
             
#         pilot_positions = self.pilot_handler.pilot_positions
#         n_pilots = len(pilot_positions)
#         if n_pilots == 0:
#             print("Warning: No pilots found for channel estimation. Returning Identity.")
#             self.H_est = np.eye(M, dtype=complex)
#             return self.H_est

#         # Find the minimum frame length *at the receiver*
#         min_rx_len = min([len(rx_symbols_per_mode[key]) for key in self.spatial_modes])
#         valid_pilot_pos = pilot_positions[pilot_positions < min_rx_len]
#         n_pilots = len(valid_pilot_pos)
        
#         if n_pilots == 0:
#             print("Error: No valid pilots found within the frame length.")
#             return np.eye(M, dtype=complex)

#         Y_pilot = np.zeros((M, n_pilots), dtype=complex)
#         P_pilot = np.zeros((M, n_pilots), dtype=complex)

#         for idx, mode_key in enumerate(self.spatial_modes):
#             Y_pilot[idx, :] = rx_symbols_per_mode[mode_key][valid_pilot_pos]
#             P_pilot[idx, :] = tx_symbols_per_mode[mode_key]['symbols'][valid_pilot_pos]

#         try:
#             P_PH = P_pilot @ P_pilot.conj().T
#             cond_num = np.linalg.cond(P_PH)
#             if cond_num > 1e6: 
#                 print(f"Warning: Pilot matrix potentially ill-conditioned (cond={cond_num:.2e}). Using pseudo-inverse.")
#                 H_est = Y_pilot @ pinv(P_pilot)
#             else:
#                  H_est = Y_pilot @ P_pilot.conj().T @ inv(P_PH)
#         except np.linalg.LinAlgError:
#             print("Warning: Matrix inversion failed. Using pseudo-inverse for H estimation.")
#             H_est = Y_pilot @ pinv(P_pilot)

#         self.H_est = H_est
#         return H_est

#     def estimate_noise_variance(self, rx_symbols_per_mode, tx_symbols_per_mode, H_est):
#         """
#         Estimate noise variance from pilot residuals.
#         """
#         M = self.n_modes
#         pilot_positions = self.pilot_handler.pilot_positions
#         n_pilots = len(pilot_positions)
#         if n_pilots == 0: return 1e-6 

#         # Find the minimum frame length *at the receiver*
#         min_rx_len = min([len(rx_symbols_per_mode[key]) for key in self.spatial_modes])
#         valid_pilot_pos = pilot_positions[pilot_positions < min_rx_len]
#         n_pilots = len(valid_pilot_pos)

#         if n_pilots == 0:
#             print("Warning: No valid pilots for noise estimation.")
#             return 1e-6

#         Y_pilot = np.zeros((M, n_pilots), dtype=complex)
#         P_pilot = np.zeros((M, n_pilots), dtype=complex)

#         for idx, mode_key in enumerate(self.spatial_modes):
#             Y_pilot[idx, :] = rx_symbols_per_mode[mode_key][valid_pilot_pos]
#             P_pilot[idx, :] = tx_symbols_per_mode[mode_key]['symbols'][valid_pilot_pos]

#         if H_est is None or H_est.shape[0] != M:
#             print("Warning: Invalid H_est provided for noise estimation. Returning default.")
#             return 1e-6

#         error = Y_pilot - H_est @ P_pilot
#         noise_var = np.mean(np.abs(error)**2)
#         noise_var = max(noise_var, 1e-9) 

#         self.noise_var_est = noise_var
#         return noise_var


# class FSORx:
#     """
#     Complete FSO-OAM Receiver - SIMPLIFIED ZF-ONLY VERSION
#     """
    
#     def __init__(self, spatial_modes, wavelength, w0, z_distance,
#                  fec_rate=0.8, pilot_handler=None,
#                  eq_method='zf', receiver_radius=None):
#         self.spatial_modes = spatial_modes
#         self.n_modes = len(spatial_modes)
#         self.wavelength = wavelength
#         self.w0 = w0
#         self.z_distance = z_distance
#         self.eq_method = 'zf'
#         self.receiver_radius = receiver_radius

#         if pilot_handler is None:
#              raise ValueError("FSORx requires a PilotHandler instance used by the transmitter.")
#         self.pilot_handler = pilot_handler

#         self.demux = OAMDemultiplexer(spatial_modes, wavelength, w0, z_distance)
#         self.channel_estimator = ChannelEstimator(self.pilot_handler, spatial_modes)
#         self.qpsk = QPSKModulator(symbol_energy=1.0)
#         self.ldpc = SimplifiedLDPC(n=1024, rate=fec_rate) 
#         self.metrics = {} 

#     def receive_sequence(self, E_rx_sequence, grid_info, tx_signals, original_data_bits, verbose=True):
#         """
#         Processes a SEQUENCE of received fields (SIMPLIFIED ZF-ONLY).
#         """
#         if verbose: print("\n" + "="*70 + "\nFSO-OAM RECEIVER PROCESSING (ZF-Only)\n" + "="*70)

#         # 1. OAM Demultiplexing
#         if verbose: print("\n1. OAM Demultiplexing Sequence...")
#         rx_symbols_per_mode = self.demux.extract_symbols_sequence(
#             E_rx_sequence, grid_info, self.receiver_radius
#         )
#         if verbose: print(f"   Extracted symbol streams for {self.n_modes} modes.")
#         first_mode = self.spatial_modes[0]
#         total_rx_symbols = len(rx_symbols_per_mode[first_mode])
#         if verbose: print(f"   Total received symbols per mode stream: {total_rx_symbols}")


#         # 2. Channel Estimation
#         if verbose: print("\n2. Channel Estimation (LS using Pilots)...")
#         H_est = self.channel_estimator.estimate_channel_ls(rx_symbols_per_mode, tx_signals)
#         if verbose:
#             print(f"   Estimated Channel Matrix H_est (Magnitude):")
#             for row in np.abs(H_est):
#                 print(f"     [{' '.join(f'{x:.3f}' for x in row)}]")
#             print(f"   Channel condition number: {np.linalg.cond(H_est):.2f}")

#         # 3. Noise Estimation
#         if verbose: print("\n3. Noise Variance Estimation...")
#         noise_var = self.channel_estimator.estimate_noise_variance(rx_symbols_per_mode, tx_signals, H_est)
#         if verbose: print(f"   Estimated Noise Variance σ² = {noise_var:.6f}")

#         # 4. Separate Pilots and Data
#         if verbose: print("\n4. Separating Pilots and Data Symbols...")
#         rx_data_symbols_per_mode = {}
        
#         # Get pilot positions valid for this frame length
#         valid_pilot_pos = self.pilot_handler.pilot_positions[self.pilot_handler.pilot_positions < total_rx_symbols]
        
#         for mode_key in self.spatial_modes:
#             data_mask = np.ones(total_rx_symbols, dtype=bool)
#             data_mask[valid_pilot_pos] = False
#             rx_data_symbols_per_mode[mode_key] = rx_symbols_per_mode[mode_key][data_mask]
#             if verbose: print(f"   Mode {mode_key}: Found {len(rx_data_symbols_per_mode[mode_key])} data symbols.")

#         # Assemble data symbols into matrix Y_data (M x N_data)
#         data_lengths = [len(v) for v in rx_data_symbols_per_mode.values()]
        
#         if len(set(data_lengths)) > 1:
#             print(f"Warning: Uneven number of data symbols across modes ({set(data_lengths)}). Truncating to minimum.")
#             min_data_len = min(data_lengths)
#             Y_data = np.array([rx_data_symbols_per_mode[mode_key][:min_data_len] for mode_key in self.spatial_modes])
#         elif not data_lengths or data_lengths[0] == 0:
#              print("Error: No data symbols found.")
#              self.metrics = {'ber': 1.0, 'bit_errors': len(original_data_bits), 'total_bits': len(original_data_bits), 'H_est': H_est, 'noise_var': noise_var}
#              return np.array([], dtype=int), self.metrics
#         else:
#              min_data_len = data_lengths[0]
#              Y_data = np.array([rx_data_symbols_per_mode[mode_key] for mode_key in self.spatial_modes])


#         # 5. Zero-Forcing Equalization
#         if verbose: print(f"\n5. Equalizing Data Symbols (ZF)...")
#         try:
#             W_ZF = inv(H_est)
#             S_est_data = W_ZF @ Y_data
#         except np.linalg.LinAlgError:
#             print("Warning: ZF matrix inversion failed. Using pseudo-inverse.")
#             W_ZF = pinv(H_est)
#             S_est_data = W_ZF @ Y_data
            
#         if verbose: print(f"   Equalized {S_est_data.shape[1]} data symbols per mode.")

#         # 6. Demodulation (Soft)
#         if verbose: print("\n6. QPSK Demodulation (Soft)...")
#         # S_est_data is (M x N_data). We must flatten it row-by-row ("C" order)
#         s_est_flat = S_est_data.flatten()
#         llrs = self.qpsk.demodulate_soft(s_est_flat, noise_var)
#         if verbose: print(f"   Generated {len(llrs)} LLRs.")

#         # 7. Decoding
#         if verbose: print("\n7. LDPC Decoding...")
#         received_bits_hard = (llrs < 0).astype(int)

#         n_ldpc = self.ldpc.n # e.g., 1024
#         k_ldpc = self.ldpc.k # e.g., 819
        
#         ### --- THIS IS THE FIX (v7) --- ###
#         # Instead of truncating to 0 codewords, we pad the stream
#         # to the *next* full codeword size. This simulates
#         # "rate matching" and allows the decoder to work.
        
#         num_codewords_full = len(received_bits_hard) // n_ldpc
#         remaining_bits = len(received_bits_hard) % n_ldpc
        
#         decoded_bits_list = []

#         # 1. Decode all the full codewords
#         if num_codewords_full > 0:
#             trunc_len = num_codewords_full * n_ldpc
#             full_blocks = received_bits_hard[:trunc_len]
#             decoded_bits_list.append(self.ldpc.decode_simple(full_blocks))
            
#         # 2. Pad and decode the final, partial codeword
#         if remaining_bits > k_ldpc: 
#             # We have enough bits (e.g., 1010) to *almost* make a
#             # codeword, but not quite.
#             print(f"Warning: Received partial codeword ({remaining_bits} bits). Padding to {n_ldpc} with zeros.")
#             pad_len = n_ldpc - remaining_bits
#             partial_block = received_bits_hard[num_codewords_full * n_ldpc:]
#             padded_block = np.concatenate([partial_block, np.zeros(pad_len, dtype=int)])
#             decoded_bits_list.append(self.ldpc.decode_simple(padded_block))
            
#         elif remaining_bits > 0:
#             # We have some leftover bits (e.g. 50), but not enough to
#             # even attempt decoding. Discard them.
#             print(f"Warning: Discarding {remaining_bits} trailing bits, insufficient for a codeword.")

#         if not decoded_bits_list:
#             print("Error: No full LDPC codewords to decode.")
#             decoded_bits = np.array([], dtype=int)
#         else:
#             decoded_bits = np.concatenate(decoded_bits_list)
#         ### --- END OF FIX --- ###
             
#         if verbose: print(f"   Decoded {len(decoded_bits)} information bits.")

#         # 8. Performance Analysis (BER)
#         if verbose: print("\n8. Performance Analysis...")
        
#         len_original = len(original_data_bits)
#         len_recovered = len(decoded_bits)
        
#         # Find the minimum length to compare
#         compare_len = min(len_original, len_recovered)
        
#         if compare_len == 0 and len_original > 0:
#             print("Error: No bits recovered for comparison.")
#             bit_errors = len_original # Count all as errors
#             ber = 1.0
#             len_mismatch_errors = len_original
#         elif len_original == 0:
#             print("Warning: No original bits to compare against.")
#             bit_errors = 0
#             ber = 0.0
#             len_mismatch_errors = 0
#         else:
#             # Trim both arrays to the minimum length
#             trimmed_original = original_data_bits[:compare_len]
#             trimmed_recovered = decoded_bits[:compare_len]
            
#             # 1. Calculate errors in the common block
#             bit_errors_common = np.sum(trimmed_original != trimmed_recovered)
            
#             # 2. Add errors for any bits that were completely lost (length mismatch)
#             len_mismatch_errors = abs(len_original - len_recovered)
            
#             # 3. Total errors
#             bit_errors = bit_errors_common + len_mismatch_errors
            
#             # 4. BER is total errors divided by the *original* number of bits
#             ber = bit_errors / len_original

#         self.metrics = {
#             'H_est': H_est,
#             'noise_var': noise_var,
#             'bit_errors': bit_errors,
#             'total_bits': len_original, # Report what we *should* have received
#             'ber': ber
#         }
#         if verbose:
#              print(f"   Original Info Bits: {len_original}")
#              print(f"   Recovered Info Bits: {len_recovered}")
#              print(f"   Compared Bits: {compare_len}")
#              print(f"   Bit Errors: {bit_errors} (incl. {len_mismatch_errors} from length mismatch)")
#              print(f"   BER: {ber:.2e}")

#         if verbose: print("\n" + "="*70)

#         # Return the recovered bits for potential further analysis
#         return decoded_bits, self.metrics



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, pinv
from scipy.fft import fft2, ifft2, fftfreq
import warnings

# --- Imports from other files ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

try:
    from lgBeam import LaguerreGaussianBeam
    # Imports are now known to be correct
    from encoding import QPSKModulator, SimplifiedLDPC, PilotHandler
    from turbulence import angular_spectrum_propagation  # Import for numerical propagation
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Please ensure lgBeam.py and encoding.py are in the directory: {SCRIPT_DIR}")
    sys.exit(1)
# --- End Imports ---

warnings.filterwarnings('ignore')

class OAMDemultiplexer:
    """
    OAM Demultiplexer using mode projection.
    """
    
    def __init__(self, spatial_modes, wavelength, w0, z_distance):
        self.spatial_modes = spatial_modes
        self.n_modes = len(spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
        self.reference_beams = {}
        
        for mode_key in self.spatial_modes:
            p, l = mode_key
            self.reference_beams[mode_key] = LaguerreGaussianBeam(
                p=p, l=l, wavelength=wavelength, w0=w0
            )

    def project_field(self, E_rx, grid_info, receiver_radius=None):
        """
        Project received field onto OAM modes.
        
        CRITICAL FIX: The received field E_rx was propagated using numerical
        angular spectrum method. Therefore, the reference beams MUST also be
        propagated numerically to match. Using analytical reference beams causes
        a mismatch that breaks mode projection.
        
        Literature: Mode projection requires the reference basis to match the
        propagation method used for the received field (Goodman 2005, Ch. 3).
        
        Note: E_rx should already have the aperture mask applied (from completeFSOSystem.py).
        We apply the aperture mask to the reference beam to match.
        """
        X = grid_info['X']
        Y = grid_info['Y']
        delta = grid_info['delta']
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)
        dA = delta**2

        if receiver_radius is not None:
            aperture_mask = (R <= receiver_radius).astype(float)
        else:
            aperture_mask = np.ones_like(R)

        symbols = {}
        
        for mode_key in self.spatial_modes:
            beam = self.reference_beams[mode_key]
            
            # CRITICAL FIX: Reference beam must be propagated NUMERICALLY to match
            # the received field, which was propagated using angular_spectrum_propagation.
            # Generate reference at z=0, then propagate numerically to z=distance.
            LG_ref_z0 = beam.generate_beam_field(R, PHI, z=0)
            LG_ref = angular_spectrum_propagation(
                LG_ref_z0.copy(), 
                delta, 
                beam.wavelength, 
                self.z_distance
            )
            
            # Apply aperture mask to reference beam to match E_rx (which already has aperture)
            LG_ref_apertured = LG_ref * aperture_mask
            
            # Compute normalization: <ref, ref> integrated over aperture
            ref_energy = np.sum(np.abs(LG_ref_apertured)**2) * dA
            
            # Compute projection: <E_rx, ref> integrated over aperture
            # Note: E_rx already has aperture mask applied in completeFSOSystem.py
            projection = np.sum(E_rx * np.conj(LG_ref_apertured)) * dA

            if ref_energy > 1e-20:
                # Normalized projection gives the symbol coefficient
                symbols[mode_key] = projection / ref_energy
            else:
                symbols[mode_key] = 0.0 + 0.0j
                
        return symbols

    def extract_symbols_sequence(self, E_rx_sequence, grid_info, receiver_radius=None):
        """
        Projects a sequence of fields.
        """
        symbols_per_mode = {mode_key: [] for mode_key in self.spatial_modes}
        num_fields = len(E_rx_sequence)

        for i in range(num_fields):
            symbols_snapshot = self.project_field(E_rx_sequence[i], grid_info, receiver_radius)
            for mode_key in self.spatial_modes:
                symbols_per_mode[mode_key].append(symbols_snapshot[mode_key])

        for mode_key in self.spatial_modes:
            symbols_per_mode[mode_key] = np.array(symbols_per_mode[mode_key])

        return symbols_per_mode


class ChannelEstimator:
    """
    Channel estimator using LS method with pilots.
    """
    
    def __init__(self, pilot_handler, spatial_modes):
        self.pilot_handler = pilot_handler
        self.spatial_modes = spatial_modes
        self.n_modes = len(spatial_modes)
        self.H_est = None
        self.noise_var_est = None

    def estimate_channel_ls(self, rx_symbols_per_mode, tx_symbols_per_mode):
        """
        LS channel estimation from pilots.
        """
        M = self.n_modes
        if self.pilot_handler.pilot_positions is None:
             raise ValueError("Pilot positions not set in handler.")
             
        pilot_positions = self.pilot_handler.pilot_positions
        n_pilots = len(pilot_positions)
        if n_pilots == 0:
            print("Warning: No pilots found for channel estimation. Returning Identity.")
            self.H_est = np.eye(M, dtype=complex)
            return self.H_est

        # Find the minimum frame length *at the receiver*
        min_rx_len = min([len(rx_symbols_per_mode[key]) for key in self.spatial_modes])
        valid_pilot_pos = pilot_positions[pilot_positions < min_rx_len]
        n_pilots = len(valid_pilot_pos)
        
        if n_pilots == 0:
            print("Error: No valid pilots found within the frame length.")
            return np.eye(M, dtype=complex)

        Y_pilot = np.zeros((M, n_pilots), dtype=complex)
        P_pilot = np.zeros((M, n_pilots), dtype=complex)

        for idx, mode_key in enumerate(self.spatial_modes):
            Y_pilot[idx, :] = rx_symbols_per_mode[mode_key][valid_pilot_pos]
            P_pilot[idx, :] = tx_symbols_per_mode[mode_key]['symbols'][valid_pilot_pos]

        try:
            P_PH = P_pilot @ P_pilot.conj().T
            cond_num = np.linalg.cond(P_PH)
            if cond_num > 1e-6: 
                print(f"Warning: Pilot matrix potentially ill-conditioned (cond={cond_num:.2e}). Using pseudo-inverse.")
                H_est = Y_pilot @ pinv(P_pilot)
            else:
                 H_est = Y_pilot @ P_pilot.conj().T @ inv(P_PH)
        except np.linalg.LinAlgError:
            print("Warning: Matrix inversion failed. Using pseudo-inverse for H estimation.")
            H_est = Y_pilot @ pinv(P_pilot)

        self.H_est = H_est
        return H_est

    def estimate_noise_variance(self, rx_symbols_per_mode, tx_symbols_per_mode, H_est):
        """
        Estimate noise variance from pilot residuals.
        """
        M = self.n_modes
        pilot_positions = self.pilot_handler.pilot_positions
        n_pilots = len(pilot_positions)
        if n_pilots == 0: return 1e-6 

        # Find the minimum frame length *at the receiver*
        min_rx_len = min([len(rx_symbols_per_mode[key]) for key in self.spatial_modes])
        valid_pilot_pos = pilot_positions[pilot_positions < min_rx_len]
        n_pilots = len(valid_pilot_pos)

        if n_pilots == 0:
            print("Warning: No valid pilots for noise estimation.")
            return 1e-6

        Y_pilot = np.zeros((M, n_pilots), dtype=complex)
        P_pilot = np.zeros((M, n_pilots), dtype=complex)

        for idx, mode_key in enumerate(self.spatial_modes):
            Y_pilot[idx, :] = rx_symbols_per_mode[mode_key][valid_pilot_pos]
            P_pilot[idx, :] = tx_symbols_per_mode[mode_key]['symbols'][valid_pilot_pos]

        if H_est is None or H_est.shape[0] != M:
            print("Warning: Invalid H_est provided for noise estimation. Returning default.")
            return 1e-6

        error = Y_pilot - H_est @ P_pilot
        noise_var = np.mean(np.abs(error)**2)
        noise_var = max(noise_var, 1e-9) 

        self.noise_var_est = noise_var
        return noise_var

class FSORx:
    """
    Complete FSO-OAM Receiver - SIMPLIFIED ZF-ONLY VERSION
    """
    
    def __init__(self, spatial_modes, wavelength, w0, z_distance,
                 fec_rate=0.8, pilot_handler=None,
                 eq_method='zf', receiver_radius=None, ldpc_instance=None):
        self.spatial_modes = spatial_modes
        self.n_modes = len(spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.z_distance = z_distance
        self.eq_method = 'zf'
        self.receiver_radius = receiver_radius

        if pilot_handler is None:
             raise ValueError("FSORx requires a PilotHandler instance used by the transmitter.")
        self.pilot_handler = pilot_handler

        self.demux = OAMDemultiplexer(spatial_modes, wavelength, w0, z_distance)
        self.channel_estimator = ChannelEstimator(self.pilot_handler, spatial_modes)
        self.qpsk = QPSKModulator(symbol_energy=1.0)
        # CRITICAL FIX: Use the SAME LDPC instance from transmitter to ensure same H matrix
        # The LDPC encoder generates a random H matrix, so transmitter and receiver must share it
        if ldpc_instance is not None:
            self.ldpc = ldpc_instance
        else:
            self.ldpc = SimplifiedLDPC(n=1024, rate=fec_rate)
            print("Warning: Receiver created new LDPC instance. H matrix may differ from transmitter!")
        self.metrics = {} 

    def receive_sequence(self, E_rx_sequence, grid_info, tx_signals, original_data_bits, verbose=True):
        """
        Processes a SEQUENCE of received fields (SIMPLIFIED ZF-ONLY).
        """
        if verbose: print("\n" + "="*70 + "\nFSO-OAM RECEIVER PROCESSING (ZF-Only)\n" + "="*70)

        # 1. OAM Demultiplexing
        if verbose: print("\n1. OAM Demultiplexing Sequence...")
        rx_symbols_per_mode = self.demux.extract_symbols_sequence(
            E_rx_sequence, grid_info, self.receiver_radius
        )
        if verbose: print(f"   Extracted symbol streams for {self.n_modes} modes.")
        first_mode = self.spatial_modes[0]
        # This is the 'true' length of the simulation, truncated to the minimum
        total_rx_symbols = len(rx_symbols_per_mode[first_mode])
        if verbose: print(f"   Total received symbols per mode stream: {total_rx_symbols}")


        # 2. Channel Estimation
        if verbose: print("\n2. Channel Estimation (LS using Pilots)...")
        H_est = self.channel_estimator.estimate_channel_ls(rx_symbols_per_mode, tx_signals)
        if verbose:
            print(f"   Estimated Channel Matrix H_est (Magnitude):")
            for row in np.abs(H_est):
                print(f"     [{' '.join(f'{x:.3f}' for x in row)}]")
            print(f"   Channel condition number: {np.linalg.cond(H_est):.2f}")

        # 3. Noise Estimation
        if verbose: print("\n3. Noise Variance Estimation...")
        noise_var = self.channel_estimator.estimate_noise_variance(rx_symbols_per_mode, tx_signals, H_est)
        if verbose: print(f"   Estimated Noise Variance σ² = {noise_var:.6f}")

        # 4. Separate Pilots and Data
        if verbose: print("\n4. Separating Pilots and Data Symbols...")
        rx_data_symbols_per_mode = {}
        
        # Get pilot positions valid for this frame length
        valid_pilot_pos = self.pilot_handler.pilot_positions[self.pilot_handler.pilot_positions < total_rx_symbols]
        
        # Create the one, correct data mask
        data_mask = np.ones(total_rx_symbols, dtype=bool)
        if len(valid_pilot_pos) > 0:
            data_mask[valid_pilot_pos] = False

        for mode_key in self.spatial_modes:
            rx_data_symbols_per_mode[mode_key] = rx_symbols_per_mode[mode_key][data_mask]
            if verbose: print(f"   Mode {mode_key}: Found {len(rx_data_symbols_per_mode[mode_key])} data symbols.")

        # Assemble data symbols into matrix Y_data (M x N_data)
        data_lengths = [len(v) for v in rx_data_symbols_per_mode.values()]
        
        if len(set(data_lengths)) > 1:
            print(f"Warning: Uneven number of data symbols across modes ({set(data_lengths)}). Truncating to minimum.")
            min_data_len = min(data_lengths)
            Y_data = np.array([rx_data_symbols_per_mode[mode_key][:min_data_len] for mode_key in self.spatial_modes])
        elif not data_lengths or data_lengths[0] == 0:
             print("Error: No data symbols found.")
             self.metrics = {'ber': 1.0, 'bit_errors': len(original_data_bits), 'total_bits': len(original_data_bits), 'H_est': H_est, 'noise_var': noise_var}
             return np.array([], dtype=int), self.metrics
        else:
             min_data_len = data_lengths[0]
             Y_data = np.array([rx_data_symbols_per_mode[mode_key] for mode_key in self.spatial_modes])


        # 5. Zero-Forcing Equalization
        if verbose: print(f"\n5. Equalizing Data Symbols (ZF)...")
        try:
            W_ZF = inv(H_est)
            S_est_data = W_ZF @ Y_data
        except np.linalg.LinAlgError:
            print("Warning: ZF matrix inversion failed. Using pseudo-inverse.")
            W_ZF = pinv(H_est)
            S_est_data = W_ZF @ Y_data
            
        if verbose: 
            print(f"   Equalized {S_est_data.shape[1]} data symbols per mode.")
            # Diagnostic: Check if equalization is working (compare first few symbols)
            
            # === FIX 1 HERE ===
            if verbose and len(tx_signals) > 0:
                tx_data_symbols = {}
                for mode_key in self.spatial_modes:
                    # Get the actual transmitted symbols for this mode
                    mode_tx_symbols_original = tx_signals[mode_key]['symbols']
                    
                    # *** APPLIED FIX ***
                    # 1. Truncate original TX symbols to match what receiver processed
                    mode_tx_symbols_truncated = mode_tx_symbols_original[:total_rx_symbols]
                    
                    # 2. Apply the correct data_mask (which is total_rx_symbols long)
                    tx_data_symbols[mode_key] = mode_tx_symbols_truncated[data_mask][:min_data_len]
                
                tx_data_matrix = np.array([tx_data_symbols[mode_key][:min_data_len] for mode_key in self.spatial_modes])
                print(f"   TX data symbols (first mode, first 5): {tx_data_matrix[0, :5]}")
                print(f"   RX equalized symbols (first mode, first 5): {S_est_data[0, :5]}")
                print(f"   Symbol error (first mode, first 5): {np.abs(tx_data_matrix[0, :5] - S_est_data[0, :5])}")
            # === END FIX 1 ===

        # 6. Demodulation
        if verbose: print("\n6. QPSK Demodulation...")
        s_est_flat = S_est_data.flatten()
        
        IDEAL_CHANNEL_NOISE_THRESHOLD = 1e-4 
        
        if noise_var < IDEAL_CHANNEL_NOISE_THRESHOLD:
            if verbose: 
                print(f"   Noise variance ({noise_var:.2e}) is negligible. Using HARD decision demodulation.")
            received_bits_hard = self.qpsk.demodulate_hard(s_est_flat)
        else:
            if verbose: 
                print(f"   Noise variance ({noise_var:.2e}) is significant. Using SOFT decision demodulation.")
            llrs = self.qpsk.demodulate_soft(s_est_flat, noise_var)
            received_bits_hard = (llrs < 0).astype(int)
        
        if verbose: 
            print(f"   Generated {len(received_bits_hard)} coded bits.")

            # === FIX 2 HERE ===
            # Diagnostic: Compare transmitted vs received coded bits
            if verbose and len(tx_signals) > 0:
                # Get transmitted coded bits (before pilots)
                tx_coded_bits_list = []
                for mode_key in self.spatial_modes:
                    # Get the original transmitted symbols
                    mode_symbols_original = tx_signals[mode_key]['symbols']

                    # *** APPLIED FIX ***
                    # 1. Truncate original TX symbols to match what receiver processed
                    mode_symbols_truncated = mode_symbols_original[:total_rx_symbols]

                    # 2. Apply the correct data_mask (which is total_rx_symbols long)
                    mode_data_symbols = mode_symbols_truncated[data_mask][:min_data_len]
                    
                    # Demodulate to bits
                    mode_bits = self.qpsk.demodulate_hard(mode_data_symbols)
                    tx_coded_bits_list.append(mode_bits)
                tx_coded_bits = np.concatenate(tx_coded_bits_list)
                
                # Compare with received
                compare_len_coded = min(len(tx_coded_bits), len(received_bits_hard))
                coded_bit_errors = np.sum(tx_coded_bits[:compare_len_coded] != received_bits_hard[:compare_len_coded])
                coded_ber = coded_bit_errors / compare_len_coded if compare_len_coded > 0 else 0
                print(f"   Coded bits comparison: {coded_bit_errors}/{compare_len_coded} errors (BER: {coded_ber:.6f})")
            # === END FIX 2 ===


        # 7. Decoding
        if verbose: print("\n7. LDPC Decoding...")
        
        n_ldpc = self.ldpc.n # e.g., 1024
        k_ldpc = self.ldpc.k # e.g., 819
        
        num_codewords_full = len(received_bits_hard) // n_ldpc
        remaining_bits = len(received_bits_hard) % n_ldpc
        
        if verbose:
            print(f"   Received coded bits: {len(received_bits_hard)}")
            print(f"   Full codewords: {num_codewords_full}, Remaining bits: {remaining_bits}")
        
        decoded_bits_list = []

        # 1. Decode all the full codewords
        if num_codewords_full > 0:
            trunc_len = num_codewords_full * n_ldpc
            full_blocks = received_bits_hard[:trunc_len]
            
            # === FIX 3 HERE ===
            # Diagnostic: Check syndromes before decoding
            if verbose and len(tx_signals) > 0:
                # Get transmitted coded bits for comparison
                tx_coded_bits_list = []
                for mode_key in self.spatial_modes:
                    # Get the original transmitted symbols
                    mode_symbols_original = tx_signals[mode_key]['symbols']

                    # *** APPLIED FIX ***
                    # 1. Truncate original TX symbols to match what receiver processed
                    mode_symbols_truncated = mode_symbols_original[:total_rx_symbols]

                    # 2. Apply the correct data_mask (which is total_rx_symbols long)
                    mode_data_symbols = mode_symbols_truncated[data_mask][:min_data_len]
                    
                    mode_bits = self.qpsk.demodulate_hard(mode_data_symbols)
                    tx_coded_bits_list.append(mode_bits)
                tx_coded_bits = np.concatenate(tx_coded_bits_list)
                
                tx_coded_trunc = tx_coded_bits[:trunc_len]
                
                # Check syndromes for each codeword
                syndromes_before = []
                for i in range(num_codewords_full):
                    rx_block = full_blocks[i*n_ldpc:(i+1)*n_ldpc]
                    tx_block = tx_coded_trunc[i*n_ldpc:(i+1)*n_ldpc]
                    rx_syndrome = np.sum(np.dot(self.ldpc.H, rx_block) % 2)
                    tx_syndrome = np.sum(np.dot(self.ldpc.H, tx_block) % 2)
                    syndromes_before.append((rx_syndrome, tx_syndrome))
                    
                    # Check if blocks match
                    block_errors = np.sum(rx_block != tx_block)
                    if block_errors > 0:
                        print(f"   Codeword {i}: {block_errors} bit errors, RX syndrome={rx_syndrome}, TX syndrome={tx_syndrome}")
                
                if all(s[0] == 0 and s[1] == 0 for s in syndromes_before):
                    print(f"   All {num_codewords_full} codewords have zero syndrome (perfect)")
            # === END FIX 3 ===
            
            decoded_bits_list.append(self.ldpc.decode_simple(full_blocks))
            
            # Diagnostic: Compare decoded bits from perfect coded bits vs received coded bits
            if verbose and len(tx_signals) > 0:
                decoded_from_perfect = self.ldpc.decode_simple(tx_coded_trunc)
                decoded_from_received = decoded_bits_list[0]
                
                compare_len_decoded = min(len(decoded_from_perfect), len(decoded_from_received))
                if compare_len_decoded > 0:
                    decoded_errors = np.sum(decoded_from_perfect[:compare_len_decoded] != decoded_from_received[:compare_len_decoded])
                    if decoded_errors > 0:
                        print(f"   WARNING: Decoder produces {decoded_errors} errors when comparing perfect vs received coded bits")
                        print(f"   This suggests decoder H matrix mismatch or decoder bug")
                    else:
                        print(f"   Decoder produces identical output from perfect and received coded bits")
                
                # Store tx_coded_trunc for later diagnostic
                self._tx_coded_bits_diag = tx_coded_trunc
            
        # 2. Pad and decode the final, partial codeword
        if remaining_bits > k_ldpc: 
            print(f"Warning: Received partial codeword ({remaining_bits} bits). Padding to {n_ldpc} with zeros.")
            pad_len = n_ldpc - remaining_bits
            partial_block = received_bits_hard[num_codewords_full * n_ldpc:]
            padded_block = np.concatenate([partial_block, np.zeros(pad_len, dtype=int)])
            decoded_bits_list.append(self.ldpc.decode_simple(padded_block))
            
        elif remaining_bits > 0:
            print(f"Warning: Discarding {remaining_bits} trailing bits, insufficient for a codeword.")

        if not decoded_bits_list:
            print("Error: No full LDPC codewords to decode.")
            decoded_bits = np.array([], dtype=int)
        else:
            decoded_bits = np.concatenate(decoded_bits_list)
             
        if verbose: print(f"   Decoded {len(decoded_bits)} information bits.")

        # 8. Performance Analysis (BER)
        if verbose: print("\n8. Performance Analysis...")
        
        len_original = len(original_data_bits)
        len_recovered = len(decoded_bits)
        
        # Diagnostic: Check if original_data_bits matches what was actually encoded
        if verbose and len(tx_signals) > 0 and hasattr(self, '_tx_coded_bits_diag'):
            decoded_from_perfect = self.ldpc.decode_simple(self._tx_coded_bits_diag)
            perfect_errors = np.sum(original_data_bits[:min(len(original_data_bits), len(decoded_from_perfect))] != decoded_from_perfect[:min(len(original_data_bits), len(decoded_from_perfect))])
            print(f"   Diagnostic: Errors when comparing original_data_bits to decoded_from_perfect: {perfect_errors}")
            if perfect_errors > 0:
                print(f"   ✗ CRITICAL: original_data_bits does NOT match what was actually encoded!")
                print(f"   First 20 original:      {original_data_bits[:20]}")
                print(f"   First 20 decoded_perf:  {decoded_from_perfect[:20]}")
                print(f"   This explains the BER mismatch!")
            else:
                print(f"   ✓ original_data_bits matches decoded_from_perfect")
        
        # Find the minimum length to compare
        compare_len = min(len_original, len_recovered)
        
        if compare_len == 0 and len_original > 0:
            print("Error: No bits recovered for comparison.")
            bit_errors = len_original # Count all as errors
            ber = 1.0
            len_mismatch_errors = len_original
        elif len_original == 0:
            print("Warning: No original bits to compare against.")
            bit_errors = 0
            ber = 0.0
            len_mismatch_errors = 0
        else:
            # Trim both arrays to the minimum length
            trimmed_original = original_data_bits[:compare_len]
            trimmed_recovered = decoded_bits[:compare_len]
            
            # 1. Calculate errors in the common block
            bit_errors_common = np.sum(trimmed_original != trimmed_recovered)
            
            # 2. Add errors for any bits that were completely lost (length mismatch)
            len_mismatch_errors = abs(len_original - len_recovered)
            
            # 3. Total errors
            bit_errors = bit_errors_common + len_mismatch_errors
            
            # 4. BER is total errors divided by the *original* number of bits
            ber = bit_errors / len_original

        self.metrics = {
            'H_est': H_est,
            'noise_var': noise_var,
            'bit_errors': bit_errors,
            'total_bits': len_original, # Report what we *should* have received
            'ber': ber
        }
        if verbose:
             print(f"   Original Info Bits: {len_original}")
             print(f"   Recovered Info Bits: {len_recovered}")
             print(f"   Compared Bits: {compare_len}")
             print(f"   Bit Errors: {bit_errors} (incl. {len_mismatch_errors} from length mismatch)")
             print(f"   BER: {ber:.2e}")

        if verbose: print("\n" + "="*70)

        # Return the recovered bits for potential further analysis
        return decoded_bits, self.metrics