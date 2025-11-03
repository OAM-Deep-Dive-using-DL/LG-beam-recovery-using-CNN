import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

try:
    from lgBeam import LaguerreGaussianBeam 
except ImportError:
    print(f"Error: Could not find 'lgBeam.py' in the script directory: {SCRIPT_DIR}")
    print("Please make sure the lgBeam.py file is saved in the same folder.")
    sys.exit(1)

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Parameters (moved to main) ---
# WAVELENGTH = 1550e-9  
# W0 = 25e-3  
# Z_PROPAGATION = 1000
# # OAM_MODES = [-4, -2, 2, 4]  # <-- RECTIFIED: This will be changed to SPATIAL_MODES
# FEC_RATE = 0.8  
# ...etc...


# --- All classes (QPSKModulator, SimplifiedLDPC, PilotHandler) are IDENTICAL. ---
# --- No changes are needed to them, so they are omitted for brevity. ---
# ... (QPSKModulator class code) ...
# ... (SimplifiedLDPC class code) ...
# ... (PilotHandler class code) ...

class QPSKModulator:
    """
    QPSK (Quadrature Phase-Shift Keying) Modulator with Gray coding.
    
    Standard QPSK constellation with Gray coding minimizes bit errors:
    - 00 -> (1+j)/√2   (I=+, Q=+)
    - 01 -> (-1+j)/√2  (I=-, Q=+)
    - 11 -> (-1-j)/√2  (I=-, Q=-)
    - 10 -> (1-j)/√2   (I=+, Q=-)
    
    Adjacent constellation points differ by only 1 bit (Gray code property).
    
    References:
    - Proakis & Salehi, "Digital Communications" (2008), Ch. 4
    - Sklar, "Digital Communications" (2001), Ch. 4
    """
    def __init__(self, symbol_energy=1.0):
        """
        Parameters:
        -----------
        symbol_energy : float
            Average symbol energy Es [Joules]. Default 1.0.
        """
        self.Es = symbol_energy
        self.A = np.sqrt(symbol_energy)

        # Gray-coded QPSK constellation map
        # Bit order: (I_bit, Q_bit) where I_bit is MSB, Q_bit is LSB
        self.constellation_map = {
            (0, 0): self.A * (1 + 1j) / np.sqrt(2),   # 00 -> I=+, Q=+
            (0, 1): self.A * (-1 + 1j) / np.sqrt(2),  # 01 -> I=-, Q=+
            (1, 1): self.A * (-1 - 1j) / np.sqrt(2),  # 11 -> I=-, Q=-
            (1, 0): self.A * (1 - 1j) / np.sqrt(2)    # 10 -> I=+, Q=-
        }
        self.constellation_points = np.array(list(self.constellation_map.values()))

    def modulate(self, bits):
        """
        Maps bits to QPSK symbols using Gray coding.
        
        Parameters:
        -----------
        bits : array_like
            Input bits (will be padded with 0 if odd length)
        
        Returns:
        --------
        symbols : ndarray
            QPSK symbols (complex, length = ceil(len(bits)/2))
        """
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([self.constellation_map[tuple(pair)] for pair in bit_pairs])
        return symbols

    def demodulate_hard(self, rx_symbols):
        """
        Hard decision demodulation: maps received symbols to bits using
        minimum Euclidean distance to constellation points.
        
        Parameters:
        -----------
        rx_symbols : array_like
            Received QPSK symbols (complex)
        
        Returns:
        --------
        bits : ndarray
            Demodulated bits (length = 2 * len(rx_symbols))
        """
        bits = []
        for symbol in rx_symbols:
            distances = np.abs(self.constellation_points - symbol)
            min_idx = np.argmin(distances)
            for key, val in self.constellation_map.items():
                if np.isclose(val, self.constellation_points[min_idx]):
                    bits.extend(list(key))
                    break
        return np.array(bits)

    def demodulate_soft(self, rx_symbols, noise_var):
        """
        Soft decision demodulation: computes Log-Likelihood Ratios (LLRs)
        for each bit using maximum a posteriori (MAP) detection.
        
        Uses log-sum-exp trick for numerical stability.
        
        Parameters:
        -----------
        rx_symbols : array_like
            Received QPSK symbols (complex)
        noise_var : float
            Noise variance σ²
        
        Returns:
        --------
        llrs : ndarray
            Log-Likelihood Ratios for each bit (length = 2 * len(rx_symbols))
            Positive LLR favors bit=0, negative LLR favors bit=1
        """
        llrs = []
        for symbol in rx_symbols:
            for bit_pos in range(2):
                # Find constellation points with bit=0 and bit=1 at position bit_pos
                points_bit0 = [s for b, s in self.constellation_map.items() if b[bit_pos] == 0]
                points_bit1 = [s for b, s in self.constellation_map.items() if b[bit_pos] == 1]
                
                # Compute log-likelihoods (negative squared distances / noise_var)
                metrics_0 = np.array([-np.abs(symbol - s)**2 / noise_var for s in points_bit0])
                metrics_1 = np.array([-np.abs(symbol - s)**2 / noise_var for s in points_bit1])
                
                # Log-sum-exp trick for numerical stability
                max_0 = np.max(metrics_0)
                max_1 = np.max(metrics_1)
                llr = (max_0 + np.log(np.sum(np.exp(metrics_0 - max_0))) -
                       max_1 - np.log(np.sum(np.exp(metrics_1 - max_1))))
                llrs.append(llr)
        return np.array(llrs)

    def plot_constellation(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        for bits, symbol in self.constellation_map.items():
            ax.plot(symbol.real, symbol.imag, 'ro', markersize=15)
            ax.annotate(f'{bits[0]}{bits[1]}',
                       xy=(symbol.real, symbol.imag),
                       xytext=(symbol.real*1.3, symbol.imag*1.3),
                       fontsize=14, fontweight='bold',
                       ha='center', va='center')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('In-Phase (I)', fontsize=12)
        ax.set_ylabel('Quadrature (Q)', fontsize=12)
        ax.set_title('QPSK Constellation\n(Gray Coded)', fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        return ax

class SimplifiedLDPC:
    """
    Simplified Low-Density Parity-Check (LDPC) code encoder/decoder.
    
    This implements a systematic LDPC code where codewords have the form:
    c = [u | p] where u is k information bits and p is m parity bits.
    
    The parity check matrix has systematic form: H = [P | I]
    where P is m×k parity part and I is m×m identity matrix.
    Parity bits are computed as: p = P · u (mod 2)
    
    Note: This is a simplified implementation for simulation purposes.
    Real LDPC codes use more sophisticated matrix construction methods
    and advanced decoding algorithms (e.g., belief propagation).
    
    References:
    - Gallager, "Low-Density Parity-Check Codes" (1962)
    - Richardson & Urbanke, "Modern Coding Theory" (2008)
    """
    def __init__(self, n=1024, rate=0.8):
        """
        Parameters:
        -----------
        n : int
            Codeword length (total bits per block)
        rate : float
            Code rate R = k/n (information bits / codeword length)
        """
        if n <= 0 or rate <= 0 or rate >= 1:
            raise ValueError(f"Invalid LDPC parameters: n={n}, rate={rate}")
        
        self.n = n
        self.k = int(n * rate)
        self.m = n - self.k
        self.rate = rate
        
        # Generate parity check matrix H = [P | I] in systematic form
        self.H, self.H_info_T, self.H_parity_inv = self._generate_systematic_H()

    def _generate_systematic_H(self):
        """
        Generates a systematic parity check matrix H = [P | I]
        where P is the parity part (m×k) and I is identity (m×m).
        
        This uses a simple random construction with column weight 3.
        For real systems, use structured construction (Gallager, PEG, etc.)
        """
        col_weight = 3
        
        # Ensure column weight doesn't exceed number of check nodes
        if col_weight > self.m:
            col_weight = self.m
        
        # Generate parity part P (m×k matrix)
        # Each column has exactly 'col_weight' ones randomly placed
        P = np.zeros((self.m, self.k), dtype=int)
        for col in range(self.k):
            check_nodes = np.random.choice(self.m, col_weight, replace=False)
            P[check_nodes, col] = 1
        
        # Systematic form: H = [P | I] where I is m×m identity
        H_parity = np.eye(self.m, dtype=int)
        H = np.concatenate([P, H_parity], axis=1).astype(int)
        
        return H, P.T, None 

    def encode(self, info_bits):
        """
        Encodes information bits using systematic LDPC encoding.
        
        For systematic encoding: c = [u | p] where
        - u: k information bits
        - p: m parity bits computed as p = P · u (mod 2)
        
        Input bits are padded to multiple of k and encoded block-by-block.
        
        Parameters:
        -----------
        info_bits : array_like
            Information bits to encode (can be any length)
        
        Returns:
        --------
        coded_bits : ndarray
            Encoded codeword bits (length = ceil(len(info_bits)/k) * n)
        """
        if len(info_bits) == 0:
            return np.array([], dtype=int)
        
        # Pad input to multiple of k bits
        num_blocks = int(np.ceil(len(info_bits) / self.k))
        padded_info_len = num_blocks * self.k
        pad_len = padded_info_len - len(info_bits)
        
        if pad_len > 0:
            padded_info_bits = np.concatenate([info_bits, np.zeros(pad_len, dtype=int)])
        else:
            padded_info_bits = info_bits

        # Encode each block
        coded_bits = []
        for i in range(num_blocks):
            block = padded_info_bits[i*self.k:(i+1)*self.k]
            parity = self._compute_parity(block)
            # Systematic codeword: [information bits | parity bits]
            codeword = np.concatenate([block, parity])
            coded_bits.append(codeword)
            
        return np.concatenate(coded_bits).astype(int)

    def _compute_parity(self, info_bits):
        """
        Computes parity bits for systematic LDPC encoding.
        
        For H = [P | I], the parity bits are: p = P · u (mod 2)
        where P is the parity part (first k columns of H).
        
        Parameters:
        -----------
        info_bits : array_like
            k information bits
        
        Returns:
        --------
        parity : ndarray
            m parity bits
        """
        # Extract parity part P from H = [P | I]
        P = self.H[:, :self.k]
        # Compute parity: p = P · u (mod 2)
        parity = np.dot(P, info_bits) % 2
        return parity

    def decode_simple(self, received_bits):
        num_blocks = len(received_bits) // self.n
        decoded = []
        for i in range(num_blocks):
            block = received_bits[i*self.n:(i+1)*self.n]
            info_bits = block[:self.k]
            syndrome = np.dot(self.H, block) % 2
            if np.sum(syndrome) == 0:
                decoded.append(info_bits)
            else:
                corrected = self._bit_flipping(block, max_iter=10)
                decoded.append(corrected[:self.k])
        return np.concatenate(decoded).astype(int)

    def _bit_flipping(self, received, max_iter):
        r = received.copy()
        for iteration in range(max_iter):
            syndrome = np.dot(self.H, r) % 2
            if np.sum(syndrome) == 0:
                break
            unsatisfied_counts = np.dot(self.H.T, syndrome)
            flip_idx = np.argmax(unsatisfied_counts)
            r[flip_idx] ^= 1
        return r

class PilotHandler:
    """
    Pilot symbol insertion and extraction for channel estimation.
    
    Pilots are known symbols inserted periodically in the data stream
    to enable channel estimation at the receiver. The pilot ratio
    defines the fraction of total symbols that are pilots.
    
    References:
    - Proakis & Salehi (2008), Ch. 10 (Channel Estimation)
    - Goldsmith, "Wireless Communications" (2005), Ch. 13
    """
    def __init__(self, pilot_ratio=0.1, pattern='uniform'):
        """
        Parameters:
        -----------
        pilot_ratio : float
            Ratio of pilots to total symbols: n_pilots / (n_pilots + n_data)
            Must be in range (0, 1)
        pattern : str
            Pilot insertion pattern: 'uniform', 'block', or 'random'
        """
        if pilot_ratio <= 0 or pilot_ratio >= 1:
            raise ValueError(f"pilot_ratio must be in (0, 1), got {pilot_ratio}")
        
        self.pilot_ratio = pilot_ratio
        self.pattern = pattern
        self.pilot_sequence = None
        self.pilot_positions = None
        
        # QPSK constellation for pilot symbols
        # Note: This matches the QPSKModulator constellation
        np.random.seed(123)
        self.qpsk_constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)


    def insert_pilots(self, data_symbols):
        n_data = len(data_symbols)
        n_pilots = int(np.ceil(n_data * self.pilot_ratio / (1 - self.pilot_ratio)))
        n_total = n_data + n_pilots
        
        pilot_indices = np.random.randint(0, 4, n_pilots)
        self.pilot_sequence = self.qpsk_constellation[pilot_indices]

        if self.pattern == 'uniform':
            # Ensure unique positions
            indices = np.linspace(0, n_total - 1, n_pilots)
            self.pilot_positions = np.unique(indices.astype(int))
            # Handle potential duplicates from rounding
            n_pilots = len(self.pilot_positions)
            self.pilot_sequence = self.pilot_sequence[:n_pilots]
            n_total = n_data + n_pilots # Recalculate total
            
        elif self.pattern == 'block':
            self.pilot_positions = np.arange(n_pilots)
        else: # Random
            self.pilot_positions = np.sort(np.random.choice(n_total, n_pilots, replace=False))

        frame = np.zeros(n_total, dtype=complex)
        data_idx = 0
        pilot_idx = 0
        
        # More robust insertion loop
        pilot_set = set(self.pilot_positions)
        for i in range(n_total):
            if i in pilot_set:
                if pilot_idx < n_pilots: 
                    frame[i] = self.pilot_sequence[pilot_idx]
                    pilot_idx += 1
            else:
                if data_idx < n_data: 
                    frame[i] = data_symbols[data_idx]
                    data_idx += 1
                # Handle case where n_data was less than available slots
                # (e.g., due to pilot position rounding)
                # In this case, frame[i] remains 0 (padding)
        
        # Trim any excess if data ran out
        if data_idx < n_data:
            print(f"Warning: Not all data symbols were inserted. {n_data - data_idx} remaining.")
        
        return frame, self.pilot_positions

    def extract_pilots(self, received_frame):
        rx_pilots = received_frame[self.pilot_positions]
        data_mask = np.ones(len(received_frame), dtype=bool)
        data_mask[self.pilot_positions] = False
        data_symbols = received_frame[data_mask]
        return data_symbols, rx_pilots

    def estimate_channel(self, rx_pilots, method='LS'):
        """
        Estimates channel gain using received pilot symbols.
        
        Uses Least Squares (LS) estimation: h_est = mean(rx_pilots / tx_pilots)
        
        Parameters:
        -----------
        rx_pilots : array_like
            Received pilot symbols (complex)
        method : str
            Estimation method ('LS' for Least Squares, currently only LS supported)
        
        Returns:
        --------
        h_est : complex
            Estimated channel gain
        """
        if len(rx_pilots) == 0:
            return 1.0  # No pilots, assume ideal channel
        
        if self.pilot_sequence is None or len(self.pilot_sequence) == 0:
            return 1.0  # No pilot sequence available
        
        # Get transmitted pilots (trim to match received length)
        tx_pilots = self.pilot_sequence[:len(rx_pilots)]
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.divide(rx_pilots, tx_pilots, 
                             out=np.zeros_like(rx_pilots, dtype=complex),
                             where=tx_pilots != 0)
        
        # Least Squares estimation: average of ratios
        # For AWGN channel: h_est = E[rx_pilots / tx_pilots]
        h_est = np.mean(ratios[tx_pilots != 0]) if np.any(tx_pilots != 0) else 1.0
        
        return h_est

class encodingRunner:
    def __init__(self,
                 spatial_modes=None, # <-- RECTIFIED: Renamed from oam_modes
                 wavelength=1550e-9,
                 w0=25e-3,
                 fec_rate=0.8,
                 pilot_ratio=0.1):
        
        if spatial_modes is None:
            self.spatial_modes = [(0, -1), (0, 1)] # <-- RECTIFIED: Default is now (p,l) tuples
        else:
            self.spatial_modes = spatial_modes
            
        self.n_modes = len(self.spatial_modes)
        self.wavelength = wavelength
        self.w0 = w0
        self.qpsk = QPSKModulator(symbol_energy=1.0)
        self.ldpc = SimplifiedLDPC(n=1024, rate=fec_rate)
        self.pilot_handler = PilotHandler(pilot_ratio=pilot_ratio, pattern='uniform')
        
        self.lg_beams = {}
        for p, l in self.spatial_modes: # <-- RECTIFIED: Iterate over (p, l) tuples
            # Use (p, l) tuple as the dictionary key
            self.lg_beams[(p, l)] = LaguerreGaussianBeam(p=p, l=l, wavelength=wavelength, w0=w0) # <-- RECTIFIED: Pass p
            
        print(f"Spatial Modes (p, l): {self.spatial_modes}") # <-- RECTIFIED
        print(f"Number of Modes: {self.n_modes}")
        print(f"Wavelength: {wavelength*1e9:.0f} [nm]")
        print(f"Beam Waist w0: {w0*1e3:.2f} [mm]")
        print(f"LDPC Code Rate: {fec_rate}")
        print(f"Pilot Ratio: {pilot_ratio:.1%}")

    def transmit(self, data_bits, verbose=True):
        if verbose: print(f"\nInput: {len(data_bits)} [info bits]")
        
        # CORRECTED LOGGING
        encoded_bits = self.ldpc.encode(data_bits)
        
        # Calculate the ACTUAL code rate after padding
        padded_info_len = int(np.ceil(len(data_bits) / self.ldpc.k)) * self.ldpc.k
        actual_rate = len(data_bits) / len(encoded_bits) if len(encoded_bits) > 0 else 0
        
        if verbose: 
            print(f"After LDPC (target rate {self.ldpc.rate:.2f}, actual {actual_rate:.2f}): {len(encoded_bits)} [coded bits]")
            if padded_info_len != len(data_bits):
                print(f"  (Input was padded from {len(data_bits)} to {padded_info_len} info bits to fit block structure)")

        qpsk_symbols = self.qpsk.modulate(encoded_bits)
        if verbose: print(f"After QPSK: {len(qpsk_symbols)} [symbols]")
        
        # Check if there are enough symbols for pilots
        if len(qpsk_symbols) == 0:
            print("Warning: No QPSK symbols generated. Aborting transmit.")
            return {}
            
        frame_with_pilots, pilot_pos = self.pilot_handler.insert_pilots(qpsk_symbols)
        if verbose: print(f"After Pilot Insertion: {len(frame_with_pilots)} [symbols] ({len(pilot_pos)} pilots)")
        
        symbols_per_mode = len(frame_with_pilots) // self.n_modes
        remainder = len(frame_with_pilots) % self.n_modes
        tx_signals = {}
        start_idx = 0
        
        for idx, (p, l) in enumerate(self.spatial_modes): # <-- RECTIFIED: Iterate over (p, l)
            end_idx = start_idx + symbols_per_mode + (1 if idx < remainder else 0)
            mode_symbols = frame_with_pilots[start_idx:end_idx]
            
            mode_key = (p, l) # <-- RECTIFIED: Use tuple as key
            
            tx_signals[mode_key] = {
                'symbols': mode_symbols,
                'frame': mode_symbols, 
                'beam': self.lg_beams[mode_key],
                'n_symbols': len(mode_symbols)
            }
            if verbose: print(f"  Mode (p={p}, l={l:+2d}): {len(mode_symbols)} symbols") # <-- RECTIFIED
            start_idx = end_idx
        return tx_signals

    def generate_spatial_field(self, tx_signals, z=500, grid_size=256, extent_mm=None):
        if extent_mm is None:
            max_beam_radius = 0
            for mode_key, sig_data in tx_signals.items(): # <-- RECTIFIED
                beam = sig_data['beam']
                w_z = beam.beam_waist(z)
                max_beam_radius = max(max_beam_radius, w_z)
            extent_mm = 3 * max_beam_radius * 1e3
        
        extent = extent_mm * 1e-3
        x = np.linspace(-extent, extent, grid_size)
        y = np.linspace(-extent, extent, grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        PHI = np.arctan2(Y, X)

        total_field = np.zeros((grid_size, grid_size), dtype=complex) 
        for mode_key, sig_data in tx_signals.items(): # <-- RECTIFIED
            if sig_data['n_symbols'] > 0: 
                beam = sig_data['beam']
                symbol = sig_data['symbols'][0] # Still just takes the first symbol for a snapshot
                field = beam.generate_beam_field(R, PHI, z)
                modulated_field = field * symbol
                total_field += modulated_field 
        
        total_intensity = np.abs(total_field)**2

        grid_info = {'x': x, 'y': y, 'X': X, 'Y': Y, 'extent_mm': extent_mm, 'grid_size': grid_size}
        return total_intensity, grid_info

    def plot_system_summary(self, data_bits, tx_signals):
        n_modes = len(self.spatial_modes) # <-- RECTIFIED
        if n_modes == 0:
            print("No modes to plot.")
            return None
            
        modes_per_row = min(4, n_modes)
        n_mode_rows = int(np.ceil(n_modes / modes_per_row))
        total_rows = 1 + n_mode_rows
        fig = plt.figure(figsize=(16, 5 * total_rows))
        fig.suptitle("FSO-MDM Transmitter System Summary", fontsize=16, fontweight='bold') # <-- RECTIFIED: MDM

        ax1 = plt.subplot(total_rows, 3, 1)
        self.qpsk.plot_constellation(ax=ax1)
        ax2 = plt.subplot(total_rows, 3, 2)
        
        first_mode = self.spatial_modes[0] # <-- RECTIFIED
        if tx_signals[first_mode]['n_symbols'] > 0:
            symbols = tx_signals[first_mode]['symbols'][:100]
            ax2.plot(symbols.real, symbols.imag, 'b.', alpha=0.6, markersize=8)
            ax2.plot(self.qpsk.constellation_points.real,
                    self.qpsk.constellation_points.imag,
                    'ro', markersize=12, label='Ideal')
            ax2.grid(True, alpha=0.3); ax2.set_xlabel('In-Phase'); ax2.set_ylabel('Quadrature')
            ax2.set_title(f'Tx Symbols - Mode (p={first_mode[0]}, l={first_mode[1]})'); ax2.legend(); ax2.axis('equal') # <-- RECTIFIED
            ax3 = plt.subplot(total_rows, 3, 3)
            ax3.plot(np.abs(symbols), 'b-', linewidth=1.5, label='|s(n)|')
            ax3.plot(np.angle(symbols), 'r-', linewidth=1.5, alpha=0.7, label='∠s(n)')
            ax3.grid(True, alpha=0.3); ax3.set_xlabel('Symbol Index'); ax3.set_ylabel('Amplitude / Phase')
            ax3.set_title('Symbol Sequence'); ax3.legend()

        for idx, (p, l) in enumerate(self.spatial_modes): # <-- RECTIFIED
            mode_key = (p, l)
            mode_row = idx // modes_per_row
            mode_col = idx % modes_per_row
            ax = plt.subplot2grid((total_rows, modes_per_row), (1 + mode_row, mode_col), fig=fig)
            
            beam = self.lg_beams[mode_key]
            extent_mm = 3 * self.w0 * 1e3
            grid_size = 200
            extent = extent_mm * 1e-3
            x = np.linspace(-extent, extent, grid_size)
            y = np.linspace(-extent, extent, grid_size)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            PHI = np.arctan2(Y, X)
            
            intensity = beam.calculate_intensity(R, PHI, 0)
            im = ax.imshow(intensity, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm], cmap='hot', origin='lower')
            ax.set_xlabel('x [mm]', fontsize=8); ax.set_ylabel('y [mm]', fontsize=8)
            # <-- RECTIFIED: Correctly uses beam.p and beam.l
            ax.set_title(f'LG$_{{{beam.p}}}^{{{beam.l}}}$ M²={beam.M_squared:.1f}', fontweight='bold', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, label='I [a.u.]') 

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

if __name__ == "__main__":
    # --- Define Simulation Parameters ---
    WAVELENGTH = 1550e-9  
    W0 = 25e-3  
    Z_PROPAGATION = 1000
    
    # <-- RECTIFIED: Changed from OAM_MODES to SPATIAL_MODES (p, l)
    # You can now add complex modes like (1, 1)
    SPATIAL_MODES = [(0, -4), (0, -2), (0, 2), (0, 4), (1, 1), (1, -1)] 
    
    FEC_RATE = 0.8  
    PILOT_RATIO = 0.1  
    QPSK_ENERGY = 1.0  
    N_INFO_BITS = 4096  
    GRID_SIZE = 512
    DPI = 300 # <-- Reduced DPI for faster testing
    
    PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
    
    # <-- RECTIFIED: FIG_NAME is no longer used globally
    # FIG_NAME = f"lg_p{0}_l{'_'.join(map(str, OAM_MODES))}_beam.png" 

    print(f"Link distance:      {Z_PROPAGATION}m ({Z_PROPAGATION/1000:.1f}km)")
    print(f"Wavelength:         {WAVELENGTH*1e9:.0f}nm")
    print(f"Beam waist:         {W0*1e3:.1f}mm")
    print(f"Spatial modes (p,l):{SPATIAL_MODES}") # <-- RECTIFIED
    print(f"FEC rate:           {FEC_RATE}")
    print(f"Info bits:          {N_INFO_BITS}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Plot directory ensured at: {PLOT_DIR}\n") 
    
    system = encodingRunner(
        spatial_modes=SPATIAL_MODES, # <-- RECTIFIED
        wavelength=WAVELENGTH,
        w0=W0,
        fec_rate=FEC_RATE,
        pilot_ratio=PILOT_RATIO
    )
    
    data_bits = np.random.randint(0, 2, N_INFO_BITS)
    print(f"Generated {N_INFO_BITS} random data bits")

    tx_signals = system.transmit(data_bits, verbose=True)

    print("\nGenerating system summary plot...")
    fig = system.plot_system_summary(data_bits, tx_signals)
    if fig:
        summary_path = os.path.join(PLOT_DIR, 'encoding_summary_generalized.png') # <-- RECTIFIED
        fig.savefig(summary_path, dpi=DPI, bbox_inches='tight')

    print(f"Generating multiplexed field at z={Z_PROPAGATION}m...")
    total_field, grid_info = system.generate_spatial_field(
        tx_signals, 
        z=Z_PROPAGATION,
        grid_size=GRID_SIZE
    )

    fig2, (ax_trans, ax_long) = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle("MDM Multiplexed Beam Propagation", fontsize=16, fontweight='bold') # <-- RECTIFIED
    
    extent_mm = grid_info['extent_mm']
    im1 = ax_trans.imshow(total_field, extent=[-extent_mm, extent_mm, -extent_mm, extent_mm],
                  cmap='hot', origin='lower', interpolation='bilinear')
    ax_trans.set_xlabel('x [mm]', fontsize=12); ax_trans.set_ylabel('y [mm]', fontsize=12)
    ax_trans.set_title(f'Transverse Field at z={Z_PROPAGATION/1000:.1f}km\n(Coherent Sum Snapshot)',
                fontsize=14, fontweight='bold')
    ax_trans.set_aspect('equal')
    plt.colorbar(im1, ax=ax_trans, label='Intensity [a.u.]')
    
    print("Calculating longitudinal propagation (this may take a moment)...")
    z_max = Z_PROPAGATION * 1.5
    num_z = 150; num_r = 200
    z_array = np.linspace(1e-6, z_max, num_z)
    
    max_beam_radius = 0
    for mode_key in system.spatial_modes: # <-- RECTIFIED
        beam = system.lg_beams[mode_key]
        w_z = beam.beam_waist(z_max)
        max_beam_radius = max(max_beam_radius, w_z)
    
    r_max = 3 * max_beam_radius
    r_array = np.linspace(-r_max, r_max, num_r)
    
    intensity_long = np.zeros((num_r, num_z))
    
    for i, z in enumerate(z_array):
        total_field_slice = np.zeros(num_r, dtype=complex) # Must be complex
        for mode_key, sig_data in tx_signals.items(): # <-- RECTIFIED
            if sig_data['n_symbols'] > 0:
                beam = sig_data['beam']
                symbol = sig_data['symbols'][0]
                field_slice = beam.generate_beam_field(np.abs(r_array), 0, z)
                total_field_slice += field_slice * symbol
        
        intensity_long[:, i] = np.abs(total_field_slice)**2
    z_array_km = z_array / 1000
    r_max_mm = r_max * 1e3
    
    vmax = np.percentile(intensity_long, 99.8) 
    im2 = ax_long.imshow(intensity_long, extent=[0, z_array_km[-1], -r_max_mm, r_max_mm],
                       aspect='auto', cmap='hot', origin='lower', interpolation='bilinear',
                       vmax=vmax)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(system.spatial_modes))) # <-- RECTIFIED
    for idx, mode_key in enumerate(system.spatial_modes): # <-- RECTIFIED
        beam = system.lg_beams[mode_key]
        w_z_array = np.array([beam.beam_waist(z)*1e3 for z in z_array])
        ax_long.plot(z_array_km, w_z_array, '--', linewidth=1.5, 
                    color=colors[idx], alpha=0.8, label=f'({beam.p},{beam.l}) w(z)') # <-- RECTIFIED
        ax_long.plot(z_array_km, -w_z_array, '--', linewidth=1.5, 
                    color=colors[idx], alpha=0.8)
    
    ax_long.axvline(Z_PROPAGATION/1000, color='lime', linestyle=':', linewidth=2.5,
                   label=f'z={Z_PROPAGATION/1000:.1f}km', alpha=0.8)
    
    ax_long.set_xlabel('Propagation Distance z [km]', fontsize=12)
    ax_long.set_ylabel('Radial Position r [mm]', fontsize=12)
    ax_long.set_title(f'Longitudinal Propagation (Coherent Sum)\nλ={WAVELENGTH*1e9:.0f}nm, w₀={W0*1e3:.1f}mm',
                     fontsize=14, fontweight='bold')
    ax_long.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
    ax_long.grid(True, alpha=0.3)
    ax_long.set_ylim(-r_max_mm, r_max_mm) 
    plt.colorbar(im2, ax=ax_long, label='Intensity [a.u.]', fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    prop_path = os.path.join(PLOT_DIR, 'mdm_multiplexed_field.png') # <-- RECTIFIED
    fig2.savefig(prop_path, dpi=DPI, bbox_inches='tight')


    print("\nTx Summary") # <-- RECTIFIED

    print(f"Information bits:      {N_INFO_BITS} [bits]")
    print(f"Code rate:             {system.ldpc.rate}")
    # Re-encode to get an accurate count, as padding might occur
    coded_bits_len = len(system.ldpc.encode(data_bits)) 
    print(f"Coded bits:            {coded_bits_len} [bits]")

    total_symbols = sum([sig['n_symbols'] for sig in tx_signals.values()])
    n_pilots = len(system.pilot_handler.pilot_positions)
    print(f"Total symbols:         {total_symbols} [symbols] (incl. {n_pilots} pilots)")
    
    # <-- RECTIFIED: More descriptive print
    print("Symbols per mode:")
    for mode_key, sig in tx_signals.items():
        print(f"  Mode (p={mode_key[0]}, l={mode_key[1]}): {sig['n_symbols']} symbols")

    # This calculation remains valid, but it's the *total system* efficiency
    # not per-mode efficiency, as it's info_bits / total_symbols_all_modes
    total_system_efficiency = N_INFO_BITS / total_symbols 
    print(f"Total System Spectral Efficiency: {total_system_efficiency:.3f} [bits/symbol]")

    print(f"\n{'BEAM PARAMETERS AT z=' + str(Z_PROPAGATION) + 'm:'}")
    for mode_key in system.spatial_modes: # <-- RECTIFIED
        beam = system.lg_beams[mode_key]
        theta_0, theta_eff = beam.effective_divergence_angle # <-- Access as property
        w_z = beam.beam_waist(Z_PROPAGATION)
        z_R = beam.z_R
        print(f"  Mode (p={beam.p}, l={beam.l:+2d}): M²={beam.M_squared:.1f}, "
              f"z_R={z_R:.1f}m, "
              f"w(z)={w_z*1e3:.1f}mm, "
              f"θ_eff={theta_eff*1e6:.1f}μrad")
    plt.show()