"""
Utility functions for CNN-based OAM receiver.

Includes:
- Symbol mapping and demapping
- LLR computation for soft LDPC decoding
- Performance metrics (SER, BER)
- Visualization helpers
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# QPSK constellation (Gray coded)
QPSK_CONSTELLATION = np.array([
    (1 + 1j) / np.sqrt(2),   # 00
    (-1 + 1j) / np.sqrt(2),  # 01
    (1 - 1j) / np.sqrt(2),   # 10
    (-1 - 1j) / np.sqrt(2),  # 11
], dtype=complex)

QPSK_BITS = np.array([
    [0, 0],  # (1+1j)/√2
    [0, 1],  # (-1+1j)/√2
    [1, 0],  # (1-1j)/√2
    [1, 1],  # (-1-1j)/√2
])


def map_to_qpsk(symbols_est):
    """
    Map estimated symbols to nearest QPSK constellation point.
    
    Args:
        symbols_est: Estimated symbols (complex array)
    
    Returns:
        symbols_hard: Hard-decided QPSK symbols
    """
    symbols_est = np.asarray(symbols_est)
    
    # Compute distance to each constellation point
    distances = np.abs(symbols_est[..., np.newaxis] - QPSK_CONSTELLATION)
    
    # Find nearest point
    indices = np.argmin(distances, axis=-1)
    
    # Map to constellation
    symbols_hard = QPSK_CONSTELLATION[indices]
    
    return symbols_hard


def qpsk_demodulate(symbols, soft=False, noise_var=1.0):
    """
    Demodulate QPSK symbols to bits.
    
    Args:
        symbols: QPSK symbols (complex array)
        soft: If True, return LLRs; if False, return hard bits
        noise_var: Noise variance for LLR calculation
    
    Returns:
        bits: Hard bits (if soft=False) or LLRs (if soft=True)
    """
    symbols = np.asarray(symbols)
    
    if soft:
        # Compute LLRs
        llrs = compute_llrs(symbols, noise_var)
        return llrs
    else:
        # Hard decision
        bits = np.zeros((*symbols.shape, 2), dtype=int)
        bits[..., 0] = (np.real(symbols) < 0).astype(int)  # I bit
        bits[..., 1] = (np.imag(symbols) < 0).astype(int)  # Q bit
        return bits


def compute_llrs(symbols_est, noise_var=1.0):
    """
    Compute log-likelihood ratios (LLRs) for soft LDPC decoding.
    
    LLR(b) = log(P(b=0|y) / P(b=1|y))
    
    Args:
        symbols_est: Estimated symbols (complex array)
        noise_var: Noise variance
    
    Returns:
        llrs: LLRs for each bit (shape: *symbols.shape, 2)
    """
    symbols_est = np.asarray(symbols_est)
    
    # Distance to each constellation point
    distances = np.abs(symbols_est[..., np.newaxis] - QPSK_CONSTELLATION) ** 2
    
    # Log probabilities (up to constant)
    log_probs = -distances / (2 * noise_var)
    
    # LLR for I bit (bit 0)
    llr_I = np.logaddexp(log_probs[..., 0], log_probs[..., 1]) - \
            np.logaddexp(log_probs[..., 2], log_probs[..., 3])
    
    # LLR for Q bit (bit 1)
    llr_Q = np.logaddexp(log_probs[..., 0], log_probs[..., 2]) - \
            np.logaddexp(log_probs[..., 1], log_probs[..., 3])
    
    # Stack LLRs
    llrs = np.stack([llr_I, llr_Q], axis=-1)
    
    return llrs


def compute_ser(symbols_true, symbols_est):
    """
    Compute Symbol Error Rate (SER).
    
    Args:
        symbols_true: True symbols
        symbols_est: Estimated symbols
    
    Returns:
        ser: Symbol error rate
    """
    symbols_true = np.asarray(symbols_true)
    symbols_est = np.asarray(symbols_est)
    
    # Map estimates to nearest QPSK point
    symbols_hard = map_to_qpsk(symbols_est)
    
    # Count errors
    errors = np.sum(symbols_hard != symbols_true)
    total = symbols_true.size
    
    ser = errors / total if total > 0 else 0.0
    
    return ser


def compute_ber(bits_true, bits_est):
    """
    Compute Bit Error Rate (BER).
    
    Args:
        bits_true: True bits
        bits_est: Estimated bits
    
    Returns:
        ber: Bit error rate
    """
    bits_true = np.asarray(bits_true).flatten()
    bits_est = np.asarray(bits_est).flatten()
    
    errors = np.sum(bits_true != bits_est)
    total = bits_true.size
    
    ber = errors / total if total > 0 else 0.0
    
    return ber


def symbols_to_tensor(symbols, device='cpu'):
    """
    Convert complex symbols to PyTorch tensor (real representation).
    
    Args:
        symbols: Complex symbols (shape: ..., num_modes)
        device: PyTorch device
    
    Returns:
        tensor: Real tensor (shape: ..., num_modes * 2)
    """
    symbols = np.asarray(symbols)
    
    # Stack real and imaginary parts
    real_repr = np.stack([np.real(symbols), np.imag(symbols)], axis=-1)
    
    # Flatten last two dimensions
    real_repr = real_repr.reshape(*symbols.shape[:-1], -1)
    
    # Convert to tensor
    tensor = torch.from_numpy(real_repr).float().to(device)
    
    return tensor


def tensor_to_symbols(tensor):
    """
    Convert PyTorch tensor (real representation) to complex symbols.
    
    Args:
        tensor: Real tensor (shape: ..., num_modes * 2)
    
    Returns:
        symbols: Complex symbols (shape: ..., num_modes)
    """
    tensor = tensor.detach().cpu().numpy()
    
    # Reshape to (..., num_modes, 2)
    num_modes = tensor.shape[-1] // 2
    tensor = tensor.reshape(*tensor.shape[:-1], num_modes, 2)
    
    # Convert to complex
    symbols = tensor[..., 0] + 1j * tensor[..., 1]
    
    return symbols


def plot_constellation(symbols_true, symbols_est, title="Constellation Diagram", 
                       save_path=None):
    """
    Plot constellation diagram comparing true and estimated symbols.
    
    Args:
        symbols_true: True symbols
        symbols_est: Estimated symbols
        title: Plot title
        save_path: Path to save figure (optional)
    """
    symbols_true = np.asarray(symbols_true).flatten()
    symbols_est = np.asarray(symbols_est).flatten()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot QPSK constellation
    ax.scatter(np.real(QPSK_CONSTELLATION), np.imag(QPSK_CONSTELLATION),
               s=200, c='red', marker='x', linewidths=3, 
               label='QPSK Constellation', zorder=3)
    
    # Plot true symbols
    ax.scatter(np.real(symbols_true), np.imag(symbols_true),
               s=50, c='blue', alpha=0.3, label='True Symbols')
    
    # Plot estimated symbols
    ax.scatter(np.real(symbols_est), np.imag(symbols_est),
               s=30, c='green', alpha=0.5, marker='^', 
               label='Estimated Symbols')
    
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('In-Phase (I)', fontsize=12)
    ax.set_ylabel('Quadrature (Q)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def normalize_intensity(intensity, method='per_sample'):
    """
    Normalize intensity image.
    
    Args:
        intensity: Intensity array
        method: 'per_sample', 'global', or 'none'
    
    Returns:
        normalized: Normalized intensity
    """
    if method == 'none':
        return intensity
    
    elif method == 'per_sample':
        # Normalize each sample to [0, 1]
        min_val = np.min(intensity, axis=(-2, -1), keepdims=True)
        max_val = np.max(intensity, axis=(-2, -1), keepdims=True)
        normalized = (intensity - min_val) / (max_val - min_val + 1e-10)
        return normalized
    
    elif method == 'global':
        # Normalize using global statistics
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        normalized = (intensity - min_val) / (max_val - min_val + 1e-10)
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test QPSK mapping
    symbols_est = np.array([0.8+0.9j, -0.7+0.6j, 0.5-0.8j, -0.9-0.7j])
    symbols_hard = map_to_qpsk(symbols_est)
    print(f"Estimated: {symbols_est}")
    print(f"Hard decision: {symbols_hard}")
    
    # Test demodulation
    bits_hard = qpsk_demodulate(symbols_hard, soft=False)
    print(f"Hard bits: {bits_hard}")
    
    llrs = qpsk_demodulate(symbols_est, soft=True, noise_var=0.1)
    print(f"LLRs shape: {llrs.shape}")
    
    # Test SER/BER
    symbols_true = QPSK_CONSTELLATION[[0, 1, 2, 3]]
    ser = compute_ser(symbols_true, symbols_est)
    print(f"SER: {ser:.2%}")
    
    # Test tensor conversion
    tensor = symbols_to_tensor(symbols_true)
    symbols_back = tensor_to_symbols(tensor)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Symbols recovered: {np.allclose(symbols_true, symbols_back)}")
    
    print("\n✓ All tests passed!")
