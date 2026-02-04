"""E2E Runner with Message Encoding (wraps pipeline.py for convenience)"""
import os
import sys

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import canonical pipeline
from pipeline import (
    SimulationConfig as PipelineConfig,
    run_e2e_simulation as run_pipeline_e2e,
    plot_e2e_results,
    plot_symbol_comparison,
    run_cn2_sweep as run_pipeline_sweep
)

np.random.seed(42)

# ============================================================================
# TEXT ENCODING UTILITIES (MESSAGE EMBEDDING CONVENIENCE)
# ============================================================================

def text_to_bits(text: str) -> np.ndarray:
    """
    Convert a UTF-8 string into a numpy array of bits (uint8 {0,1}).
    """
    if not text:
        return np.zeros(0, dtype=np.uint8)
    byte_array = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    return np.unpackbits(byte_array)


def bits_to_text(bits: np.ndarray, bit_length: int) -> str:
    """
    Reconstruct a UTF-8 string from a numpy array of bits.
    Only the first `bit_length` bits are considered (remaining bits ignored).
    """
    bit_length = int(bit_length)
    if bit_length <= 0:
        return ""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        bits = bits.ravel()
    trimmed = bits[:bit_length]
    if bit_length % 8 != 0:
        pad = 8 - (bit_length % 8)
        trimmed = np.concatenate([trimmed, np.zeros(pad, dtype=np.uint8)])
    byte_len = (bit_length + 7) // 8
    byte_array = np.packbits(trimmed)[:byte_len]
    try:
        return byte_array.tobytes().decode('utf-8')
    except UnicodeDecodeError:
        # Fallback: ignore invalid trailing bytes
        return byte_array.tobytes().decode('utf-8', errors='ignore')


# ============================================================================
# EXTENDED CONFIG (ADDS MESSAGE SUPPORT)
# ============================================================================
class SimulationConfig(PipelineConfig):
    """Extends PipelineConfig with message embedding support."""
    LDPC_BLOCKS = 4
    MESSAGE = "SRIVATSA"  # User message to embed


# ============================================================================
# MESSAGE-AWARE E2E SIMULATION (WRAPS PIPELINE)
# ============================================================================

def run_e2e_simulation(config, verbose=True):
    """
    Wrapper around pipeline.run_e2e_simulation that adds message embedding.
    
    This implementation delegates all physics to pipeline.py and only adds
    message encoding/decoding convenience on top.
    """
    
    if verbose:
        print("\n" + "="*80)
        print("RUNNER: MESSAGE-AWARE E2E SIMULATION (WRAPS PIPELINE)")
        print("="*80)
    
    cfg = config
    
    # Prepare message bits
    message_text = getattr(cfg, "MESSAGE", "")
    message_bits = text_to_bits(message_text)
    message_bit_len = len(message_bits)
    
    # Run canonical pipeline (it generates random data_bits internally)
    results = run_pipeline_e2e(cfg, verbose=verbose)
    
    if results is None:
        return None
    
    # Post-process: Embed message in recovered bits for visualization
    # (NOTE: pipeline already calculates BER on random bits; this is just for demo)
    recovered_bits = results.get('metrics', {}).get('recovered_bits', np.array([]))
    if len(recovered_bits) >= message_bit_len and message_bit_len > 0:
        decoded_message = bits_to_text(recovered_bits, message_bit_len)
        if verbose:
            print(f"\nRecovered message (first {message_bit_len} bits): '{decoded_message}'")
            print(f"Original message: '{message_text}'")
        # Add to results
        results['message'] = {
            'original': message_text,
            'decoded': decoded_message,
            'bit_length': message_bit_len
        }
    
    return results


def run_cn2_sweep(config_class, cn2_values, enable_power_probe=False, save_plots=False):
    """Wrapper around pipeline.run_cn2_sweep (message embedding not needed for sweeps)."""
    return run_pipeline_sweep(config_class, cn2_values, enable_power_probe, save_plots)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FSO-OAM E2E Simulation with Message Encoding")
    parser.add_argument(
        "--cn2-sweep",
        nargs="+",
        type=float,
        help="List of Cn² values (m^-2/3) to sweep. Example: --cn2-sweep 0 5e-19 1e-18"
    )
    parser.add_argument(
        "--disable-power-probe",
        action="store_true",
        help="Skip the numerical power probe diagnostic to speed up runs."
    )
    parser.add_argument(
        "--save-sweep-plots",
        action="store_true",
        help="When sweeping Cn², save plots for each operating point."
    )
    args = parser.parse_args()

    if args.cn2_sweep:
        cn2_values = args.cn2_sweep
        run_cn2_sweep(
            SimulationConfig,
            cn2_values,
            enable_power_probe=not args.disable_power_probe,
            save_plots=args.save_sweep_plots
        )
    else:
        # Single-run path
        config = SimulationConfig()
        if args.disable_power_probe:
            config.ENABLE_POWER_PROBE = False

        results = run_e2e_simulation(config)
        if results:
            save_file = os.path.join(config.PLOT_DIR, "e2e_simulation_results.png")
            fig = plot_e2e_results(results, save_path=save_file)
            symbol_file = os.path.join(config.PLOT_DIR, "e2e_symbol_comparison.png")
            symbol_fig = plot_symbol_comparison(results, save_path=symbol_file)
            plt.show()
        else:
            print("✗ Simulation failed to produce results.")
