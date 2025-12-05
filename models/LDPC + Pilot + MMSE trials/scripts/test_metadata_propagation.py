#!/usr/bin/env python3
"""
Minimal test to check if noise_disabled flag is being passed through the pipeline.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
from pipeline import SimulationConfig
from encoding import encodingRunner, FSO_MDM_Frame

print("=" * 80)
print("METADATA PROPAGATION TEST")
print("=" * 80)

# Create config
cfg = SimulationConfig()
print(f"\n1. Config: ADD_NOISE = {cfg.ADD_NOISE}")
print(f"   Expected: noise_disabled = {not cfg.ADD_NOISE}")

# Create transmitter
print("\n2. Creating transmitter...")
transmitter = encodingRunner(
    spatial_modes=cfg.SPATIAL_MODES[:2],  # Use only 2 modes for speed
    wavelength=cfg.WAVELENGTH,
    w0=cfg.W0,
    fec_rate=cfg.FEC_RATE,
    pilot_ratio=cfg.PILOT_RATIO
)

# Generate minimal data
print("\n3. Generating test frame...")
data_bits = np.random.randint(0, 2, 100)
tx_frame = transmitter.transmit(data_bits, verbose=False)

# Check if metadata exists
print("\n4. Checking tx_frame metadata...")
if hasattr(tx_frame, 'metadata'):
    print(f"   ✓ tx_frame.metadata exists")
    print(f"   Keys: {list(tx_frame.metadata.keys())}")
    
    if 'noise_disabled' in tx_frame.metadata:
        print(f"   ✓ 'noise_disabled' key found")
        print(f"   Value: {tx_frame.metadata['noise_disabled']}")
    else:
        print(f"   ✗ 'noise_disabled' key NOT FOUND")
        print(f"   → This is the problem! The flag is not being set in pipeline.py")
else:
    print(f"   ✗ tx_frame.metadata does NOT exist")

# Now manually add the flag (as pipeline.py should do)
print("\n5. Manually adding noise_disabled flag (simulating pipeline.py)...")
if not hasattr(tx_frame, 'metadata') or tx_frame.metadata is None:
    tx_frame.metadata = {}
tx_frame.metadata['noise_disabled'] = not cfg.ADD_NOISE
print(f"   Set: tx_frame.metadata['noise_disabled'] = {tx_frame.metadata['noise_disabled']}")

# Test receiver
print("\n6. Testing receiver noise variance estimation...")
from receiver import ChannelEstimator
from encoding import PilotHandler

pilot_handler = PilotHandler(pilot_ratio=cfg.PILOT_RATIO)
chan_est = ChannelEstimator(pilot_handler, cfg.SPATIAL_MODES[:2])

# Create mock data
rx_symbols_per_mode = {mode: np.array([]) for mode in cfg.SPATIAL_MODES[:2]}
H_est = np.eye(2, dtype=complex)

# Call estimate_noise_variance
noise_var = chan_est.estimate_noise_variance(rx_symbols_per_mode, tx_frame, H_est)

print(f"\n7. Result:")
print(f"   Estimated noise_var = {noise_var:.3e}")
if noise_var == 1e-6:
    print(f"   ✅ SUCCESS: noise_var = 1e-6 (as expected when noise_disabled=True)")
else:
    print(f"   ✗ FAILURE: Expected 1e-6, got {noise_var:.3e}")

print("\n" + "=" * 80)
