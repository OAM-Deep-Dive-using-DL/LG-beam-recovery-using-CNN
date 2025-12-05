#!/usr/bin/env python3
"""
Force clean test - removes Python cache and runs a minimal simulation.
"""

import sys
import os
import shutil

# Remove Python cache
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
pycache_dir = os.path.join(SCRIPT_DIR, "__pycache__")
if os.path.exists(pycache_dir):
    print(f"Removing Python cache: {pycache_dir}")
    shutil.rmtree(pycache_dir)
    print("✓ Cache cleared")

# Force reimport
if 'pipeline' in sys.modules:
    del sys.modules['pipeline']
if 'receiver' in sys.modules:
    del sys.modules['receiver']
if 'encoding' in sys.modules:
    del sys.modules['encoding']

sys.path.insert(0, SCRIPT_DIR)

print("\n" + "=" * 80)
print("CLEAN TEST: Noise Variance Fix")
print("=" * 80)

# Now import fresh modules
from pipeline import run_e2e_simulation, SimulationConfig

# Create a minimal config for fast testing
cfg = SimulationConfig()
cfg.CN2 = 0  # No turbulence for fastest test
cfg.NUM_SCREENS = 1  # Minimal screens
cfg.N_INFO_BITS = 100  # Minimal data
cfg.SPATIAL_MODES = [(0, -1), (0, 1)]  # Only 2 modes
cfg.N_GRID = 256  # Smaller grid
cfg.ADD_NOISE = False  # Noise disabled
cfg.ENABLE_POWER_PROBE = False  # Skip power probe for speed

print(f"\nTest Configuration:")
print(f"  ADD_NOISE = {cfg.ADD_NOISE}")
print(f"  Spatial modes: {len(cfg.SPATIAL_MODES)}")
print(f"  Info bits: {cfg.N_INFO_BITS}")
print(f"  Grid: {cfg.N_GRID}x{cfg.N_GRID}")
print(f"  Cn² = {cfg.CN2}")

print("\n" + "-" * 80)
print("Running simulation...")
print("-" * 80 + "\n")

results = run_e2e_simulation(cfg, verbose=True)

if results:
    metrics = results['metrics']
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Noise variance: {metrics['noise_var']:.3e}")
    print(f"BER: {metrics['ber']:.3e}")
    print(f"Bit errors: {metrics['bit_errors']} / {metrics['total_bits']}")
    
    if metrics['noise_var'] == 1e-6:
        print("\n✅ SUCCESS: noise_var = 1e-6 (fix is working!)")
    else:
        print(f"\n✗ FAILURE: noise_var = {metrics['noise_var']:.3e} (expected 1e-6)")
        print("   → The fix is NOT being applied. Check if pipeline.py line 168 is executed.")
    
    if metrics['ber'] < 0.01:
        print(f"✅ SUCCESS: BER = {metrics['ber']:.3e} (< 1%)")
    else:
        print(f"⚠️  WARNING: BER = {metrics['ber']:.3e} (> 1%)")
else:
    print("\n✗ Simulation failed to produce results")

print("=" * 80)
