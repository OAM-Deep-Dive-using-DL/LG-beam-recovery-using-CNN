#!/usr/bin/env python3
"""
Quick diagnostic test - runs minimal simulation to check aperture mask behavior.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
from pipeline import SimulationConfig

print("=" * 80)
print("APERTURE MASK DIAGNOSTIC TEST")
print("=" * 80)

# Create minimal config for fast testing
cfg = SimulationConfig()
cfg.CN2 = 0  # No turbulence for fastest test
cfg.NUM_SCREENS = 1
cfg.N_INFO_BITS = 100
cfg.SPATIAL_MODES = [(0, -1), (0, 1), (0, -4), (0, 4)]  # Test low and high order
cfg.N_GRID = 256  # Smaller grid for speed
cfg.ENABLE_POWER_PROBE = True  # Need this for diagnostics

print(f"\nConfiguration:")
print(f"  Spatial modes: {cfg.SPATIAL_MODES}")
print(f"  Grid: {cfg.N_GRID}x{cfg.N_GRID}")
print(f"  Receiver diameter: {cfg.RECEIVER_DIAMETER} m")
print(f"  Distance: {cfg.DISTANCE} m")

print("\n" + "=" * 80)
print("Expected Results:")
print("=" * 80)
print("If beam is INSIDE aperture:")
print("  → Beam waists < 250mm")
print("  → Power reduction ≈ 0% (physically correct!)")
print("  → Radial distribution shows >95% power within r=0.25m")
print("")
print("If mask is BROKEN:")
print("  → Mask min/max not 0.0/1.0")
print("  → Unique values != 2")
print("")
print("If field is OUTSIDE aperture:")
print("  → Beam waists > 250mm")
print("  → Power reduction > 30%")
print("  → Radial distribution shows <70% power within r=0.25m")
print("=" * 80)

print("\nRun full simulation to see diagnostic output:")
print("  python pipeline.py")
print("\nOr run minimal test (faster):")
print("  python -c 'from pipeline import run_e2e_simulation, SimulationConfig; cfg=SimulationConfig(); cfg.CN2=0; cfg.NUM_SCREENS=1; cfg.N_INFO_BITS=100; cfg.SPATIAL_MODES=[(0,-1),(0,1)]; cfg.N_GRID=256; run_e2e_simulation(cfg)'")
print("=" * 80)
