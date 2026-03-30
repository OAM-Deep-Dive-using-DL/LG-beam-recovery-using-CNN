
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
LEVEL = "curriculum_lvl1_ideal"

def inspect_file(filename):
    path = DATA_DIR / filename
    print(f"\nInspecting {filename}...")
    if not path.exists():
        print("  ❌ File not found")
        return

    with h5py.File(path, 'r') as f:
        img = f['intensity']
        sym = f['symbols']
        n_modes = f.attrs['n_modes']
        print(f"  Shape: Img {img.shape}, Sym {sym.shape}")
        
        # Stats
        img_slice = img[:100]
        sym_slice = sym[:100]
        
        print(f"  Img Mean: {np.mean(img_slice):.4f}, Std: {np.std(img_slice):.4f}, Max: {np.max(img_slice):.4f}")
        print(f"  Sym Mean: {np.mean(sym_slice):.4f}, Std: {np.std(sym_slice):.4f}")
        
        # Check first symbol
        print(f"  Sample 0 Symbol 0: {sym_slice[0, 0]}")
        
        return sym_slice

print("=== Data Inspection ===")
sym_train = inspect_file(f"{LEVEL}_train.h5")
sym_val = inspect_file(f"{LEVEL}_val.h5")
sym_test = inspect_file(f"{LEVEL}_test.h5")

# Check distribution consistency
if sym_train is not None and sym_test is not None:
    print("\nComparing Train vs Test Symbols:")
    print(f"  Train Std: {np.std(sym_train):.4f}")
    print(f"  Test Std:  {np.std(sym_test):.4f}")
    
    # Check if Test symbols look like valid QPSK (+/- 0.707)
    # Combine real and imag
    test_flat = sym_test.flatten()
    print(f"  Test Values Sample (first 10): {test_flat[:10]}")
