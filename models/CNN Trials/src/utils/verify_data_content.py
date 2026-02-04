
import h5py
import numpy as np
import argparse
import sys
from pathlib import Path

def verify_h5(path):
    path = Path(path).resolve()
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)
        
    print(f"=== Inspecting {path.name} ===")
    
    with h5py.File(path, 'r') as f:
        # 1. Attributes
        print("\n[Attributes]")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")
            
        # 2. Datasets
        print("\n[Datasets]")
        if 'intensity' not in f or 'symbols' not in f:
            print("❌ Critical: Missing 'intensity' or 'symbols' dataset!")
            return
            
        intensity = f['intensity']
        symbols = f['symbols']
        cn2 = f['cn2']
        
        print(f"  intensity: {intensity.shape}, dtype={intensity.dtype}")
        print(f"  symbols:   {symbols.shape}, dtype={symbols.dtype}")
        print(f"  cn2:       {cn2.shape}, dtype={cn2.dtype}")
        
        # 3. Value Checks (Load first chunk)
        print("\n[Value Verification]")
        # Check first 100 or all
        subset_size = min(len(intensity), 100)
        data = intensity[:subset_size]
        
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        
        print(f"  Range:     [{min_val:.6f}, {max_val:.6f}]")
        print(f"  Mean:      {mean_val:.6f}")
        
        if max_val > 1.05: # Allow small epsilon
             print("❌ ERROR: Data is NOT normalized to [0, 1]! range exceeds 1.0")
        elif max_val > 0.1: # Must have some signal
             print("✓ Range is within [0, 1].")
        else:
             print("⚠️ Warning: Max value is very low. Might be empty/black images?")
             
        if np.isnan(data).any():
             print("❌ ERROR: NaNs detected!")
        else:
             print("✓ No NaNs.")
             
        # Check Symbols
        sym_data = symbols[:subset_size]
        print(f"  Symbols Range: [{np.min(sym_data):.4f}, {np.max(sym_data):.4f}]")
        if np.isnan(sym_data).any():
             print("❌ ERROR: NaNs in symbols!")
             
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to h5 file')
    args = parser.parse_args()
    verify_h5(args.path)
