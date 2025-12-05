import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

f_path = "fso_oam_turbulence_v1_train.h5"

if not os.path.exists(f_path):
    print(f"File not found: {f_path}")
    exit(1)

with h5py.File(f_path, 'r') as f:
    print(f"Keys: {list(f.keys())}")
    print(f"Attrs: {dict(f.attrs)}")
    
    intensity = f['intensity'][:]
    symbols = f['symbols'][:]
    cn2 = f['cn2'][:]
    
    print(f"Intensity shape: {intensity.shape}, dtype: {intensity.dtype}")
    print(f"Symbols shape: {symbols.shape}, dtype: {symbols.dtype}")
    print(f"Cn2 shape: {cn2.shape}")
    
    print(f"Intensity range: [{np.min(intensity):.4f}, {np.max(intensity):.4f}]")
    print(f"Cn2 range: [{np.min(cn2):.2e}, {np.max(cn2):.2e}]")
    
    # Check for NaNs
    if np.isnan(intensity).any():
        print("WARNING: NaNs in intensity!")
    else:
        print("No NaNs in intensity.")
        
    # Plot first sample
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(intensity[0], cmap='hot')
    plt.title(f"Sample 0 (Cn2={cn2[0]:.2e})")
    plt.colorbar()
    
    plt.subplot(122)
    plt.imshow(intensity[-1], cmap='hot')
    plt.title(f"Sample -1 (Cn2={cn2[-1]:.2e})")
    plt.colorbar()
    
    plt.savefig("dataset_inspection.png")
    print("Saved dataset_inspection.png")
