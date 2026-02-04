import h5py
import numpy as np

path = 'models/CNN Trials/data/dataset/config_fso_test.h5'
with h5py.File(path, 'r') as f:
    intensity = f['intensity'][:1000] # Check first 1000
    print(f"Shape: {intensity.shape}")
    print(f"Min: {np.min(intensity)}")
    print(f"Max: {np.max(intensity)}")
    print(f"Mean: {np.mean(intensity)}")
    print(f"Std: {np.std(intensity)}")
