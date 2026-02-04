import h5py
import numpy as np
import matplotlib.pyplot as plt

path = 'models/CNN Trials/data/dataset/toy_experiment_A_train.h5'
with h5py.File(path, 'r') as f:
    print("Keys:", list(f.keys()))
    
    # Check Intensity
    img = f['intensity'][0]
    print(f"Image 0 Shape: {img.shape}")
    print(f"Image 0 Stats: Min={img.min()}, Max={img.max()}, Mean={img.mean()}")
    
    # Check Labels
    sym = f['symbols'][0]
    print(f"Symbol 0 Shape: {sym.shape}")
    print(f"Symbol 0 Raw (I/Q): \n{sym}")
    
    # Check if symbols look like QPSK
    # QPSK magnitude should be consistent
    mags = np.linalg.norm(sym, axis=-1)
    print(f"Symbol 0 Magnitudes: {mags}")
    
    # Check Pilot vs Data
    # In this toy experiment, mode 0 is pilot, mode 1 is data? 
    # Wait, config says spatial_modes: [[0, 1]]. Pilot is [0, 0].
    # So index 0 in 'symbols' corresponds to mode [0, 1].
    
    # Let's save the image to check visually
    plt.imshow(img, cmap='inferno')
    plt.colorbar()
    plt.title(f"Sample 0 Intensity (Max={img.max():.4f})")
    plt.savefig('debug_sample_0.png')
    print("Saved debug_sample_0.png")
