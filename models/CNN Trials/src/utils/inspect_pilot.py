import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_pilot(dataset_path):
    print(f"Inspecting {dataset_path}...")
    
    with h5py.File(dataset_path, 'r') as f:
        intensity = f['intensity'][:]
        
    print(f"Loaded {len(intensity)} samples.")
    
    # Calculate average intensity
    avg_intensity = np.mean(intensity, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(avg_intensity, cmap='hot')
    plt.title("Average Intensity (All Samples)")
    plt.colorbar()
    
    # Plot center profile
    center_y = avg_intensity.shape[0] // 2
    profile = avg_intensity[center_y, :]
    
    plt.subplot(1, 2, 2)
    plt.plot(profile)
    plt.title("Center Horizontal Profile")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("pilot_inspection.png")
    print("Saved 'pilot_inspection.png'")

if __name__ == "__main__":
    inspect_pilot("dataset/fso_oam_sanity_train.h5")
