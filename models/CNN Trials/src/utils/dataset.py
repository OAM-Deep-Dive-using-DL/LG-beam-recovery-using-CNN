import torch
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np

class FSODataset(Dataset):
    def __init__(self, h5_path, split='train', augment=False, cache_to_ram=False):
        self.h5_path = h5_path
        self.split = split
        self.augment = augment
        self.cache_to_ram = cache_to_ram
        self.file = None # Only used if NOT caching to RAM
        
        print(f"Initializing {split} dataset from {h5_path}...")
        
        if self.cache_to_ram:
            print("  -> Loading entire dataset into RAM (this may take a few seconds)...")
            with h5py.File(h5_path, 'r') as f:
                # Load all data into numpy arrays
                self.images = f['intensity'][:]
                self.symbols = f['symbols'][:]
                self.length = len(self.images)
                self.n_modes = f.attrs['n_modes']
                self.spatial_modes = f.attrs['spatial_modes'][:]
            print("  -> RAM Cache Complete.")
        else:
            print("  -> Using Lazy Loading (Disk I/O per sample)...")
            # Open briefly just to get metadata
            with h5py.File(h5_path, 'r') as f:
                self.length = len(f['intensity'])
                self.n_modes = f.attrs['n_modes']
                # Load spatial modes small enough for RAM
                self.spatial_modes = f.attrs['spatial_modes'][:]
            
        # Pre-initialize Normalization
        # Mean: 0.449, Std: 0.226 (Average of R,G,B ImageNet stats)
        norm_mean = [0.449]
        norm_std = [0.226]
        self.transform = transforms.Normalize(mean=norm_mean, std=norm_std)
            
        print(f"Found {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.cache_to_ram:
            # Read from RAM (Fast)
            # Copy already handled by slicing numpy array, but to be safe for torch striding
            img_np = self.images[idx].copy()
            sym_np = self.symbols[idx].copy()
        else:
            # Read from disk (Slow)
            if self.file is None:
                self.file = h5py.File(self.h5_path, 'r')
                
            # Read from disk on demand
            # Copy is essential to detach from HDF5 object
            img_np = self.file['intensity'][idx].copy() 
            sym_np = self.file['symbols'][idx].copy()
        
        # Add channel dim: [64, 64] -> [1, 64, 64] (or 128x128)
        img_np = np.expand_dims(img_np, axis=0) # Axis 0 because it's single sample [H, W] -> [C, H, W]
        
        # Physics-Aware Augmentation
        if self.augment:
            # Random multiple of 90 degrees: 0, 1, 2, 3
            k = np.random.randint(0, 4)
            if k > 0:
                # Rotate image (k * 90 degrees counter-clockwise)
                # axes=(1, 2) because img is [C, H, W]
                img_np = np.rot90(img_np, k, axes=(1, 2))
                
                # Update Symbols (Physics: Phase Shift)
                # Rotation by alpha = k * pi/2
                # New Sym = Old Sym * exp(-i * l * alpha)
                #         = Old Sym * exp(-i * l * k * pi/2)
                alpha = k * (np.pi / 2.0)
                
                # Complexify symbols for easy rotation
                # sym_np is [n_modes, 2] -> [n_modes] (complex)
                sym_complex = sym_np[:, 0] + 1j * sym_np[:, 1]
                
                # Apply phase shift per mode
                for i, mode in enumerate(self.spatial_modes):
                    _, l = mode
                    phase_shift = np.exp(-1j * l * alpha)
                    sym_complex[i] = sym_complex[i] * phase_shift
                    
                # De-complexify
                sym_np[:, 0] = np.real(sym_complex)
                sym_np[:, 1] = np.imag(sym_complex)

        # To Tensor
        img = torch.from_numpy(img_np.copy()).float()
        sym = torch.from_numpy(sym_np.copy()).float()

        # Apply Normalization
        img = self.transform(img)
        
        # Power target (assuming constant 1.0 for active modes in this dataset)
        pwr = torch.ones(self.n_modes).float()
        
        return img, sym, pwr