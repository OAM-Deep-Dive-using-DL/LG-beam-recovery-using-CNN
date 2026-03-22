import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class FSODataset(Dataset):
    def __init__(self, h5_path, split='train', augment=False, cache_to_ram=False, normalize_mode='none'):
        self.h5_path = h5_path
        self.split = split
        self.augment = augment
        self.cache_to_ram = cache_to_ram
        self.normalize_mode = normalize_mode
        self.file = None  # Only used if NOT caching to RAM
        
        print(f"Initializing {split} dataset from {h5_path}...")
        
        if self.cache_to_ram:
            print("  -> Loading entire dataset into RAM (this may take a few seconds)...")
            with h5py.File(h5_path, 'r') as f:
                # Load all data into numpy arrays
                self.images = f['intensity'][:]
                self.symbols = f['symbols'][:]
                self.cn2 = f['cn2'][:]
                self.length = len(self.images)
                self.n_modes = int(f.attrs['n_modes'])
                self.spatial_modes = np.asarray(f.attrs['spatial_modes'])
                self.input_shape = tuple(np.asarray(f.attrs.get('input_shape', self.images.shape[1:]), dtype=int))
                self.generator = f.attrs.get('generator', 'unknown')
                self.noise_model = f.attrs.get('noise_model', 'unknown')
            print("  -> RAM Cache Complete.")
        else:
            print("  -> Using Lazy Loading (Disk I/O per sample)...")
            # Open briefly just to get metadata
            with h5py.File(h5_path, 'r') as f:
                self.length = len(f['intensity'])
                self.n_modes = int(f.attrs['n_modes'])
                self.spatial_modes = np.asarray(f.attrs['spatial_modes'])
                self.input_shape = tuple(np.asarray(f.attrs.get('input_shape', f['intensity'].shape[1:]), dtype=int))
                self.generator = f.attrs.get('generator', 'unknown')
                self.noise_model = f.attrs.get('noise_model', 'unknown')
                # Small enough to cache even for large datasets and needed by evaluation.
                self.cn2 = f['cn2'][:]

        if self.normalize_mode not in {'none', 'imagenet'}:
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")
            
        print(f"Found {self.length} samples.")
        print(f"  -> Generator: {self.generator}")
        print(f"  -> Noise Model: {self.noise_model}")
        if self.normalize_mode == 'none':
            print("  -> Loader normalization: NONE (uses stored physics-normalized intensity).")
        else:
            print("  -> Loader normalization: IMAGENET-style single-channel mean/std.")

    def __len__(self):
        return self.length

    def _normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == 'none':
            return img
        # Legacy compatibility path only; not recommended for the new curriculum HDF5s.
        return (img - 0.449) / 0.226

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            finally:
                self.file = None

    def __del__(self):
        self.close()

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
        img = self._normalize_image(img)
        
        # Legacy output to preserve the training/eval call signature.
        pwr = torch.ones(self.n_modes).float()
        
        return img, sym, pwr