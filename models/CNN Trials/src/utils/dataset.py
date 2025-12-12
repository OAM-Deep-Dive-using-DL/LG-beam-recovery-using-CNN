import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class FSODataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        
        print(f"Loading {split} data from {h5_path} into RAM...")
        with h5py.File(h5_path, 'r') as f:
            self.intensity = f['intensity'][:]
            self.symbols = f['symbols'][:]
            self.cn2 = f['cn2'][:]
            self.n_modes = f.attrs['n_modes']
            
        # Normalize intensity to [0, 1] if not already (it should be)
        # Add channel dim: [N, 64, 64] -> [N, 1, 64, 64]
        self.intensity = np.expand_dims(self.intensity, axis=1)
        
        print(f"Loaded {len(self.intensity)} samples.")

    def __len__(self):
        return len(self.intensity)

    def __getitem__(self, idx):
        # Inputs
        img = torch.from_numpy(self.intensity[idx]).float()
        
        # Targets
        sym = torch.from_numpy(self.symbols[idx]).float() # [8, 2]
        
        # Power target: Magnitude of symbols (should be 1.0 for QPSK)
        # But if we had fading, it might vary. For now, QPSK is constant envelope.
        # So target power is all 1s.
        # Wait, if a mode is NOT transmitted, power is 0.
        # But in our dataset, we transmit ALL modes currently?
        # Let's check config. Yes, we sum all modes.
        # So power target is always 1.0 for active modes.
        pwr = torch.ones(self.n_modes).float()
        
        return img, sym, pwr
