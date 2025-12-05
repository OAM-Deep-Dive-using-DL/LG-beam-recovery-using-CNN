import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MultiHeadResNet

class FSODataset(Dataset):
    def __init__(self, h5_path, split='train'):
        self.h5_path = h5_path
        self.split = split
        
        # Open file to read metadata (keep open? or open per access? 
        # h5py is not thread-safe with multiprocessing dataloaders usually.
        # Best practice: Open in __getitem__ or use 'swmr' mode if writing.
        # Since we are reading static files, we can open in __init__ but need care with workers.
        # Actually, for PyTorch DataLoader with num_workers > 0, we should open in __getitem__.
        # But that's slow. 
        # Solution: Open in __init__, but close and reopen in worker_init_fn?
        # Simpler: Read all into memory if small (25k samples * 16KB = 400MB -> Easy).
        
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    train_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_train.h5", 'train')
    val_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_val.h5", 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = MultiHeadResNet(n_modes=train_dataset.n_modes).to(device)
    
    # Loss & Optimizer
    criterion_sym = nn.MSELoss()
    criterion_pwr = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, syms, pwrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            imgs, syms, pwrs = imgs.to(device), syms.to(device), pwrs.to(device)
            
            optimizer.zero_grad()
            pred_syms, pred_pwrs = model(imgs)
            
            loss_sym = criterion_sym(pred_syms, syms)
            loss_pwr = criterion_pwr(pred_pwrs, pwrs)
            
            # Weighted sum
            loss = loss_sym + 0.1 * loss_pwr
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, syms, pwrs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                imgs, syms, pwrs = imgs.to(device), syms.to(device), pwrs.to(device)
                
                pred_syms, pred_pwrs = model(imgs)
                
                loss_sym = criterion_sym(pred_syms, syms)
                loss_pwr = criterion_pwr(pred_pwrs, pwrs)
                
                loss = loss_sym + 0.1 * loss_pwr
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Update Scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}")
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  ✓ Saved best model")
            
    # Plot History
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('training_history.png')
    print("\nTraining complete. Saved 'best_model.pth' and 'training_history.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=Path('dataset'))
    parser.add_argument('--dataset_name', type=str, default='fso_oam_turbulence_v1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
