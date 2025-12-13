import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MultiHeadResNet

from utils.dataset import FSODataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    train_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_train.h5", 'train')
    val_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_val.h5", 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = MultiHeadResNet(n_modes=train_dataset.n_modes, backbone_name=args.backbone).to(device)
    
    # Loss & Optimizer
    criterion_sym = nn.MSELoss()
    criterion_pwr = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training Loop
    # Resume logic
    start_epoch = 0
    best_val_loss = float('inf')
    model_name = f"best_model_{args.backbone}.pth"
    last_name = f"last_model_{args.backbone}.pth"
    
    if args.resume:
        if (Path(last_name).exists()):
            print(f"Resuming from {last_name}...")
            checkpoint = torch.load(last_name, map_location=device)
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 start_epoch = checkpoint['epoch'] + 1
                 best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            else:
                 # Legacy support for pure state_dict
                 model.load_state_dict(checkpoint)
        else:
            print(f"Checkpoint {last_name} not found. Starting from scratch.")

    # Training Loop
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training for {args.epochs} epochs (Backbone: {args.backbone})...")
    
    for epoch in range(start_epoch, args.epochs):
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
        
        # Save Checkpoint (Last)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, last_name)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_name) # Save just weights for easy loading
            print(f"  ✓ Saved {model_name}")
            
    # Plot History
            
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
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet18_cbam'])
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
