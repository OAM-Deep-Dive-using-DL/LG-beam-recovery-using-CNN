import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import FSOModel
from utils.dataset import FSODataset
from training.loss import PolarLoss

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# Dataset Note:
# Training data should be generated using the canonical generator:
#   models/CNN Trials/data/generators/generate_dataset.py
# Default location: models/CNN Trials/data/dataset/

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.debug:
        print("!! DEBUG MODE: Anomaly Detection Enabled (Training will be slower) !!")
        torch.autograd.set_detect_anomaly(True)
    
    # Load Data
    # Enable physics-aware augmentation for training
    # Respect args.augmentation flag (default False for debugging/toy experiments)
    augment_enabled = args.augment if args.augment else False
    if augment_enabled:
        print("Physics-aware augmentation ENABLED.")
    else:
        print("Physics-aware augmentation DISABLED.")
        
        
    print(f"Dataset: Normalization [Mean=0.449, Std=0.226] ENABLED.")
    train_dataset = FSODataset(args.data_dir / f"{args.dataset_name}_train.h5", 'train', augment=augment_enabled)
    
    # Try to load validation dataset
    val_path = args.data_dir / f"{args.dataset_name}_val.h5"
    # Persistent workers only valid if workers > 0
    use_persistent = (args.workers > 0)
    
    if val_path.exists():
        val_dataset = FSODataset(val_path, 'val')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, persistent_workers=use_persistent)
        has_val = True
    else:
        print(f"Warning: Validation dataset not found: {val_path}")
        print(f"Training will proceed without validation.")
        has_val = False
        val_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, persistent_workers=use_persistent)
    
    # Model
    print(f"Initializing {args.backbone}...")
    model = FSOModel(n_modes=train_dataset.n_modes, backbone_name=args.backbone, dropout_rate=args.dropout).to(device)
    
    # Loss & Optimizer
    if args.loss == 'polar':
        print("Using PolarLoss (Magnitude + Phase Cosine)")
        criterion_sym = PolarLoss(alpha=1.0, beta=1.0)
    else:
        print("Using MSELoss")
        criterion_sym = nn.MSELoss()
        
    criterion_pwr = nn.BCELoss()
    
    # Tuning Hyperparameters based on mode
    current_lr = 5e-5 if args.finetune else args.lr
    if args.finetune:
        print(f"Finetuning Mode: Setting Base LR to {current_lr}")
    
    # Tunable weight decay
    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=args.weight_decay)
    
    # Scheduler setup
    if args.finetune:
        # Smart Restart: Use ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    else:
        # Normal Training: Cosine Annealing (Monotonic Descent) - No restarts to avoid "shocks"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('mps') # Use 'mps' for Mac
    
    # Training Loop
    # Resume logic
    start_epoch = 0
    best_val_loss = float('inf')
    # Use backbone name for model filenames to avoid overwriting different architectures
    model_name = f"best_model_{args.backbone}.pth"
    last_name = f"last_model_{args.backbone}.pth"
    
    if args.finetune:
        if (Path(model_name).exists()):
            print(f"Finetuning from {model_name}...")
            checkpoint = torch.load(model_name, map_location=device)
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
                 # Start from next epoch of best model (to keep history consistent-ish)
                 start_epoch = checkpoint.get('epoch', 0) + 1
                 best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            else:
                 model.load_state_dict(checkpoint)
        else:
            print(f"Error: {model_name} not found for finetuning.")
            sys.exit(1)
            
    elif args.resume:
        if (Path(last_name).exists()):
            print(f"Resuming from {last_name}...")
            checkpoint = torch.load(last_name, map_location=device)
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 # Only load scheduler if it matches the current type (basic check)
                 # If resuming a finetune run, we expect ReduceLROnPlateau state
                 # If resuming normal run, we expect Cosine state
                 # For simplicity, we try to load. If structure mismatches, it might fail or warn.
                 try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
                 except:
                    print("Warning: Could not load scheduler state. Starting fresh scheduler.")
                 
                 start_epoch = checkpoint['epoch'] + 1
                 best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            else:
                 # Legacy support for pure state_dict
                 model.load_state_dict(checkpoint)
        else:
            print(f"Checkpoint {last_name} not found. Starting from scratch.")

    # Training Loop
    history = {'train_loss': [], 'val_loss': []}
    
    # Initialize Early Stopping with looser patience for Cosine Annealing
    # We want to survive the restarts
    early_stopping = EarlyStopping(patience=20, verbose=True, path=model_name)
    
    print(f"\nStarting training for {args.epochs} epochs (Backbone: {args.backbone})...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, syms, pwrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            imgs, syms, pwrs = imgs.to(device), syms.to(device), pwrs.to(device)
            

            optimizer.zero_grad()
            
            # Mixed Precision Context
            with torch.amp.autocast('mps'):
                pred_syms, pred_pwrs = model(imgs)
                
                loss_sym = criterion_sym(pred_syms, syms)
                
                # Only compute power loss if power head exists
                if pred_pwrs is not None:
                    loss_pwr = criterion_pwr(pred_pwrs, pwrs)
                    
                    if args.loss == 'polar':
                        loss = criterion_sym(pred_syms, syms)
                        loss += 0.1 * loss_pwr
                    else:
                        # Add magnitude regularization
                        pred_magnitude = torch.norm(pred_syms, dim=-1)  # [batch, modes]
                        target_magnitude = torch.norm(syms, dim=-1)     # [batch, modes]
                        loss_magnitude = criterion_sym(pred_magnitude, target_magnitude)
                        # Weighted sum with magnitude emphasis
                        loss = loss_sym + 0.1 * loss_pwr + 0.2 * loss_magnitude
                else:
                    if args.loss == 'polar':
                        loss = criterion_sym(pred_syms, syms)
                    else:
                        # No power head - use symbol loss + magnitude regularization only
                        pred_magnitude = torch.norm(pred_syms, dim=-1)  # [batch, modes]
                        target_magnitude = torch.norm(syms, dim=-1)     # [batch, modes]
                        loss_magnitude = criterion_sym(pred_magnitude, target_magnitude)
                        loss = loss_sym + 0.2 * loss_magnitude
            
            # Loss Check
            if torch.isnan(loss):
                print(f"\n[!] NaN Loss detected at Epoch {epoch+1}, Step {len(train_loader)}")
                print(f"    Scaler Scale: {scaler.get_scale()}")
                # Skip backward to let scaler recover if possible, or trigger early stop
                optimizer.zero_grad()
                scaler.update() 
                continue
            
            # Scaled Backward Pass
            scaler.scale(loss).backward()
            
            # Unscale logic
            scaler.unscale_(optimizer)
            
            # Gradient Clipping (Pre-Update)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        # Update Scheduler (Called per epoch for CosineAnnealingLR)
        # Note: step() is called after the epoch
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate (if validation data exists)
        if has_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, syms, pwrs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    imgs, syms, pwrs = imgs.to(device), syms.to(device), pwrs.to(device)
                    
                    pred_syms, pred_pwrs = model(imgs)
                    
                    loss_sym = criterion_sym(pred_syms, syms)
                    
                    # Only compute power loss if power head exists
                    if pred_pwrs is not None:
                        loss_pwr = criterion_pwr(pred_pwrs, pwrs)
                        # Add magnitude regularization to validation too
                        pred_magnitude = torch.norm(pred_syms, dim=-1)
                        target_magnitude = torch.norm(syms, dim=-1)
                        loss_magnitude = criterion_sym(pred_magnitude, target_magnitude)
                    else:
                         # No power head - validation with symbol loss + magnitude only
                         pred_magnitude = torch.norm(pred_syms, dim=-1)
                         target_magnitude = torch.norm(syms, dim=-1)
                         
                         if args.loss == 'polar':
                             loss = criterion_sym(pred_syms, syms)
                             if pred_pwrs is not None:
                                 loss += 0.1 * criterion_pwr(pred_pwrs, pwrs)
                         else:
                             loss_magnitude = criterion_sym(pred_magnitude, target_magnitude)
                             loss = loss_sym + 0.2 * loss_magnitude
                             if pred_pwrs is not None:
                                 loss += 0.1 * criterion_pwr(pred_pwrs, pwrs)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1:3d}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}")
            
            # Step Scheduler for Plateau (requires validation metric)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            
            # Check Early Stopping
            early_stopping(avg_val_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping triggered. Recovering best model weights.")
                model.load_state_dict(torch.load(early_stopping.path))
                break
        else:
            # No validation data, just print training loss
            print(f"Epoch {epoch+1:3d}/{args.epochs}: Train Loss={avg_train_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.1e}")
            avg_val_loss = float('inf')  # Dummy value for checkpoint saving
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Save Checkpoint (Last)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }, last_name)
        
        # Save Best (using early stopping)
        if has_val and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save weights separately
            torch.save(model.state_dict(), model_name) 
            print(f"  ✓ Saved {model_name}")
            
    # Plot History
            
    # Plot History
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training History ({args.backbone})')
    plt.savefig(f'training_history_{args.backbone}.png')
    
    # Save History to JSON
    json_path = f'training_history_{args.backbone}.json'
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining complete. Saved '{model_name}', '{json_path}', and 'training_history_{args.backbone}.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FSO-OAM CNN Receiver")
    parser.add_argument('--data_dir', type=Path, default=Path('../../data/dataset'),
                        help='Path to dataset directory (default: ../../data/dataset)')
    parser.add_argument('--dataset_name', type=str, default='fso_oam_turbulence_v1',
                        help='Dataset name prefix (e.g., fso_oam_turbulence_v1)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32) # Reduced for 8GB Mac
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers') # Reduced for 8GB Mac
    parser.add_argument('--backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small', 'efficientnet_b0', 'efficientnet_v2_s'])
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('--lr', type=float, default=1e-4)  # Reduced to 1e-4 for stability
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--debug', action='store_true', help='Enable anomaly detection for debugging NaN')
    parser.add_argument('--finetune', action='store_true', help='Smart restart: Load best model, lower LR, and switch to ReduceLROnPlateau')
    parser.add_argument('--augment', action='store_true', help='Enable physics-aware augmentation')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'polar'], help='Loss function to use')
    args = parser.parse_args()
    
    # Validate dataset exists
    train_path = args.data_dir / f"{args.dataset_name}_train.h5"
    val_path = args.data_dir / f"{args.dataset_name}_val.h5"
    
    if not train_path.exists():
        print(f"\nError: Training dataset not found: {train_path}")
        print(f"\nGenerate dataset first using the canonical generator:")
        print(f"  cd ../../data/generators")
        print(f"  python generate_dataset.py --config configs/config.json --split all")
        print(f"\nOr specify custom dataset location with --data_dir flag.\n")
        import sys
        sys.exit(1)
    
    if not val_path.exists():
        print(f"\nWarning: Validation dataset not found: {val_path}")
        print(f"Training will proceed but validation metrics will be unavailable.\n")
    
    train(args)
