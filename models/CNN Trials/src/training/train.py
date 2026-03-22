import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

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


def unwrap_dataset(dataset):
    """Return the underlying FSODataset if wrapped by torch.utils.data.Subset."""
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not mps_available:
            raise RuntimeError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device option: {device_arg}")


def get_runtime_config(device: torch.device, workers: int) -> dict:
    amp_device = device.type if device.type in {"cuda", "mps"} else None
    pin_memory = device.type == "cuda"
    persistent_workers = workers > 0
    return {
        "amp_device": amp_device,
        "amp_enabled": amp_device is not None,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "non_blocking": pin_memory,
    }

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
    device = resolve_device(args.device)
    runtime = get_runtime_config(device, args.workers)
    amp_device = runtime["amp_device"]
    amp_enabled = runtime["amp_enabled"]
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    if amp_enabled:
        print(f"Mixed precision: ENABLED on {amp_device}.")
    else:
        print("Mixed precision: DISABLED on CPU.")
    print(
        "Runtime config: "
        f"batch_size={args.batch_size}, workers={args.workers}, "
        f"pin_memory={runtime['pin_memory']}, "
        f"persistent_workers={runtime['persistent_workers']}"
    )
    
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
        
        
    print("Dataset: using stored physics-normalized intensity maps (no ImageNet normalization).")
    train_dataset = FSODataset(
        args.data_dir / f"{args.dataset_name}_train.h5",
        'train',
        augment=augment_enabled,
        normalize_mode='none'
    )
    if args.max_train_samples is not None:
        max_train = min(args.max_train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, range(max_train))
        print(f"Training subset enabled: {max_train} samples.")
    
    # Try to load validation dataset
    val_path = args.data_dir / f"{args.dataset_name}_val.h5"
    if val_path.exists():
        val_dataset = FSODataset(val_path, 'val', normalize_mode='none')
        if args.max_val_samples is not None:
            max_val = min(args.max_val_samples, len(val_dataset))
            val_dataset = Subset(val_dataset, range(max_val))
            print(f"Validation subset enabled: {max_val} samples.")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=runtime["pin_memory"],
            persistent_workers=runtime["persistent_workers"],
        )
        has_val = True
    else:
        print(f"Warning: Validation dataset not found: {val_path}")
        print(f"Training will proceed without validation.")
        has_val = False
        val_loader = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=runtime["pin_memory"],
        persistent_workers=runtime["persistent_workers"],
    )
    
    train_dataset_base = unwrap_dataset(train_dataset)

    # Model
    print(f"Initializing {args.backbone}...")
    model = FSOModel(n_modes=train_dataset_base.n_modes, backbone_name=args.backbone, dropout_rate=args.dropout).to(device)
    
    # Loss & Optimizer
    if args.loss == 'polar':
        print("Using PolarLoss (Magnitude + Phase Cosine)")
        criterion_sym = PolarLoss(alpha=1.0, beta=1.0)
    else:
        print("Using MSELoss")
        criterion_sym = nn.MSELoss()
        
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
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled) if amp_enabled else None
    
    # Training Loop
    # Resume logic
    start_epoch = 0
    best_val_loss = float('inf')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use backbone name for model filenames to avoid overwriting different architectures
    model_name = f"best_model_{args.backbone}.pth"
    last_name = f"last_model_{args.backbone}.pth"
    model_path = output_dir / model_name
    last_path = output_dir / last_name
    
    init_model_path = args.init_model_path.resolve() if args.init_model_path is not None else model_path

    if args.finetune:
        if init_model_path.exists():
            print(f"Finetuning from {init_model_path}...")
            checkpoint = torch.load(init_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
                 # Start from next epoch of best model (to keep history consistent-ish)
                 start_epoch = checkpoint.get('epoch', 0) + 1
                 best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            else:
                 model.load_state_dict(checkpoint)
        else:
            print(f"Error: {init_model_path} not found for finetuning.")
            sys.exit(1)
            
    elif args.resume:
        if last_path.exists():
            print(f"Resuming from {last_path}...")
            checkpoint = torch.load(last_path, map_location=device)
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
            print(f"Checkpoint {last_path} not found. Starting from scratch.")

    # Training Loop
    history = {'train_loss': [], 'val_loss': []}
    
    # Initialize Early Stopping with looser patience for Cosine Annealing
    # We want to survive the restarts
    early_stopping = EarlyStopping(patience=20, verbose=True, path=str(model_path))
    
    print(f"\nStarting training for {args.epochs} epochs (Backbone: {args.backbone})...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, syms, _pwrs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            imgs = imgs.to(device, non_blocking=runtime["non_blocking"])
            syms = syms.to(device, non_blocking=runtime["non_blocking"])
            

            optimizer.zero_grad()
            
            # Mixed Precision Context
            with torch.amp.autocast(amp_device, enabled=amp_enabled) if amp_enabled else torch.autocast(device_type='cpu', enabled=False):
                pred_syms, _pred_pwrs = model(imgs)
                
                loss_sym = criterion_sym(pred_syms, syms)
                if args.loss == 'polar':
                    loss = loss_sym
                else:
                    pred_magnitude = torch.norm(pred_syms, dim=-1)  # [batch, modes]
                    target_magnitude = torch.norm(syms, dim=-1)     # [batch, modes]
                    loss_magnitude = criterion_sym(pred_magnitude, target_magnitude)
                    loss = loss_sym + 0.2 * loss_magnitude
            
            # Loss Check
            if torch.isnan(loss):
                print(f"\n[!] NaN Loss detected at Epoch {epoch+1}, Step {len(train_loader)}")
                if scaler is not None:
                    print(f"    Scaler Scale: {scaler.get_scale()}")
                # Skip backward to let scaler recover if possible, or trigger early stop
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.update()
                continue
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            
            # Gradient Clipping (Pre-Update)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            train_loss += loss.item()
            
        # Update Scheduler (Called per epoch for CosineAnnealingLR)
        # Note: step() is called after the epoch
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate (if validation data exists)
        run_validation = has_val and ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1)
        if run_validation:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, syms, pwrs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    imgs = imgs.to(device, non_blocking=runtime["non_blocking"])
                    syms = syms.to(device, non_blocking=runtime["non_blocking"])
                    
                    pred_syms, _pred_pwrs = model(imgs)
                    
                    loss_sym = criterion_sym(pred_syms, syms)
                    if args.loss == 'polar':
                        loss = loss_sym
                    else:
                        pred_magnitude = torch.norm(pred_syms, dim=-1)
                        target_magnitude = torch.norm(syms, dim=-1)
                        loss_magnitude = criterion_sym(pred_magnitude, target_magnitude)
                        loss = loss_sym + 0.2 * loss_magnitude
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
                model.load_state_dict(torch.load(early_stopping.path, map_location=device))
                break
        else:
            # No validation data, just print training loss
            if has_val:
                print(f"Epoch {epoch+1:3d}/{args.epochs}: Train Loss={avg_train_loss:.4f}, Val skipped, LR={optimizer.param_groups[0]['lr']:.1e}")
                avg_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
            else:
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
        }, last_path)
        
        # Save Best (using early stopping)
        if has_val and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save weights separately
            torch.save(model.state_dict(), model_path) 
            print(f"  ✓ Saved {model_path}")
            
    # Plot History
            
    # Plot History
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training History ({args.backbone})')
    plt.savefig(output_dir / f'training_history_{args.backbone}.png')
    
    # Save History to JSON
    json_path = output_dir / f'training_history_{args.backbone}.json'
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining complete. Saved '{model_path}', '{json_path}', and '{output_dir / f'training_history_{args.backbone}.png'}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FSO-OAM CNN Receiver")
    parser.add_argument('--data_dir', type=Path, default=Path('../../data/generated_curriculum'),
                        help='Path to dataset directory (default: ../../data/generated_curriculum)')
    parser.add_argument('--dataset_name', type=str, default='fso_oam_turbulence_v1',
                        help='Dataset name prefix (e.g., fso_oam_turbulence_v1)')
    parser.add_argument('--output_dir', type=Path, default=Path('../outputs/training'),
                        help='Directory for checkpoints and training history')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32) # Reduced for 8GB Mac
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers') # Reduced for 8GB Mac
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device selection (default: auto = cuda > mps > cpu)')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', choices=['convnext_tiny', 'convnext_small', 'efficientnet_b0', 'efficientnet_v2_s'])
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('--lr', type=float, default=1e-4)  # Reduced to 1e-4 for stability
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--debug', action='store_true', help='Enable anomaly detection for debugging NaN')
    parser.add_argument('--finetune', action='store_true', help='Smart restart: Load best model, lower LR, and switch to ReduceLROnPlateau')
    parser.add_argument('--init_model_path', type=Path, default=None,
                        help='Explicit checkpoint/weights path to initialize from when finetuning')
    parser.add_argument('--augment', action='store_true', help='Enable physics-aware augmentation')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'polar'], help='Loss function to use')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optional cap on training samples for smoke tests')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Optional cap on validation samples for smoke tests')
    parser.add_argument('--val_every', type=int, default=1,
                        help='Run validation every N epochs (default: 1)')
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
