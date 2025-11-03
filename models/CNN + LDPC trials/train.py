import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# --- Import our custom files ---
from cnn_model import FSO_MultiTask_CNN
try:
    from generate_dataset import DataConfig # Use the same config
except ImportError:
    print("Warning: Could not import DataConfig. Using fallback values.")
    class DataConfig:
        N_GRID = 128
        N_MODES = 6

# --- 1. Parameters ---
DATA_FILE = "fso_cnn_dataset.h5"
MODEL_FILE = "fso_multitask_model.pth" # PyTorch model file
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 12
VALIDATION_SPLIT = 0.15 # Use 15% for validation

# Loss weights
LOSS_WEIGHTS = {
    'bits': 1.0,
    'H': 0.1
}

# --- 2. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"--- Using device: {device} ---")


# --- 3. CUSTOM HDF5 DATASET CLASS ---
class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        if not os.path.exists(h5_path):
            print(f"Error: Dataset file not found at {h5_path}")
            print("Please run generate_dataset.py first.")
            sys.exit(1)
            
        self.h5_path = h5_path
        
        # We open the file here, but DO NOT load data.
        # This handle will be 'None' and re-opened in __getitem__
        # to support multiprocessing (num_workers > 0)
        self.h5_file = None
        
        # Open the file once just to get the length
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['X'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open the file if it's not open (for multi-worker loading)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Get ONE sample from disk
        X_sample = self.h5_file['X'][idx]
        Y_bits_sample = self.h5_file['Y_bits'][idx]
        Y_H_sample = self.h5_file['Y_H'][idx]
        
        # Convert to Tensors and format
        # X: (H, W, C) -> (C, H, W)
        X_tensor = torch.from_numpy(X_sample).float().permute(2, 0, 1)
        Y_bits_tensor = torch.from_numpy(Y_bits_sample).float()
        Y_H_tensor = torch.from_numpy(Y_H_sample).float()
        
        return X_tensor, Y_bits_tensor, Y_H_tensor

# --- 4. Create DataLoaders (No Memory Overload) ---
print(f"--- Initializing DataLoaders from {DATA_FILE} ---")
# Create the single, large dataset object
full_dataset = HDF5Dataset(DATA_FILE)

# Create indices for splitting
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SPLIT * dataset_size))
np.random.seed(42) # Ensure reproducible split
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Create PyTorch Subset objects
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

print(f"Total samples: {dataset_size}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# --- RECTIFIED: num_workers=0 ---
# HDF5 is not "thread-safe" and conflicts with PyTorch's
# multiprocessing. Setting num_workers=0 forces it to load
# in the main thread, which is safer and avoids crashes.
loader_args = {'num_workers': 0, 'pin_memory': True} if device.type != 'cpu' else {'num_workers': 0}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_args)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_args)

# --- 5. Build Model, Losses, and Optimizer ---
print("\n--- Building Model ---")
cfg = DataConfig()
model = FSO_MultiTask_CNN(input_channels=2, n_modes=cfg.N_MODES).to(device)

criterion_bits = nn.BCEWithLogitsLoss()
criterion_H = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=1e-6)

# --- 6. Training Loop ---
print(f"--- Starting Training for {EPOCHS} Epochs ---")
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0
    
    for X_batch, Y_bits_batch, Y_H_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        X_batch = X_batch.to(device)
        Y_bits_batch = Y_bits_batch.to(device)
        Y_H_batch = Y_H_batch.to(device)
        
        optimizer.zero_grad()
        pred_bits, pred_H = model(X_batch)
        loss_bits = criterion_bits(pred_bits, Y_bits_batch)
        loss_H = criterion_H(pred_H, Y_H_batch)
        loss = (LOSS_WEIGHTS['bits'] * loss_bits) + (LOSS_WEIGHTS['H'] * loss_H)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_bits_batch, Y_H_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
            X_batch = X_batch.to(device)
            Y_bits_batch = Y_bits_batch.to(device)
            Y_H_batch = Y_H_batch.to(device)
            
            pred_bits, pred_H = model(X_batch)
            loss_bits = criterion_bits(pred_bits, Y_bits_batch)
            loss_H = criterion_H(pred_H, Y_H_batch)
            loss = (LOSS_WEIGHTS['bits'] * loss_bits) + (LOSS_WEIGHTS['H'] * loss_H)
            
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        print(f"  Val loss improved ({best_val_loss:.6f} -> {avg_val_loss:.6f}). Saving model to {MODEL_FILE}")
        torch.save(model.state_dict(), MODEL_FILE)
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  Val loss did not improve. Early stopping counter: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"--- Early stopping triggered after {epoch+1} epochs. ---")
        break

# --- 7. Evaluate Model ---
print("\n--- Evaluating Model on Test Set ---")
# Load the best model
try:
    model.load_state_dict(torch.load(MODEL_FILE))
except Exception as e:
    print(f"Could not load best model, using last model state. Error: {e}")
model.eval()

# We'll use the val_loader (which reads from disk) for evaluation
Y_pred_bits_prob_list = []
Y_pred_H_list = []
Y_bits_test_list = []
Y_H_test_list = []

with torch.no_grad():
    for X_batch, Y_bits_batch, Y_H_batch in tqdm(val_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        
        pred_bits_logits, pred_H = model(X_batch)
        
        Y_pred_bits_prob_list.append(torch.sigmoid(pred_bits_logits).cpu())
        Y_pred_H_list.append(pred_H.cpu())
        Y_bits_test_list.append(Y_bits_batch.cpu())
        Y_H_test_list.append(Y_H_batch.cpu())

# Concatenate all batches
Y_pred_bits_prob_numpy = torch.cat(Y_pred_bits_prob_list).numpy()
Y_pred_H_numpy = torch.cat(Y_pred_H_list).numpy()
Y_bits_test_numpy = torch.cat(Y_bits_test_list).numpy()
Y_H_test_numpy = torch.cat(Y_H_test_list).numpy()

# --- A. Bit Error Rate (BER) Calculation ---
Y_pred_bits = (Y_pred_bits_prob_numpy > 0.5).astype(int)
total_bits = np.prod(Y_bits_test_numpy.shape)
bit_errors = np.sum(Y_pred_bits != Y_bits_test_numpy)
ber = bit_errors / total_bits

print("\n--- FINAL PERFORMANCE (Message Recovery) ---")
print(f"Total Test Bits: {total_bits}")
print(f"Total Bit Errors: {bit_errors}")
print(f"Raw CNN Bit Error Rate (BER): {ber:.4e}")

# --- B. Channel Matrix (H) Evaluation ---
H_mse = np.mean((Y_H_test_numpy - Y_pred_H_numpy)**2)
print(f"\n--- FINAL PERFORMANCE (Mode Recovery) ---")
print(f"Channel Matrix H (Real/Imag) MSE: {H_mse:.4e}")

# --- 8. Visualize H-Matrix Estimation ---
print("Generating visualization of H-matrix estimation...")
sample_idx = 42 # Pick a random sample from the test set

H_true_ri = Y_H_test_numpy[sample_idx]
H_pred_ri = Y_pred_H_numpy[sample_idx]

H_true_complex = H_true_ri[:,:,0] + 1j * H_true_ri[:,:,1]
H_pred_complex = H_pred_ri[:,:,0] + 1j * H_pred_ri[:,:,1]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle(f"CNN 'Mode Recovery' on Test Sample #{sample_idx}\n"
             f"H Matrix MSE: {np.mean((H_true_ri - H_pred_ri)**2):.2e}", 
             fontsize=16)

vmax = np.max(np.abs(H_true_complex))
im1 = axes[0, 0].imshow(np.abs(H_true_complex), cmap='viridis', vmin=0, vmax=vmax)
axes[0, 0].set_title("Ground Truth |H|")
plt.colorbar(im1, ax=axes[0, 0], label="Magnitude")

im2 = axes[0, 1].imshow(np.abs(H_pred_complex), cmap='viridis', vmin=0, vmax=vmax)
axes[0, 1].set_title("CNN Predicted |H|")
plt.colorbar(im2, ax=axes[0, 1], label="Magnitude")

im3 = axes[1, 0].imshow(np.angle(H_true_complex), cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[1, 0].set_title("Ground Truth angle(H)")
plt.colorbar(im3, ax=axes[1, 0], label="Phase [rad]")

im4 = axes[1, 1].imshow(np.angle(H_pred_complex), cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[1, 1].set_title("CNN Predicted angle(H)")
plt.colorbar(im4, ax=axes[1, 1], label="Phase [rad]")

for ax in axes.flat:
    ax.set_xlabel("Transmitted Mode Index")
    ax.set_ylabel("Received Mode Index")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cnn_H_matrix_evaluation.png")
plt.show()

print(f"\n--- Training complete ---")
print(f"  Final Raw BER: {ber:.4e}")
print(f"  Final H_MSE: {H_mse:.4e}")
print(f"  Best model saved to: {MODEL_FILE}")
print(f"  H-matrix plot saved to: cnn_H_matrix_evaluation.png")