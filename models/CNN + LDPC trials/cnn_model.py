"""
cnn_model.py (PyTorch Version)
Defines the multi-task CNN architecture for FSO-OAM.
- Head 1 (output_bits): Recovers the transmitted bits (Message)
- Head 2 (output_H): Estimates the channel matrix (Mode)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FSO_MultiTask_CNN(nn.Module):
    def __init__(self, input_channels=2, n_modes=6):
        """
        Initializes the multi-task CNN model.
        
        Parameters:
        - input_channels: 2 (Real, Imaginary)
        - n_modes: Number of spatial modes (e.g., 6)
        """
        super(FSO_MultiTask_CNN, self).__init__()
        
        self.n_modes = n_modes
        self.n_output_bits = n_modes * 2
        self.n_output_H_flat = n_modes * n_modes * 2
        
        # --- Convolutional Backbone (Shared Feature Extractor) ---
        # PyTorch is (N, C, H, W)
        
        # Input: (N, 2, 128, 128)
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 128 -> 64
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64 -> 32
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32 -> 16
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16 -> 8
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8 -> 4
        )
        
        # --- Classifier Head Base ---
        # The flattened size is 512 channels * 4 * 4 = 8192
        self.flatten = nn.Flatten(start_dim=1) # Flatten all dims except batch
        
        self.shared_dense = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        # --- HEAD 1: Message Recovery (Bits) ---
        self.bit_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.n_output_bits)
            # We output raw logits and use nn.BCEWithLogitsLoss (more stable)
        )
        
        # --- HEAD 2: Mode Recovery (Channel Matrix H) ---
        self.H_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.n_output_H_flat)
            # Output is linear for regression
        )

    def forward(self, x):
        # x shape: (N, 2, 128, 128)
        
        # Backbone
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Flatten
        x = self.flatten(x) # Shape: (N, 8192)
        
        # Shared Dense
        shared_out = self.shared_dense(x)
        
        # Heads
        output_bits = self.bit_head(shared_out) # Shape: (N, 12)
        
        output_H_flat = self.H_head(shared_out) # Shape: (N, 72)
        output_H = output_H_flat.view(-1, self.n_modes, self.n_modes, 2) # Shape: (N, 6, 6, 2)
        
        return output_bits, output_H

if __name__ == "__main__":
    from generate_dataset import DataConfig # Import config from data generator
    
    cfg = DataConfig()
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FSO_MultiTask_CNN(n_modes=cfg.N_MODES).to(device)
    
    # Test with a dummy input
    # (N, C, H, W) = (Batch, 2, 128, 128)
    dummy_input = torch.randn(16, 2, cfg.N_GRID, cfg.N_GRID).to(device)
    
    bits, H = model(dummy_input)
    
    print(f"Model built successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output 'bits' shape: {bits.shape}")
    print(f"Output 'H' shape: {H.shape}")
    
    # Simple parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")