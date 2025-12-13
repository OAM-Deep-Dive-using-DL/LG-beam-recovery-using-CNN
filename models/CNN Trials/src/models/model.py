import torch
import torch.nn as nn
import torchvision.models as models
from resnet_cbam import resnet18_cbam

class MultiHeadResNet(nn.Module):
    """
    Multi-Head ResNet for FSO-OAM Signal Recovery.
    
    Backbone: ResNet-18 (Modified for 1-channel, 64x64 input)
    
    Heads:
    1. Symbol Head (Regression): Predicts Real/Imag parts of QPSK symbols.
       Output: [batch, n_modes, 2]
    2. Power Head (Classification/Regression): Predicts mode power presence.
       Output: [batch, n_modes] (Sigmoid activation)
    """
    def __init__(self, n_modes=8, input_channels=1, backbone_name='resnet18'):
        super(MultiHeadResNet, self).__init__()
        
        # Load Backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        elif backbone_name == 'resnet18_cbam':
            self.backbone = resnet18_cbam(pretrained=False) # No imagenet weights for custom structure
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        # Modify first layer for 1-channel input (Intensity)
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We keep stride=2 for 64x64 -> 32x32
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the original FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 1. Symbol Head (Regression)
        # Predicts Real and Imag parts for each mode
        self.symbol_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_modes * 2)  # Output: Flattened [Re_0, Im_0, Re_1, Im_1, ...]
        )
        
        # 2. Power Head (Auxiliary Task)
        # Predicts probability/energy of each mode
        self.power_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_modes),
            nn.Sigmoid()  # Output: [0, 1] for each mode
        )
        
        self.n_modes = n_modes

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        
        # Symbol Prediction
        symbols_flat = self.symbol_head(features)
        # Reshape to [batch, n_modes, 2]
        symbols = symbols_flat.view(-1, self.n_modes, 2)
        
        # Power Prediction
        powers = self.power_head(features)
        
        return symbols, powers

if __name__ == "__main__":
    # Test the model
    model = MultiHeadResNet(n_modes=8)
    x = torch.randn(4, 1, 64, 64)
    sym, pwr = model(x)
    print(f"Input: {x.shape}")
    print(f"Symbols Output: {sym.shape}") # Should be [4, 8, 2]
    print(f"Power Output: {pwr.shape}")   # Should be [4, 8]
