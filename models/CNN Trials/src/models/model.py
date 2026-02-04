import torch
import torch.nn as nn
import torchvision.models as models
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class ConvNeXtAdapter(nn.Module):
    """
    Adapter for ConvNeXt to work with 1-channel input and FSO-OAM task.
    """
    def __init__(self, model_name='convnext_tiny', input_channels=1, pretrained=True):
        super().__init__()
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if model_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights=weights)
            self.num_features = 768
        elif model_name == 'convnext_small':
            self.backbone = models.convnext_small(weights=weights)
            self.num_features = 768
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {model_name}")
            
        # Modify Stem for 1-channel Input
        # Original: Conv2d(3, 96, kernel_size=4, stride=4)
        # We keep the stride=4 to downsample 128x128 -> 32x32 feature map quickly
        original_stem = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            input_channels, 
            original_stem.out_channels, 
            kernel_size=original_stem.kernel_size, 
            stride=original_stem.stride
        )
        
        # [Concept Fix] Smart Initialization for Transfer Learning
        # Instead of random initialization, we average the pretrained RGB weights.
        # This preserves edge/texture detection capabilities.
        with torch.no_grad():
            # original_stem.weight shape: [96, 3, 4, 4]
            # We want [96, 1, 4, 4]
            # Average across the channel dimension (dim=1)
            avg_weight = torch.mean(original_stem.weight, dim=1, keepdim=True)
            self.backbone.features[0][0].weight.copy_(avg_weight)
            # Bias is already correct (out_channels matches)
            if original_stem.bias is not None:
                self.backbone.features[0][0].bias.copy_(original_stem.bias)
        
        # Remove Classifier
        # We only want features. 
        # torchvision ConvNeXt: features -> avgpool -> classifier
        # We will expose a method to get features directly or just replace classifier with Identity
        self.backbone.classifier = nn.Identity()
        
    def forward(self, x):
        # Forward pass through backbone
        # features: [N, 768, H, W] -> avgpool -> [N, 768, 1, 1] -> classifier(Identity) -> [N, 768, 1, 1]
        x = self.backbone(x)
        return torch.flatten(x, 1) # [N, 768]

class EfficientNetAdapter(nn.Module):
    """
    Adapter for EfficientNet (B0/V2) to work with 1-channel input.
    """
    def __init__(self, model_name='efficientnet_b0', input_channels=1, pretrained=True):
        super().__init__()
        
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=weights)
            self.num_features = 1280
        elif model_name == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights=weights)
            self.num_features = 1280
        else:
            raise ValueError(f"Unknown EfficientNet variant: {model_name}")
            
        # Modify first layer
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
    def forward(self, x):
        # features -> avgpool -> classifier(Identity)
        x = self.backbone(x) 
        return torch.flatten(x, 1)

class FSOModel(nn.Module):
    """
    Unified FSO-OAM Receiver Model.
    
    Architecture:
    1. Backbone: ConvNeXt or EfficientNet
    2. Head: Symbol Regression (Complex QPSK prediction)
    """
    def __init__(self, n_modes=8, input_channels=1, backbone_name='convnext_tiny', dropout_rate=0.3):
        super(FSOModel, self).__init__()
        
        # Backbone Adapter
        if 'convnext' in backbone_name:
            self.backbone = ConvNeXtAdapter(backbone_name, input_channels)
            num_features = self.backbone.num_features
        elif 'efficientnet' in backbone_name:
            self.backbone = EfficientNetAdapter(backbone_name, input_channels)
            num_features = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Symbol Head (Regression)
        # 768 -> 1024 -> 512 -> n_modes*2
        self.symbol_head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.01), # LeakyReLU often better for regression
            nn.Dropout(dropout_rate), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, n_modes * 2) 
        )
        
        self.n_modes = n_modes

    def forward(self, x):
        # Extract Features
        features = self.backbone(x) # [N, 768]
        
        # Regress Symbols
        symbols_flat = self.symbol_head(features)
        
        # Reshape: [N, n_modes, 2]
        symbols = symbols_flat.view(-1, self.n_modes, 2)
        
        # Legacy return signature (symbols, power=None)
        return symbols, None

if __name__ == "__main__":
    # Sanity Check
    print("Testing FSOModel (ConvNeXt-Tiny & EfficientNet-B0)...")
    
    # Test ConvNeXt
    print("1. ConvNeXt-Tiny")
    model = FSOModel(n_modes=8, backbone_name='convnext_tiny')
    x = torch.randn(4, 1, 128, 128)
    sym, _ = model(x)
    print(f"Output: {sym.shape}")
    
    # Test EfficientNet
    print("2. EfficientNet-B0")
    model = FSOModel(n_modes=8, backbone_name='efficientnet_b0')
    sym, _ = model(x)
    print(f"Output: {sym.shape}")
    
    print("✓ Sanity Check Passed")
