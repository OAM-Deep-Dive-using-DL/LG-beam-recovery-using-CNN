"""
ResNet-18 architecture for OAM symbol regression.

Modified from torchvision ResNet to output continuous values (symbol regression)
instead of classification logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Receiver(nn.Module):
    """
    ResNet-18 for OAM symbol regression.
    
    Args:
        input_channels (int): Number of input channels (1 for intensity, 2 for complex field)
        output_size (int): Number of output values (8 modes × 2 for I/Q = 16)
        dropout (float): Dropout probability for final FC layers
    """
    
    def __init__(self, input_channels=1, output_size=16, dropout=0.3):
        super(ResNet18Receiver, self).__init__()
        
        self.input_channels = input_channels
        self.output_size = output_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer with multiple blocks."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Symbol estimates of shape (batch, output_size)
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict_symbols(self, x):
        """
        Predict QPSK symbols from input.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            symbols: Complex symbols of shape (batch, num_modes)
        """
        # Get raw predictions
        predictions = self.forward(x)
        
        # Reshape to (batch, num_modes, 2) for I/Q
        num_modes = self.output_size // 2
        predictions = predictions.view(-1, num_modes, 2)
        
        # Convert to complex symbols
        symbols = predictions[..., 0] + 1j * predictions[..., 1]
        
        return symbols


def create_model(config):
    """
    Create ResNet-18 model from configuration.
    
    Args:
        config (dict): Model configuration
    
    Returns:
        model: ResNet18Receiver instance
    """
    model = ResNet18Receiver(
        input_channels=config.get('input_channels', 1),
        output_size=config.get('output_size', 16),
        dropout=config.get('dropout', 0.3)
    )
    return model


if __name__ == "__main__":
    # Test model
    print("Testing ResNet-18 Receiver...")
    
    model = ResNet18Receiver(input_channels=1, output_size=16, dropout=0.3)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 64)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test symbol prediction
    symbols = model.predict_symbols(x)
    print(f"Symbols shape: {symbols.shape}")
    print(f"Sample symbols: {symbols[0]}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    print("\n✓ Model test passed!")
