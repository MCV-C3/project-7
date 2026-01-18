import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from typing import List


class OptimizedCNN(nn.Module):
    """
    Optimized CNN architecture after architecture search + adaptive pooling experiments.
    
    Key design decisions validated through systematic experiments:
    - Narrow channels [16,32,64,128]: Reduces conv overfitting (from arch search)
    - Global Average Pooling (1×1): Eliminates spatial redundancy, reduces FC params
    - Direct classification: No hidden FC layer (128 → 8), minimal overfitting
    
    Performance: 75.87% test acc, 2.88% train-test gap, 98,952 total params
    Baseline comparison: Original SimpleCNN had 28.58% gap with 6.8M params
    """
    
    def __init__(self, num_classes: int = 8, input_channels: int = 3, dropout: float = 0.3):
        """
        Initialize OptimizedCNN with validated architecture.
        
        Args:
            num_classes (int): Number of output classes (default: 8 for MIT scenes)
            input_channels (int): Number of input channels (3 for RGB images)
            dropout (float): Dropout probability for regularization (default: 0.3)
        """
        super(OptimizedCNN, self).__init__()
        
        # Store activation
        self.relu = nn.ReLU()
        
        # Convolutional Block 1: 3 -> 16 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        # Convolutional Block 2: 16 -> 32 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        # Convolutional Block 3: 32 -> 64 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        # Convolutional Block 4: 64 -> 128 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )
        
        # Global Average Pooling (1×1)
        # Reduces 128×14×14 = 25,088 features → 128×1×1 = 128 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Direct classification: 128 → 8 (no hidden layer)
        # Parameters: 128×8 + 8 bias = 1,032 params (vs 3.2M in original baseline)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global Average Pooling
        x = self.adaptive_pool(x)  # (batch, 128, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128)
        
        # Direct classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    Simple CNN built from scratch for image classification.
    Architecture:
        - 4 Convolutional blocks with increasing channels
        - Each block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        - Adaptive pooling to handle variable input sizes
        - Fully connected classifier
    """
    
    def __init__(self, num_classes: int = 8, input_channels: int = 3, dropout: float = 0.3):
        """
        Initialize the SimpleCNN model.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB images)
            dropout (float): Dropout probability for regularization
        """
        super(SimpleCNN, self).__init__()
        
        # Store activation for consistency
        self.relu = nn.ReLU()
        
        # Convolutional Block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        # Convolutional Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        # Convolutional Block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        # Convolutional Block 4: 128 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )
        
        # Adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class FlexibleCNN(nn.Module):
    """
    Flexible CNN for architecture search experiments.
    Configurable depth (number of blocks) and width (channel progression).
    Maintains same design patterns as SimpleCNN for fair comparison.
    """
    
    def __init__(
        self, 
        num_classes: int = 8,
        input_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
        pooling_type: str = 'max',
        fc_hidden: int = 512,
        dropout: float = 0.3,
        pool_output_size: tuple = (7, 7),
        use_fc_hidden: bool = True,
        adaptive_pool_type: str = 'avg'
    ):
        """
        Initialize FlexibleCNN with configurable architecture.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB)
            channels (List[int]): Channel progression for each conv block
            kernel_size (int): Kernel size for all conv layers (default: 3)
            pooling_type (str): 'max' for MaxPool2d, 'strided_conv' for strided convolution (inside conv blocks)
            fc_hidden (int): Hidden units in FC layer (default: 512)
            dropout (float): Dropout probability
            pool_output_size (tuple): Output size for adaptive pooling (default: (7, 7), use (1, 1) for GAP)
            use_fc_hidden (bool): If True, use hidden FC layer; if False, direct classification
            adaptive_pool_type (str): 'avg' for AdaptiveAvgPool2d, 'max' for AdaptiveMaxPool2d (default: 'avg')
        """
        super(FlexibleCNN, self).__init__()
        
        self.num_blocks = len(channels)
        self.channels = channels
        self.relu = nn.ReLU()
        self.use_fc_hidden = use_fc_hidden
        self.pool_output_size = pool_output_size
        
        # Build convolutional blocks dynamically
        self.blocks = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in channels:
            block = self._make_conv_block(
                in_channels, 
                out_channels, 
                kernel_size, 
                pooling_type
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        # Adaptive pooling to get fixed size output
        if adaptive_pool_type == 'max':
            self.adaptive_pool = nn.AdaptiveMaxPool2d(pool_output_size)
        else:  # default to avg
            self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_output_size)
        
        # Calculate flattened feature size
        flattened_size = channels[-1] * pool_output_size[0] * pool_output_size[1]
        
        # Fully connected layers
        if use_fc_hidden:
            self.fc1 = nn.Linear(flattened_size, fc_hidden)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(fc_hidden, num_classes)
        else:
            # Direct classification (no hidden layer)
            self.fc1 = None
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(flattened_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                         kernel_size: int, pooling_type: str) -> nn.Sequential:
        """Create a convolutional block with Conv -> BN -> ReLU -> Pool."""
        padding = kernel_size // 2  # Maintain spatial dimensions
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        
        # Add pooling layer
        if pooling_type == 'max':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif pooling_type == 'strided_conv':
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Pass through all convolutional blocks
        for block in self.blocks:
            x = block(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        if self.use_fc_hidden:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            # Direct classification
            x = self.dropout(x)
            x = self.fc2(x)
        
        return x


def build_transforms(train: bool = True) -> transforms.Compose:
    """
    Build image transformations for training or testing.
    
    Args:
        train (bool): If True, return training transforms, else return test transforms
        
    Returns:
        transforms.Compose: Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((224, 224)),
        ])
    else:
        return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((224, 224)),
        ])


