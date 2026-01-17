import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms


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


