import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.
    
    Uses both average and max pooling for richer feature aggregation.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Initialize Channel Attention.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Reduction ratio for bottleneck (default: 4)
        """
        super(ChannelAttention, self).__init__()
        
        # Shared MLP for both avg and max pooled features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Use 1x1 convolutions instead of Linear (paper-accurate, avoids reshaping)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling branch (B, C, 1, 1)
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling branch (B, C, 1, 1)
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid (stays in conv format)
        out = self.sigmoid(avg_out + max_out)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.
    
    Focuses on 'where' is important by aggregating channel information.
    """
    
    def __init__(self, kernel_size: int = 7, dilation: int = 1):
        """
        Initialize Spatial Attention.
        
        Args:
            kernel_size (int): Kernel size for conv layer (default: 7)
            dilation (int): Dilation rate for convolution (default: 1)
        """
        super(SpatialAttention, self).__init__()
        
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Aggregate channel information
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        out = self.sigmoid(self.conv(out))  # (B, 1, H, W)
        
        return x * out


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies channel and spatial attention:
    1. Channel Attention: Focuses on 'what' is meaningful
    2. Spatial Attention: Focuses on 'where' is meaningful
    
    Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """
    
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7, dilation: int = 1):
        """
        Initialize CBAM block.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Reduction ratio for channel attention (default: 4)
            kernel_size (int): Kernel size for spatial attention (default: 7)
            dilation (int): Dilation rate for spatial attention (default: 1)
        """
        super(CBAMBlock, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size, dilation)
    
    def forward(self, x):
        """
        Forward pass through CBAM block.
        
        Args:
            x (torch.Tensor): Input features of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention-refined features of shape (B, C, H, W)
        """
        # Channel attention first
        x = self.channel_attention(x)
        
        # Then spatial attention
        x = self.spatial_attention(x)
        
        return x


class CBAMOptimizedCNN(nn.Module):
    """
    Optimized CNN with CBAM (Convolutional Block Attention Module).
    
    Extends OptimizedCNN by adding CBAM blocks for both channel and spatial attention.
    CBAM is placed BEFORE pooling to preserve spatial information for spatial attention.
    
    Architecture:
    - Base: [16,32,64,128] channels + GAP + direct classification
    - Addition: CBAM blocks (channel + spatial attention) before pooling in each block
    
    """
    
    def __init__(self, num_classes: int = 8, input_channels: int = 3, 
                 dropout: float = 0.3, reduction: int = 4,
                 spatial_kernel: int = 7, spatial_dilation: int = 1,
                 num_cbam_blocks: int = 4):
        """
        Initialize CBAMOptimizedCNN with dual attention.
        
        Args:
            num_classes (int): Number of output classes (default: 8 for MIT scenes)
            input_channels (int): Number of input channels (3 for RGB images)
            dropout (float): Dropout probability for regularization (default: 0.3)
            reduction (int): Channel attention reduction ratio (default: 4)
            spatial_kernel (int): Kernel size for spatial attention (default: 7)
        """
        super(CBAMOptimizedCNN, self).__init__()

        # Use cbam
        assert 0 <= num_cbam_blocks <= 4 and type(num_cbam_blocks) == int
        self.use_cbam = [
            num_cbam_blocks >= 4,
            num_cbam_blocks >= 3,
            num_cbam_blocks >= 2,
            num_cbam_blocks >= 1,
        ]
        
        # Convolutional Block 1: 3 -> 16 channels (with CBAM before pooling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.cbam1 = CBAMBlock(16, reduction=reduction, kernel_size=spatial_kernel, dilation=spatial_dilation)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        
        # Convolutional Block 2: 16 -> 32 channels (with CBAM before pooling)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.cbam2 = CBAMBlock(32, reduction=reduction, kernel_size=spatial_kernel, dilation=spatial_dilation)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        
        # Convolutional Block 3: 32 -> 64 channels (with CBAM before pooling)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cbam3 = CBAMBlock(64, reduction=reduction, kernel_size=spatial_kernel, dilation=spatial_dilation)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        
        # Convolutional Block 4: 64 -> 128 channels (with CBAM before pooling)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.cbam4 = CBAMBlock(128, reduction=reduction, kernel_size=spatial_kernel, dilation=spatial_dilation)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        
        # Global Average Pooling (1×1)
        # Reduces 128×14×14 = 25,088 features → 128×1×1 = 128 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Direct classification: 128 → 8 (no hidden layer)
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network with CBAM attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Block 1: Conv -> CBAM -> Pool
        x = self.conv1(x)
        if self.use_cbam[0]:
            x = self.cbam1(x)  # Dual attention before pooling
        x = self.pool1(x)
        
        # Block 2: Conv -> CBAM -> Pool
        x = self.conv2(x)
        if self.use_cbam[1]:
            x = self.cbam2(x)  # Dual attention before pooling
        x = self.pool2(x)
        
        # Block 3: Conv -> CBAM -> Pool
        x = self.conv3(x)
        if self.use_cbam[2]:
            x = self.cbam3(x)  # Dual attention before pooling
        x = self.pool3(x)
        
        # Block 4: Conv -> CBAM -> Pool
        x = self.conv4(x)
        if self.use_cbam[3]:
            x = self.cbam4(x)  # Dual attention before pooling
        x = self.pool4(x)
        
        # Global Average Pooling
        x = self.adaptive_pool(x)  # (batch, 128, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128)
        
        # Direct classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
