"""
CosmicDreams: Core Neural Network Modules

Contains the ResidualBlock3D building block used in ResNet-Rollout architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block for spatiotemporal processing.
    
    Architecture:
        Conv3d(in_channels, out_channels, 3×3×3)
        → GroupNorm → SiLU 
        → Conv3d(out_channels, out_channels, 3×3×3)
        → GroupNorm → SiLU
        → Residual connection
    
    Args:
        channels: Number of input/output channels
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Output tensor [B, C, T, H, W]
        """
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        return x + residual

