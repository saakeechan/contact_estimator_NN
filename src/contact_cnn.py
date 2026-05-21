import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

"""
Two network architectures are available:

1. TCN (Temporal Convolutional Network):
   - Uses weight normalization (not batch norm)
   - Exponentially increasing dilation rates: 2^0, 2^1, 2^2, ...
   - Residual connections with optional 1x1 convolution
   - Dropout for regularization
   - Configurable: num_channels, kernel_size, num_blocks, dropout

2. contact_cnn (Original):
   - Simple dilated CNN architecture
   - Fixed dilation pattern: 1, 2, 4, 8
   - Uses LeakyReLU activation
   - No residual connections
   - Fewer parameters, faster training

Both architectures:
- Support causal convolutions (no future information leakage)
- Output velocity predictions for all timesteps (dense supervision)
- Extract last timestep for inference
"""


class TCNResidualBlock(nn.Module):
    """
    Residual block for TCN as shown in Figure 2.
    Contains two causal convolution layers with weight normalization, dropout, and skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        
        # First causal conv layer (with weight normalization)
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second causal conv layer (with weight normalization)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 1x1 convolution for residual connection if dimensions don't match
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        # Save input for residual connection
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Apply 1x1 conv to residual if needed
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Add residual connection
        out = out + residual
        out = self.relu_out(out)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network with residual blocks as shown in Figure 2.
    Uses exponentially increasing dilation rates: 2^0, 2^1, 2^2, ..., 2^(num_blocks-1)
    """
    def __init__(self, window_size=10, num_features=12, num_channels=64, kernel_size=3, num_blocks=5, dropout=0.2):
        super(TCN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        layers = []
        num_levels = num_blocks
        
        for i in range(num_levels):
            dilation_rate = 2 ** i
            in_channels = num_features if i == 0 else num_channels
            out_channels = num_channels
            
            layers.append(TCNResidualBlock(
                in_channels, out_channels, kernel_size, dilation_rate, dropout
            ))
        
        self.tcn_backbone = nn.Sequential(*layers)
        
        # Velocity prediction head (sequence-to-sequence)
        self.velocity_head = nn.Sequential(
            nn.Conv1d(num_channels, 1, kernel_size=1),
            nn.Softplus()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, num_features) - RAW features
        
        Returns:
            velocity_seq: (batch_size, 1, window_size) - velocity predictions for all timesteps
            velocity_out: (batch_size, 1) - velocity prediction at last timestep only
        """
        # Permute to (batch_size, num_features, window_size) for Conv1d
        x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        
        # Pass through TCN backbone
        features = self.tcn_backbone(x)  # [B, num_channels, T]
        
        # Velocity prediction
        velocity_seq = self.velocity_head(features)  # [B, 1, T]
        velocity_out = velocity_seq[:, :, -1]  # [B, 1] - last timestep
        
        return velocity_seq, velocity_out


class CausalConv1d(nn.Module):
    """Causal convolution with weight normalization."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        ))

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # pad only left
        return self.conv(x)

class contact_cnn(nn.Module):
    def __init__(self, window_size=10, num_features=12):
        super(contact_cnn, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # Shared convolutional backbone
        # Preserves temporal dimension throughout - no MaxPool downsampling
        # Input: [B, num_features, T]
        # Uses dilated convolutions to capture multi-scale temporal patterns
        
        self.conv_backbone = nn.Sequential(
            # Block 1 (causal)
            CausalConv1d(
                in_ch=num_features,
                out_ch=128,
                kernel_size=3,
                dilation=1
            ),
            nn.LeakyReLU(0.01),

            # Block 2 (causal)
            CausalConv1d(
                in_ch=128,
                out_ch=128,
                kernel_size=3,
                dilation=2
            ),
            nn.LeakyReLU(0.01),

            # nn.Dropout(p=0.05),

            # Block 3 (causal, more dilated)
            CausalConv1d(
                in_ch=128,
                out_ch=64,
                kernel_size=3,
                dilation=4
            ),
            nn.LeakyReLU(0.01),

            # Block 4 (causal)
            CausalConv1d(
                in_ch=64,
                out_ch=64,
                kernel_size=3,
                dilation=8
            ),
            nn.LeakyReLU(0.01),

            # nn.Dropout(p=0.05),
        )
        
        # Branch 1: Velocity prediction (sequence-to-sequence with Conv1d)
        # Predicts velocity for all timesteps, but only uses last one at inference
        # Input: [B, 128, T] from conv_backbone
        # Output: [B, 1, T] - velocity for each timestep (x-axis only)
        self.velocity_head = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Softplus()
)

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features) - RAW features from csv2numpy.py
        
        # Permute to (batch_size, num_features, window_size) for Conv1d
        x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        
        # Pass through shared convolutional backbone
        features = self.conv_backbone(x)  # [B, 128, T]
        
        # Branch 1: Velocity prediction (sequence-to-sequence)
        # Predict velocity for all timesteps
        velocity_seq = self.velocity_head(features)  # [B, 1, T]
        
        # Extract last timestep for final output (used at inference)
        velocity_out = velocity_seq[:, :, -1]  # [B, 1]
        
        # # Branch 2: Contact classification (single value at last timestep)
        # contact_out = self.contact_head(features)  # [B, 1]
        
        # return contact_out, velocity_out

        # Return both sequence (for training) and last timestep (for inference)
        # During training: use velocity_seq for dense supervision
        # During inference: use velocity_out (last timestep only)
        return velocity_seq, velocity_out


class ContactCNNWithNormalization(nn.Module):
    """
    Wrapper class that embeds global normalization statistics into the model.
    This wrapper can be exported to ONNX so normalization is done inside the model
    during inference, eliminating the need to normalize data externally in C++.
    
    Normalization strategy: Global z-score normalization
    - Uses mean and std computed from training data (stored as buffers)
    - Normalize: (x - global_mean) / (global_std + eps)
    - Statistics are saved with the model and exported to ONNX
    
    Input shape: (batch_size, window_size, num_features) - RAW features from csv2numpy.py (LEFT LEG ONLY)
    Output shapes:
        - velocity_seq: (batch_size, 1, window_size) - velocity predictions for all timesteps (for training)
        - velocity_out: (batch_size, 1) - velocity prediction at last timestep only (for inference)
    
    Feature layout:
    - Input: RAW features from csv2numpy.py
    - Z-score normalization is applied to all input features
    """
    def __init__(self, base_model, global_mean=None, global_std=None, eps=1e-8):
        """
        Args:
            base_model: The contact_cnn model to wrap
            global_mean: Tensor of shape (1, 1, num_features) with training data mean per feature
            global_std: Tensor of shape (1, 1, num_features) with training data std per feature
            eps: Small constant to prevent division by zero
        """
        super(ContactCNNWithNormalization, self).__init__()
        self.base_model = base_model
        self.eps = eps
        
        # Get num_features from base model
        num_features = base_model.num_features
        
        # Register as buffers (not parameters - won't be trained, but saved with model and exported to ONNX)
        if global_mean is not None:
            self.register_buffer('global_mean', global_mean)
        else:
            # Fallback: no normalization if stats not provided
            self.register_buffer('global_mean', torch.zeros(1, 1, num_features))
            
        if global_std is not None:
            self.register_buffer('global_std', global_std)
        else:
            # Fallback: no normalization if stats not provided
            self.register_buffer('global_std', torch.ones(1, 1, num_features))
        
    def forward(self, x):
        """
        Apply z-score normalization, then pass through base model.
        
        Args:
            x: Raw input data (batch_size, window_size, num_features) - NOT z-score normalized (LEFT LEG ONLY)
               Features: q[3,4](2) + p[x,z](2) + v[x,z](2) + tau_est[1-5](5) + tau_mse(1) = 12 features
        
        Returns:
            velocity_seq: (batch_size, 1, window_size) - velocity predictions for all timesteps (for training)
            velocity_out: (batch_size, 1) - velocity prediction at last timestep only (for inference)
        """
        # Apply global z-score normalization to all input features
        # These statistics are embedded in the model and exported to ONNX
        x_normalized = (x - self.global_mean) / (self.global_std + self.eps)
        
        # Pass normalized data through the base model
        return self.base_model(x_normalized)

