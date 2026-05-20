import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )

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
            nn.ReLU(),

            # Block 2 (causal)
            CausalConv1d(
                in_ch=128,
                out_ch=128,
                kernel_size=3,
                dilation=1
            ),
            nn.ReLU(),

            # nn.Dropout(p=0.05),

            # # Block 3 (causal, more dilated)
            # CausalConv1d(
            #     in_ch=128,
            #     out_ch=128,
            #     kernel_size=3,
            #     dilation=2
            # ),
            # nn.ReLU(),

            # Block 4 (causal)
            CausalConv1d(
                in_ch=128,
                out_ch=64,
                kernel_size=3,
                dilation=1
            ),
            nn.ReLU(),

            # nn.Dropout(p=0.05),
        )
        
        # Branch 1: Velocity prediction (MLP on last timestep)
        self.velocity_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # # Branch 2: Contact classification (MLP, single prediction)
        # # Global average pooling over time dimension
        # self.contact_head = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),  # [B, 128, T] → [B, 128, 1]
        #     nn.Flatten(),              # [B, 128, 1] → [B, 128]
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(64, 1)           # Binary contact prediction
        # )

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features) - RAW features from csv2numpy.py
        # Permute to (batch_size, num_features, window_size) for Conv1d
        x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        
        # Pass through shared convolutional backbone
        features = self.conv_backbone(x)  # [B, 64, T]
        
        # Branch 1: Velocity prediction (last timestep only)
        v_last = features[:, :, -1]  # Extract last timestep: [B, 64, T] → [B, 64]
        velocity_out = self.velocity_head(v_last)  # [B, 1] - single velocity prediction
        
        # # Flattened version (commented out):
        # features_flat = features.flatten(start_dim=1)  # [B, 64, T] → [B, 64*T]
        # velocity_out = self.velocity_head(features_flat)  # [B, 1] - single velocity prediction
        
        # # Branch 2: Contact classification (single value at last timestep)
        # contact_out = self.contact_head(features)  # [B, 1]
        
        # return contact_out, velocity_out

        return velocity_out  # Only velocity output for now, contact head is commented out


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
        - contact: (batch_size, 1) - binary contact prediction (last timestep)
        - velocity: (batch_size, 1) - velocity prediction at last timestep only
    
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
            contact_out: (batch_size, 1) - binary contact prediction (last timestep)
            velocity_out: (batch_size, 1) - velocity prediction at last timestep only
        """
        # Apply global z-score normalization to all input features
        # These statistics are embedded in the model and exported to ONNX
        x_normalized = (x - self.global_mean) / (self.global_std + self.eps)
        
        # Pass normalized data through the base model
        return self.base_model(x_normalized)

