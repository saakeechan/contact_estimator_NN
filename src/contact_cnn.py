import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

"""
Three network architectures are available:

1. AttentionTCN (Hybrid Transformer + TCN):
   - Multi-head causal self-attention for long-range dependencies
   - TCN backbone with residual blocks for local temporal features
   - Weight normalization (not batch norm)
   - Exponentially increasing dilation rates: 2^0, 2^1, 2^2, ...
   - Learnable attention blending (gamma parameter)
   - Most expressive but slowest
   - Configurable: d_model, num_heads, tcn_num_channels, tcn_kernel_size, tcn_num_blocks, tcn_dropout

2. TCN (Pure Temporal Convolutional Network):
   - Residual blocks with exponentially increasing dilation
   - Weight normalization (not batch norm)
   - Exponentially increasing dilation rates: 2^0, 2^1, 2^2, ...
   - Good balance between expressiveness and speed
   - Configurable: tcn_num_channels, tcn_kernel_size, tcn_num_blocks, tcn_dropout

3. contact_cnn (Vanilla CNN):
   - Inception-style multi-scale architecture
   - Parallel branches with kernel sizes 3, 5, 7 (captures patterns at different temporal scales)
   - Concatenates multi-scale features
   - Uses GELU activation
   - No residual connections
   - Fastest, fewest parameters

All architectures:
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


class AttentionTCN(nn.Module):
    """
    Transformer + TCN hybrid architecture with causal attention.
    Adds multi-head self-attention before TCN backbone to capture long-range dependencies.
    
    Architecture:
    1. Input projection: map raw features to d_model dimension
    2. Layer normalization + causal multi-head self-attention: capture global temporal context
    3. Residual fusion with learnable gamma (initialized to 0.0)
    4. Layer normalization before TCN
    5. TCN backbone: local temporal feature extraction
    6. Velocity prediction head
    
    Key design choices:
    - Causal attention mask: prevents looking into the future (critical for real-time inference)
    - Layer normalization: stabilizes training (standard in transformers)
    - gamma starts at 0.0: model initially behaves like vanilla TCN, gradually learns to use attention
    """
    def __init__(self, window_size=10, num_features=12, d_model=64, num_heads=4, 
                 tcn_num_channels=64, tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=0.2):
        super(AttentionTCN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # 1. Project raw features into transformer dimension
        self.input_proj = nn.Linear(num_features, d_model)
        
        # 2. Layer normalization (pre-norm style for stable training)
        self.norm1 = nn.LayerNorm(d_model)  # Before attention
        self.norm2 = nn.LayerNorm(d_model)  # Before TCN
        
        # 3. Multi-head self-attention (with causal masking)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True  # Input: [B, T, D]
        )
        
        # 4. Learnable residual weight (starts at 0.0 for stable training)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
        # 5. TCN backbone (same as original TCN but takes d_model as input)
        tcn_layers = []
        for i in range(tcn_num_blocks):
            dilation_rate = 2 ** i
            in_channels = d_model if i == 0 else tcn_num_channels
            out_channels = tcn_num_channels
            
            tcn_layers.append(TCNResidualBlock(
                in_channels, out_channels, tcn_kernel_size, dilation_rate, tcn_dropout
            ))
        
        self.tcn_backbone = nn.Sequential(*tcn_layers)
        
        # 6. Velocity prediction head
        self.velocity_head = nn.Sequential(
            nn.Conv1d(tcn_num_channels, 1, kernel_size=1),
            # nn.GELU()
        )
        
        # 7. Contact detection head (MLP: 256 → 32 → 1)
        self.contact_head = nn.Sequential(
            nn.Linear(tcn_num_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No activation - BCEWithLogitsLoss applies sigmoid internally
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, num_features) - RAW features
        
        Returns:
            velocity_seq: (batch_size, 1, window_size) - velocity in m/s for all timesteps
            velocity_out: (batch_size, 1) - velocity in m/s at last timestep only
            contact_out: (batch_size, 1) - contact logits at last timestep
        """
        # x: [B, T, F]
        
        # 1. Project to d_model dimension
        z = self.input_proj(x)  # [B, T, D]
        
        B, T, D = z.shape
        
        # 2. Create causal mask (prevents attending to future timesteps)
        # Upper triangular matrix with diagonal=1 blocks future positions
        causal_mask = torch.triu(
            torch.ones(T, T, device=z.device, dtype=torch.bool),
            diagonal=1
        )  # [T, T]
        
        # 3. Pre-normalization + Multi-head self-attention with causal masking
        z_normed = self.norm1(z)  # [B, T, D]
        attn_out, _ = self.mha(
            z_normed, z_normed, z_normed,
            attn_mask=causal_mask
        )  # [B, T, D]
        
        # 4. Residual fusion with learnable gamma
        # gamma=0.0 initially → starts as identity (z = z + 0*attn_out = z)
        # gamma learns during training to blend in attention as needed
        z = z + self.gamma * attn_out  # [B, T, D]
        
        # 5. Pre-normalization before TCN
        z = self.norm2(z)  # [B, T, D]
        
        # 6. Convert to Conv1d format for TCN
        z = z.permute(0, 2, 1)  # [B, T, D] → [B, D, T]
        
        # 7. TCN backbone
        features = self.tcn_backbone(z)  # [B, tcn_num_channels, T]
        
        # 8. Velocity prediction
        velocity_seq = self.velocity_head(features)  # [B, 1, T]
        velocity_out = velocity_seq[:, :, -1]  # [B, 1] - last timestep
        
        # 9. Contact prediction (MLP on last timestep features)
        features_last = features[:, :, -1]  # [B, tcn_num_channels]
        contact_out = self.contact_head(features_last)  # [B, 1]
        
        return velocity_seq, velocity_out, contact_out


class TCN(nn.Module):
    """
    Pure Temporal Convolutional Network (TCN) without attention.
    Uses residual blocks with exponentially increasing dilation rates.
    
    Architecture:
    1. Input projection: map raw features to tcn_num_channels dimension
    2. TCN backbone: stacked residual blocks with increasing dilation
    3. Velocity prediction head
    
    Key design choices:
    - Exponentially increasing dilation: 2^0, 2^1, 2^2, ... (receptive field grows exponentially)
    - Residual connections: enable training of deep networks
    - Weight normalization: stabilizes training without batch statistics
    - Causal convolutions: no future information leakage
    """
    def __init__(self, window_size=10, num_features=12, tcn_num_channels=64, 
                 tcn_kernel_size=3, tcn_num_blocks=5, tcn_dropout=0.2):
        super(TCN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # 1. Project raw features into TCN channel dimension
        self.input_proj = nn.Conv1d(num_features, tcn_num_channels, kernel_size=1)
        
        # 2. TCN backbone with residual blocks
        tcn_layers = []
        for i in range(tcn_num_blocks):
            dilation_rate = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, ...
            
            tcn_layers.append(TCNResidualBlock(
                tcn_num_channels, tcn_num_channels, 
                tcn_kernel_size, dilation_rate, tcn_dropout
            ))
        
        self.tcn_backbone = nn.Sequential(*tcn_layers)
        
        # 3. Velocity prediction head
        self.velocity_head = nn.Sequential(
            nn.Conv1d(tcn_num_channels, 1, kernel_size=1),
        )
        
        # 4. Contact detection head (MLP: 256 → 32 → 1)
        self.contact_head = nn.Sequential(
            nn.Linear(tcn_num_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No activation - BCEWithLogitsLoss applies sigmoid internally
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, num_features) - RAW features
        
        Returns:
            velocity_seq: (batch_size, 1, window_size) - velocity in m/s for all timesteps
            velocity_out: (batch_size, 1) - velocity in m/s at last timestep only
            contact_out: (batch_size, 1) - contact logits at last timestep
        """
        # x: [B, T, F]
        
        # 1. Convert to Conv1d format
        x = x.permute(0, 2, 1)  # [B, T, F] → [B, F, T]
        
        # 2. Project to TCN channel dimension
        z = self.input_proj(x)  # [B, F, T] → [B, tcn_num_channels, T]
        
        # 3. TCN backbone
        features = self.tcn_backbone(z)  # [B, tcn_num_channels, T]
        
        # 4. Velocity prediction
        velocity_seq = self.velocity_head(features)  # [B, 1, T]
        velocity_out = velocity_seq[:, :, -1]  # [B, 1] - last timestep
        
        # 5. Contact prediction (MLP on last timestep features)
        features_last = features[:, :, -1]  # [B, tcn_num_channels]
        contact_out = self.contact_head(features_last)  # [B, 1]
        
        return velocity_seq, velocity_out, contact_out


class CausalConv1d(nn.Module):
    """Causal convolution with weight normalization."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # pad only left
        return self.conv(x)

class contact_cnn(nn.Module):
    def __init__(self, window_size=10, num_features=12):
        super(contact_cnn, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        
        # Initial convolutional layer
        self.conv_initial = nn.Sequential(
            CausalConv1d(
                in_ch=num_features,
                out_ch=64,
                kernel_size=3,
                dilation=1
            ),
            nn.GELU(),
        )
        
        # Parallel multi-scale branches with different kernel sizes (Inception-style)
        # Each branch captures patterns at different temporal scales
        self.branch_k3 = nn.Sequential(
            CausalConv1d(in_ch=64, out_ch=32, kernel_size=3, dilation=1),
            nn.GELU(),
        )
        self.branch_k5 = nn.Sequential(
            CausalConv1d(in_ch=64, out_ch=32, kernel_size=5, dilation=1),
            nn.GELU(),
        )
        self.branch_k7 = nn.Sequential(
            CausalConv1d(in_ch=64, out_ch=32, kernel_size=7, dilation=1),
            nn.GELU(),
        )
        
        # After concatenation: 32 + 32 + 32 = 96 channels
        # Final processing layers
        self.conv_post = nn.Sequential(
            CausalConv1d(
                in_ch=96,  # Concatenated from 3 branches
                out_ch=64,
                kernel_size=3,
                dilation=2
            ),
            nn.GELU(),
        )
        
        # Velocity prediction head
        self.velocity_head = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
        )
        
        # Contact detection head (MLP: 256 → 32 → 1)
        self.contact_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No activation - BCEWithLogitsLoss applies sigmoid internally
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features) - RAW features from csv2numpy.py
        
        # Permute to (batch_size, num_features, window_size) for Conv1d
        x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        
        # Initial convolution
        x = self.conv_initial(x)  # [B, 64, T]
        
        # Parallel multi-scale branches
        branch_3 = self.branch_k3(x)  # [B, 32, T]
        branch_5 = self.branch_k5(x)  # [B, 32, T]
        branch_7 = self.branch_k7(x)  # [B, 32, T]
        
        # Concatenate branches along channel dimension
        features = torch.cat([branch_3, branch_5, branch_7], dim=1)  # [B, 96, T]
        
        # Final processing
        features = self.conv_post(features)  # [B, 64, T]
        
        # Velocity prediction (sequence-to-sequence)
        velocity_seq = self.velocity_head(features)  # [B, 1, T]
        
        # Extract last timestep for final output (used at inference)
        velocity_out = velocity_seq[:, :, -1]  # [B, 1]
        
        # Contact prediction (MLP on last timestep features)
        features_last = features[:, :, -1]  # [B, 64]
        contact_out = self.contact_head(features_last)  # [B, 1]
        
        # Return both sequence (for training) and last timestep (for inference)
        # During training: use velocity_seq for dense supervision
        # During inference: use velocity_out (last timestep only)
        return velocity_seq, velocity_out, contact_out


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
        - contact_out: (batch_size, 1) - contact logits at last timestep
    
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
            contact_out: (batch_size, 1) - contact logits at last timestep
        """
        # Apply global z-score normalization to all input features
        # These statistics are embedded in the model and exported to ONNX
        x_normalized = (x - self.global_mean) / (self.global_std + self.eps)
        
        # Pass normalized data through the base model
        return self.base_model(x_normalized)