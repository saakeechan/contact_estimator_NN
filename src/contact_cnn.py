import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

class contact_cnn(nn.Module):
    def __init__(self, window_size=10, num_features=25):
        super(contact_cnn, self).__init__()
        self.num_features = num_features
        self.block1 = nn.Sequential(
            # First convolutional layer
            # Takes num_features RAW input features from csv2numpy.py (LEFT LEG ONLY)
            # Input layout: q(6) + qd(6) + p(3) + v(3) + tau_est(6) + tau_mse(1)
            # Produces 64 learned feature maps (filters)
            # kernel_size=3: each filter looks at 3 consecutive timesteps
            # stride=1: moves one timestep at a time
            # padding=1: adds 1 zero on each side to maintain length
            nn.Conv1d(in_channels=num_features,  # Input: num_features from config (LEFT leg only)
                    out_channels=64,      # Output: 64 learned patterns
                    kernel_size=3,        # Look at 3 timesteps at once
                    stride=1,             # Slide by 1 timestep
                    padding=1),           # Keep same length (window_size→window_size)
            
            # Activation function
            # Introduces non-linearity (allows learning complex patterns)
            # ReLU(x) = max(0, x): zeros out negative values
            nn.ReLU(),
            
            # Second convolutional layer
            # Refines the 64 features from first conv layer
            # Learns combinations of low-level patterns
            # Same parameters as first conv (except in_channels)
            nn.Conv1d(in_channels=64,      # Input: 64 features from previous layer
                    out_channels=64,      # Output: 64 refined features
                    kernel_size=3,        # Look at 3 timesteps
                    stride=1,             # Slide by 1 timestep
                    padding=1),           # Keep same length (150→150)
            
            # Second activation
            nn.ReLU(),
            
            # REGULARIZATION: Dropout layer
            # During training: randomly sets 50% of neuron outputs to 0
            # During inference: does nothing (automatically disabled)
            # Purpose: prevents overfitting by forcing redundant learning
            # p=0.5 means 50% dropout probability
            # Does NOT change tensor dimensions
            nn.Dropout(p=0.5),              # Randomly drop 50% of neurons
            
            # DOWNSAMPLING: Max pooling layer
            # Reduces temporal dimension by taking max in each window
            # kernel_size=2: looks at 2 consecutive values
            # stride=2: moves by 2 (non-overlapping windows)
            # Takes max of [t0,t1], then [t2,t3], then [t4,t5], etc.
            # Reduces length: 150 → 75 timesteps
            # Purpose: (1) reduce computation (2) focus on strongest signals
            nn.MaxPool1d(kernel_size=2,     # Window size of 2
                        stride=2)           # Move by 2 (no overlap)
        )
        # After block1: (batch, 150, 67) → (batch, 75, 64)

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

    
        # Calculate FC input size based on window_size
        # 2 MaxPool layers (stride=2) reduce window by 4x
        # Final conv outputs 64 channels
        fc_input_size = (window_size // 4) * 64
        
        # # COMMENTED OUT: Two separate MLPs approach
        # # MLP for LEFT leg contact detection
        # self.fc_contact = nn.Sequential(
        #     nn.Linear(in_features=fc_input_size,
        #               out_features=64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=64,
        #               out_features=16),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=16,
        #               out_features=1),  # 1 output: binary contact for LEFT leg
        # )
        # 
        # # MLP for RIGHT leg contact detection (separate head, same architecture)
        # self.fc_contact_right = nn.Sequential(
        #     nn.Linear(in_features=fc_input_size,
        #               out_features=64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=64,
        #               out_features=16),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=16,
        #               out_features=1),  # 1 output: binary contact for RIGHT leg
        # )
        
        # Single MLP for both legs contact detection with 2 outputs
        self.fc_contact = nn.Sequential(
            nn.Linear(in_features=fc_input_size,
                      out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256,
                      out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64,
                      out_features=2),  # 2 outputs: binary contact for both legs [left, right]
        )
        
        # # Separate MLP for left leg velocity regression
        # self.fc_velocity_left = nn.Sequential(
        #     nn.Linear(in_features=fc_input_size,
        #               out_features=2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=2048,
        #               out_features=512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=512,
        #               out_features=1),  # 1 output: velocity norm for left leg
        # )
        
        # # Separate MLP for right leg velocity regression
        # self.fc_velocity_right = nn.Sequential(
        #     nn.Linear(in_features=fc_input_size,
        #               out_features=2048),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=2048,
        #               out_features=512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=512,
        #               out_features=1),  # 1 output: velocity norm for right leg
        # )

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features) - RAW features from csv2numpy.py
        
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        
        # Single MLP with 2 outputs
        contact_out = self.fc_contact(block2_out_reshape)  # Shape: (batch, 2) -> [:, 0] = left, [:, 1] = right
        
        return contact_out


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
    Output shape: (batch_size, 2) for contact ([:, 0]=left, [:, 1]=right)
    
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
               Features: q(6) + qd(6) + p(3) + v(3) + tau_est(6) + tau_mse(1)
        
        Returns:
            contact_out: (batch_size, 2) - contact prediction logits [:, 0]=left, [:, 1]=right
        """
        # Apply global z-score normalization to all input features
        # These statistics are embedded in the model and exported to ONNX
        x_normalized = (x - self.global_mean) / (self.global_std + self.eps)
        
        # Pass normalized data through the base model
        return self.base_model(x_normalized)

