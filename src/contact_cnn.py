import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

class contact_cnn(nn.Module):
    def __init__(self, window_size=10):
        super(contact_cnn, self).__init__()
        self.block1 = nn.Sequential(
            # First convolutional layer
            # Takes 20 input feature channels (q, qd, IMU, p, v, tau_est, tau_cmd, cmd_vel, tau_mse)
            # Produces 64 learned feature maps (filters)
            # kernel_size=3: each filter looks at 3 consecutive timesteps
            # stride=1: moves one timestep at a time
            # padding=1: adds 1 zero on each side to maintain length
            nn.Conv1d(in_channels=20,      # Input: 20 features after τ_mse augmentation (19 base + 1 computed in forward)
                    out_channels=64,      # Output: 64 learned patterns
                    kernel_size=3,        # Look at 3 timesteps at once
                    stride=1,             # Slide by 1 timestep
                    padding=1),           # Keep same length (150→150)
            
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
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

    
        # Calculate FC input size based on window_size
        # 2 MaxPool layers (stride=2 each) reduce window by 4x total
        # Final conv outputs 128 channels
        fc_input_size = (window_size // 4) * 128
        
        # Contact detection branch (classification)
        self.fc_contact_left = nn.Sequential(
            nn.Linear(in_features=fc_input_size,
                      out_features=5096),  # Intermediate layer for contact features
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=5096,
                      out_features=512),  # Intermediate layer for contact features
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=1),  # 1 leg: independent binary classification
        )
        
        # Foot velocity regression branch (parallel to contact branch)
        self.fc_velocity_left = nn.Sequential(
            nn.Linear(in_features=fc_input_size,
                      out_features=4),  # Intermediate layer for velocity features
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4,
                      out_features=2),  # Intermediate layer for velocity features
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2,
                      out_features=1),  # 1 output: velocity norm (magnitude) for left foot
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, 20) - includes tau_mse
        # Feature layout: acc(0-2) + omega(3-5) + p(6-8) + v(9-11) + tau_est(12-17) + cmd_vel(18) + tau_mse(19)
        
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        
        # Two parallel outputs: contact classification and velocity regression
        contact_out = self.fc_contact_left(block2_out_reshape)  # Left leg contact (binary)
        velocity_out = self.fc_velocity_left(block2_out_reshape)  # Left foot velocity norm
        
        return contact_out, velocity_out


class contact_cnn_1block(nn.Module):
    def __init__(self):
        super(contact_cnn_1block, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=67,
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


        self.fc = nn.Sequential(
            nn.Linear(in_features=4800,
                      out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=2),  # 2 legs: independent binary classification
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block1_out_reshape = block1_out.view(block1_out.shape[0], -1)
        fc_out = self.fc(block1_out_reshape)
        return fc_out



class contact_cnn_1conv_2blocks(nn.Module):
    def __init__(self):
        super(contact_cnn_1conv_2blocks, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=67,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )


        self.fc = nn.Sequential(
            nn.Linear(in_features=4736,
                      out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=2),  # 2 legs: independent binary classification
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        fc_out = self.fc(block2_out_reshape)
        return fc_out


class contact_cnn_4blocks(nn.Module):
    def __init__(self):
        super(contact_cnn_4blocks, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=67,
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

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4608,
                      out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=2),  # 2 legs: independent binary classification
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)

        block4_out_reshape = block4_out.view(block4_out.shape[0], -1)
        fc_out = self.fc(block4_out_reshape)
        return fc_out

class contact_cnn_256(nn.Module):
    def __init__(self):
        super(contact_cnn_256, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=67,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=9472,
                      out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=2),  # 2 legs: independent binary classification
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        fc_out = self.fc(block2_out_reshape)
        return fc_out


class contact_2d_cnn(nn.Module):
    def __init__(self):
        super(contact_2d_cnn, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=2304,
                      out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=2),  # 2 legs: independent binary classification
        )

        # 4 seperate ones
        # and then left-right based 2 MLPs

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)
        block4_out_reshape = block4_out.view(block4_out.shape[0], -1)
        fc_out = self.fc(block4_out_reshape)
        return fc_out


class ContactCNNWithNormalization(nn.Module):
    """
    Wrapper class that embeds global normalization statistics into the model.
    This wrapper can be exported to ONNX so normalization is done inside the model
    during inference, eliminating the need to normalize data externally in C++.
    
    Normalization strategy: Global z-score normalization
    - Uses mean and std computed from training data (stored as buffers)
    - Normalize: (x - global_mean) / (global_std + eps)
    - Statistics are saved with the model and exported to ONNX
    
    Input shape: (batch_size, window_size, 37) - RAW unnormalized data
    Output shape: (batch_size, 1) for contact + (batch_size, 1) for velocity
    
    Feature layout (19 features input, 20 after tau_mse):
        acc: 0-2 (3) - IMU acceleration
        omega: 3-5 (3) - IMU angular velocity
        p: 6-8 (3) - foot position
        v: 9-11 (3) - foot velocity
        tau_est: 12-17 (6) - joint torques estimated
        cmd_vel: 18 (1) - command velocity
        tau_mse: 19 (1) - computed from RAW tau_est BEFORE normalization
    """
    def __init__(self, base_model, global_mean=None, global_std=None, eps=1e-8):
        """
        Args:
            base_model: The contact_cnn model to wrap
            global_mean: Tensor of shape (1, 1, 20) with training data mean per feature (including tau_mse)
            global_std: Tensor of shape (1, 1, 20) with training data std per feature (including tau_mse)
            eps: Small constant to prevent division by zero
        """
        super(ContactCNNWithNormalization, self).__init__()
        self.base_model = base_model
        self.eps = eps
        
        # Register as buffers (not parameters - won't be trained, but saved with model and exported to ONNX)
        if global_mean is not None:
            self.register_buffer('global_mean', global_mean)
        else:
            # Fallback: no normalization if stats not provided
            self.register_buffer('global_mean', torch.zeros(1, 1, 20))
            
        if global_std is not None:
            self.register_buffer('global_std', global_std)
        else:
            # Fallback: no normalization if stats not provided
            self.register_buffer('global_std', torch.ones(1, 1, 20))
        
    def forward(self, x):
        """
        Apply feature engineering, then normalization, then pass through base model.
        
        Args:
            x: Raw input data (batch_size, window_size, 19) - NOT normalized
        
        Returns:
            contact_out: (batch_size, 1) - contact prediction logits
            velocity_out: (batch_size, 1) - velocity prediction
        """
        # STEP 1: Compute tau_mse from RAW tau_est (before normalization)
        # This preserves the physical meaning of mean squared torque
        tau_est = x[:, :, 12:18]  # Extract raw tau_est: (batch_size, window_size, 6)
        tau_mse = torch.mean(tau_est ** 2, dim=2, keepdim=True)  # (batch_size, window_size, 1)
        
        # Concatenate tau_mse as 20th feature
        x_with_tau_mse = torch.cat([x, tau_mse], dim=2)  # (batch_size, window_size, 20)
        
        # STEP 2: Apply global z-score normalization to ALL 20 features
        # These statistics are embedded in the model and exported to ONNX
        x_normalized = (x_with_tau_mse - self.global_mean) / (self.global_std + self.eps)
        
        # STEP 3: Pass normalized data through the base model
        return self.base_model(x_normalized)

