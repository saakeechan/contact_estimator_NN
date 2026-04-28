import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

class contact_cnn(nn.Module):
    def __init__(self, window_size=10):
        super(contact_cnn, self).__init__()
        self.block1 = nn.Sequential(
            # First convolutional layer
            # Takes 67 input feature channels (q, qd, IMU, p, v, tau, tau_cmd, cmd_vel)
            # Produces 64 learned feature maps (filters)
            # kernel_size=3: each filter looks at 3 consecutive timesteps
            # stride=1: moves one timestep at a time
            # padding=1: adds 1 zero on each side to maintain length
            nn.Conv1d(in_channels=30,      # Input: 30 sensor features (IMU, p, v, tau)
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
        self.fc_contact = nn.Sequential(
            nn.Linear(in_features=fc_input_size,
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
        
        # Foot velocity regression branch (parallel to contact branch)
        self.fc_velocity = nn.Sequential(
            nn.Linear(in_features=fc_input_size,
                      out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=6),  # 6 outputs: 3D velocity for each foot (left xyz, right xyz)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        
        # Two parallel outputs
        contact_out = self.fc_contact(block2_out_reshape)
        velocity_out = self.fc_velocity(block2_out_reshape)
        
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
    Wrapper class that adds per-window normalization to the contact_cnn model.
    This wrapper can be exported to ONNX so normalization is done inside the model
    during inference, eliminating the need to normalize data externally in C++.
    
    Normalization: z-score normalization per window
    - Compute mean and std along time dimension (dim=1) per feature
    - Normalize: (x - mean) / std
    - Handle std == 0 by replacing with 1 to avoid division by zero
    
    Input shape: (batch_size, window_size, num_features)
    Output shape: (batch_size, 4) for 4 contact combinations

    // Number of features per timestep (q, qd, acc, omega, p, v, tau)
    """
    def __init__(self, base_model, eps=1e-8):
        super(ContactCNNWithNormalization, self).__init__()
        self.base_model = base_model
        self.eps = eps  # Small constant to prevent division by zero
        
    def forward(self, x):
        """
        Args:
            x: Raw input data (batch_size, window_size, num_features)
               NOT normalized
        
        Returns:
            Output predictions (batch_size, 4)
        
        Feature layout (43 total):
            acc: 0-2 (3 features) - IMU acceleration
            omega: 3-5 (3 features) - IMU angular velocity
            p: 6-11 (6 features) - position
            v: 12-17 (6 features) - velocity
            tau: 18-29 (12 features) - joint torques
            tau_cmd: 30-41 (12 features) - joint torque commands
            cmd_vel: 42 (1 feature) - command velocity
        """
        # Split features into groups
        x_imu = x[:, :, :6]           # acc and omega (to be normalized)
        x_others = x[:, :, 6:]          # p, v, tau (not normalized)
        
        # Normalize only IMU features (acc and omega)
        # Compute mean and std per feature across time dimension
        mean_imu = torch.mean(x_imu, dim=1, keepdim=True)  # (batch, 1, 6)
        std_imu = torch.std(x_imu, dim=1, keepdim=True)    # (batch, 1, 6)
        
        # Replace zero std with 1 to avoid NaN (prevents division by zero)
        std_imu = torch.where(std_imu == 0, torch.ones_like(std_imu), std_imu)
        
        # Normalize IMU features: z-score normalization
        x_imu_normalized = (x_imu - mean_imu) / std_imu
        
        # Concatenate: keep q, qd, others unchanged; replace IMU with normalized
        x_normalized = torch.cat([x_imu_normalized, x_others], dim=2)
        
        # Pass normalized data through the base model
        return self.base_model(x_normalized)
    
        #     # Split features into groups
        # x_q_qd = x[:, :, 0:24]           # q and qd (not normalized)
        # x_imu = x[:, :, 24:30]           # acc and omega (to be normalized)
        # x_others = x[:, :, 30:]          # p, v, tau, tau_cmd, cmd_vel (not normalized)
        
        # # Normalize only IMU features (acc and omega)
        # # Compute mean and std per feature across time dimension
        # mean_imu = torch.mean(x_imu, dim=1, keepdim=True)  # (batch, 1, 6)
        # std_imu = torch.std(x_imu, dim=1, keepdim=True)    # (batch, 1, 6)
        
        # # Replace zero std with 1 to avoid NaN (prevents division by zero)
        # std_imu = torch.where(std_imu == 0, torch.ones_like(std_imu), std_imu)
        
        # # Normalize IMU features: z-score normalization
        # x_imu_normalized = (x_imu - mean_imu) / std_imu
        
        # # Concatenate: keep q, qd, others unchanged; replace IMU with normalized
        # x_normalized = torch.cat([x_q_qd, x_imu_normalized, x_others], dim=2)
        
        # # Pass normalized data through the base model
        # return self.base_model(x_normalized)
