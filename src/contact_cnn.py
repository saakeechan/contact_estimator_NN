import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

class contact_cnn(nn.Module):
    def __init__(self, window_size=150):
        super(contact_cnn, self).__init__()
        self.block1 = nn.Sequential(
            # First convolutional layer
            # Takes 54 input feature channels (q, qd, IMU, p, v)
            # Produces 64 learned feature maps (filters)
            # kernel_size=3: each filter looks at 3 consecutive timesteps
            # stride=1: moves one timestep at a time
            # padding=1: adds 1 zero on each side to maintain length
            nn.Conv1d(in_channels=54,      # Input: 54 sensor features
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
        # After block1: (batch, 150, 54) → (batch, 75, 64)

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
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_input_size,
                      out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512,
                      out_features=4),  # 2 legs: 2^2 = 4 contact combinations
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block2_out_reshape = block2_out.view(block2_out.shape[0], -1)
        fc_out = self.fc(block2_out_reshape)
        return fc_out


class contact_cnn_1block(nn.Module):
    def __init__(self):
        super(contact_cnn_1block, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=54,
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
                      out_features=4),  # 2 legs: 2^2 = 4 contact combinations
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
            nn.Conv1d(in_channels=54,
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
                      out_features=4),  # 2 legs: 2^2 = 4 contact combinations
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
            nn.Conv1d(in_channels=54,
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
                      out_features=4),  # 2 legs: 2^2 = 4 contact combinations
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
            nn.Conv1d(in_channels=54,
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
                      out_features=4),  # 2 legs: 2^2 = 4 contact combinations
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
                      out_features=4),  # 2 legs: 2^2 = 4 contact combinations
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
