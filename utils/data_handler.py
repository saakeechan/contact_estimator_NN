import os
import argparse
import glob

import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data import Dataset, DataLoader

class contact_dataset(Dataset):

    def __init__(self, data_path, label_path, window_size, device='cuda'):
        """
        At initialization we load .npy files for data, label, and foot velocities.
        self.data: a 2D array of all data points. rows are time axis, columns are features. (num_data, num_features)
        self.label: a vector of left foot contact states (0 or 1). (num_data, 1)
        self.foot_velocity: left foot velocity magnitude (norm). (num_data, 1)
        """
        data = np.load(data_path)
        label = np.load(label_path)
        
        # Load foot velocities - LEFT FOOT NORM ONLY
        velocity_path = data_path.replace('_data.npy', '_foot_velocities.npy')
        if not os.path.exists(velocity_path):
            velocity_path = data_path.replace('.npy', '_foot_velocities.npy')
        
        if os.path.exists(velocity_path):
            foot_velocity = np.load(velocity_path)
            print(f"Loaded foot velocities from {velocity_path}")
        else:
            print(f"Warning: No foot velocity file found. Creating zero velocities.")
            foot_velocity = np.zeros((len(data), 1), dtype=np.float32)
        
        self.window_size = window_size
        self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        self.foot_velocity = torch.from_numpy(foot_velocity).type('torch.FloatTensor').to(device)
        
        # Labels are already left foot contact only (0 or 1) from csv2numpy.py
        # No need for bit manipulation - use directly for BCEWithLogitsLoss
        label_binary = label.astype(np.float32).reshape(-1, 1)  # Shape: (num_data, 1)
        self.label = torch.from_numpy(label_binary).type('torch.FloatTensor').to(device)
        
        # Load run boundaries to prevent window bleeding across different runs
        boundary_path = data_path.replace('.npy', '_boundaries.npy')
        boundaries = None
        if os.path.exists(boundary_path):
            boundaries = np.load(boundary_path)
            print(f"Loaded {len(boundaries)} run boundaries from {boundary_path}")
        else:
            print(f"Warning: No boundary file found at {boundary_path}. Windows may span across different runs.")
        
        # Pre-compute valid window indices (windows that don't cross run boundaries)
        self.valid_indices = []
        total_possible_windows = data.shape[0] - window_size + 1
        
        for idx in range(total_possible_windows):
            window_end = idx + window_size
            is_valid = True
            
            if boundaries is not None:
                for boundary in boundaries:
                    if idx < boundary < window_end:
                        is_valid = False
                        break
            
            if is_valid:
                self.valid_indices.append(idx)
        
        num_invalid = total_possible_windows - len(self.valid_indices)
        print(f"Valid windows: {len(self.valid_indices)} / {total_possible_windows} (skipped {num_invalid} boundary-crossing windows)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        
        """
        In this function we get a batch size of our data and convert them into 3D tensor.

        self.data: a 2D array of all data points. rows are time axis, columns are features. (num_data, num_features)

        We use a sliding window of size = window size to create a 3rd dimension for the network 
        to inference along the time axis. After this, at each time step we will have
        window_size x num_features. Thus we can take window_size of data into consideration each time.

        The data is normalized along time domain in the model, so we return raw unnormalized data here.

        Ex. If the window size = 10. new_data[0,:,:] = data[0:10,:], new_data[1,:,:] = data[1:11,:].
                                     new_label[0] = label[9],        new_label[1] = label[10].
        
        Output: 
        - data: (batch_size, window_size, num_features)
        - label: (batch_size, 1) - binary contact for left leg only
        - velocity: (batch_size, 1) - norm (magnitude) of left foot velocity
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Map from valid index to actual data index
        real_idx = self.valid_indices[idx]
        
        # Return raw unnormalized data (normalization done inside the model)
        # Feature layout (32 features): acc(0-2) + omega(3-5) + q(6-11) + qd(12-17) + p(18-20) + v(21-23) + tau_est(24-29) + tau_mse(30) + cmd_vel(31)
        this_data = self.data[real_idx:real_idx+self.window_size,:]
        
        this_label = self.label[real_idx+self.window_size-1]
        this_velocity = self.foot_velocity[real_idx+self.window_size-1]
            
        sample = {'data': this_data, 'label': this_label, 'velocity': this_velocity}

        return sample


# def main():
#     parser = argparse.ArgumentParser(description='Train network')
#     parser.add_argument('--data_folder', type=str, help='path to contact dataset', default="/home/justin/data/2021-02-21_contact_data_in_lab/cnn_data/")
#     args = parser.parse_args()

#     data = load_data_from_mat(args.data_folder,0.7,0.15)


# if __name__ == '__main__':
#     main()
