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
        At initialization we load .npy files for data, labels, and foot velocities.
        self.data: a 2D array of all data points. rows are time axis, columns are features. (num_data, num_features)
        self.label: LEFT leg contact state (0 or 1). Shape: (num_data, 1) - LEFT leg only
        self.foot_velocity: LEFT leg foot velocity magnitude (norm). Shape: (num_data, 1) - LEFT leg only
        """
        data = np.load(data_path)
        label = np.load(label_path)
        
        # # Load foot velocities - LEFT LEG ONLY
        # velocity_path = data_path.replace('_data.npy', '_foot_velocities.npy')
        # if not os.path.exists(velocity_path):
        #     velocity_path = data_path.replace('.npy', '_foot_velocities.npy')
        
        # if os.path.exists(velocity_path):
        #     foot_velocity = np.load(velocity_path)
        #     print(f"Loaded foot velocities from {velocity_path}")
        # else:
        #     print(f"Warning: No foot velocity file found. Creating zero velocities.")
        #     foot_velocity = np.zeros((len(data), 1), dtype=np.float32)  # LEFT leg only
        
        self.window_size = window_size
        self.data = torch.from_numpy(data).type('torch.FloatTensor').to(device)
        
        # Load foot velocities - LEFT LEG ONLY
        velocity_path = data_path.replace('all_data.npy', 'all_foot_velocities.npy')
        if os.path.exists(velocity_path):
            foot_velocity = np.load(velocity_path)
            print(f"Loaded foot velocities from {velocity_path}")
        else:
            print(f"Warning: No foot velocity file found at {velocity_path}. Creating zero velocities.")
            foot_velocity = np.zeros((len(data), 1), dtype=np.float32)  # LEFT leg only
        
        self.foot_velocity = torch.from_numpy(foot_velocity).type('torch.FloatTensor').to(device)
        
        # Labels are LEFT leg contact (0 or 1) - Shape: (num_data, 1) - LEFT leg only
        label_binary = label.astype(np.float32)  # Shape: (num_data, 1)
        self.label = torch.from_numpy(label_binary).type('torch.FloatTensor').to(device)
        
        # Load run boundaries to prevent window bleeding across different runs
        boundary_path = data_path.replace('.npy', '_boundaries.npy')
        if os.path.exists(boundary_path):
            boundaries = np.load(boundary_path).tolist()  # Convert to list for consistency
            print(f"Loaded {len(boundaries)} run boundaries from {boundary_path}")
        else:
            print(f"Warning: No boundary file found at {boundary_path}. Treating all data as one run.")
            # If no boundaries, treat all data as one run
            boundaries = [data.shape[0]]
        
        # Validate boundaries
        if len(boundaries) == 0:
            raise ValueError("No run boundaries found. Cannot process data.")
        
        # Pre-compute valid window indices AND track which run each window belongs to
        # This is critical for preventing data leakage - we'll split by runs, not windows
        self.valid_indices = []
        self.window_to_run_id = []  # Maps window index to run ID
        total_possible_windows = data.shape[0] - window_size + 1
        
        if total_possible_windows <= 0:
            raise ValueError(f"Window size ({window_size}) is larger than data length ({data.shape[0]}). No valid windows.")
        
        # Boundaries mark END of each run. Convert to [start, end) pairs
        run_starts = [0] + boundaries[:-1]
        run_ends = boundaries
        
        for idx in range(total_possible_windows):
            window_end = idx + window_size
            
            # Find which run this window belongs to
            # A window belongs to a run if it starts AND ends within that run
            window_run_id = None
            for run_id, (run_start, run_end) in enumerate(zip(run_starts, run_ends)):
                if run_start <= idx and window_end <= run_end:
                    window_run_id = run_id
                    break
            
            # Only include windows that belong entirely to one run (don't cross boundaries)
            if window_run_id is not None:
                self.valid_indices.append(idx)
                self.window_to_run_id.append(window_run_id)
        
        num_invalid = total_possible_windows - len(self.valid_indices)
        num_runs = len(boundaries)
        print(f"Valid windows: {len(self.valid_indices)} / {total_possible_windows} (skipped {num_invalid} boundary-crossing windows)")
        print(f"Windows distributed across {num_runs} runs")
        
        # Validate that we have valid windows
        if len(self.valid_indices) == 0:
            raise ValueError(f"No valid windows found. Check window_size ({window_size}) vs run lengths.")

    def __len__(self):
        return len(self.valid_indices)
    
    def get_run_id(self, window_idx):
        """Get the run ID for a given window index."""
        return self.window_to_run_id[window_idx]
    
    def get_windows_by_run_ids(self, run_ids):
        """Get all window indices that belong to the specified run IDs."""
        return [i for i, run_id in enumerate(self.window_to_run_id) if run_id in run_ids]

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
        - label: (batch_size, 1) - binary contact for left leg (last timestep)
        - velocity: (batch_size, window_size, 1) - full velocity sequence (model extracts last timestep)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Map from valid index to actual data index
        real_idx = self.valid_indices[idx]
        
        # Return raw unnormalized data (normalization done inside the model)
        # Feature layout: LEFT leg features from csv2numpy.py
        this_data = self.data[real_idx:real_idx+self.window_size,:]
        
        # Label: contact at last timestep
        this_label = self.label[real_idx+self.window_size-1]  # Shape: (1,) - left leg only
        
        # Velocity: full sequence (training code extracts last timestep)
        this_velocity = self.foot_velocity[real_idx:real_idx+self.window_size, :]  # Shape: (window_size, 1)
            
        sample = {'data': this_data, 'label': this_label, 'velocity': this_velocity}

        return sample


# def main():
#     parser = argparse.ArgumentParser(description='Train network')
#     parser.add_argument('--data_folder', type=str, help='path to contact dataset', default="/home/justin/data/2021-02-21_contact_data_in_lab/cnn_data/")
#     args = parser.parse_args()

#     data = load_data_from_mat(args.data_folder,0.7,0.15)


# if __name__ == '__main__':
#     main()
