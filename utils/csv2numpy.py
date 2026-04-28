import os
import argparse
import glob
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import yaml

def csv2numpy_one_seq(data_pth, save_pth):
    """
    Load data from CSV files and generate numpy files without splitting into train/val/test.
    
    Inputs:
    - data_pth: path to CSV data folder
    - save_pth: path to numpy saving directory
    
    Expected CSV columns for bipedal robot:
    - Joint positions (12): left/right hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    - Joint velocities (12): same joints as positions
    - IMU data (6): acc_body_x/y/z, gyro_body_x/y/z
    - FK foot positions (6): fk_left/right_foot_pos_x/y/z
    - FK foot velocities (6): fk_left/right_foot_vel_x/y/z
    - Joint torques (12): same joints as positions
    - Joint commands (12): same joints as positions
    - Command velocity (1): cmd_vel_x (forward velocity command)
    - Contact labels (2): lfoot-contact, rfoot-contact (binary)
    
    Output:
    - data array: (num_data, 67) = q(12) + qd(12) + acc(3) + omega(3) + p(6) + v(6) + tau(12) + tau_cmd(12) + cmd_vel(1)
    - label array: (num_data, 1) = binary contacts converted to decimal
    """
    
    # Define column names for data extraction
    joint_names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
    ]
    
    for data_name in glob.glob(data_pth + '*.csv'):
        print("loading... ", data_name)
        
        # Load CSV data
        df = pd.read_csv(data_name)
        
            # # Extract joint positions (q) - 12 values
            # q_cols = ['joint_pos_' + j for j in joint_names]
            # q = df[q_cols].values
            
            # # Extract joint velocities (qd) - 12 values
            # qd_cols = ['joint_vel_' + j for j in joint_names]
            # qd = df[qd_cols].values
        
        
        # Extract IMU data in body frame
        imu_acc = df[['acc_body_x', 'acc_body_y', 'acc_body_z']].values  # 3 values
        imu_omega = df[['gyro_body_x', 'gyro_body_y', 'gyro_body_z']].values  # 3 values
        
        # Extract foot positions from FK - 6 values (3 per foot)
        p = df[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z',
                'fk_right_foot_pos_x', 'fk_right_foot_pos_y', 'fk_right_foot_pos_z']].values
        
        # Extract foot velocities from FK - 6 values (3 per foot)
        v = df[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z',
                'fk_right_foot_vel_x', 'fk_right_foot_vel_y', 'fk_right_foot_vel_z']].values
        
        # Extract joint torques (tau_est) - 12 values
        tau_cols = ['joint_torque_' + j for j in joint_names]
        tau_est = df[tau_cols].values

        tau_cmd_cols = ['joint_action_' + j for j in joint_names]
        tau_cmd = df[tau_cmd_cols].values

        # Extract command velocity - 1 value
        cmd_vel = df[['cmd_vel_x']].values  # 1 value
        
        # Extract contact labels - binary (2 values)
        # Contacts are ordered as [left_foot, right_foot]
        contacts = df[['lfoot-contact', 'rfoot-contact']].values.astype(int)
        
        # Concatenate data: acc(3) + omega(3) + p(6) + v(6) + tau(12) + tau_cmd(12) + cmd_vel(1) = 43 features
        data = np.concatenate((imu_acc, imu_omega, p, v, tau_est, tau_cmd, cmd_vel), axis=1)
        
        # Convert binary contact labels to decimal
        label = binary2decimal(contacts).reshape((-1, 1))
        
        print("Saving data to: " + save_pth + os.path.splitext(os.path.basename(data_name))[0] + ".npy")
        
        np.save(save_pth + os.path.splitext(os.path.basename(data_name))[0] + ".npy", data)
        np.save(save_pth + os.path.splitext(os.path.basename(data_name))[0] + "_label.npy", label)
        
        print("Done!")


def csv2numpy_split(data_pth, save_pth, train_ratio=0.7, val_ratio=0.15):
    """
    Load data from CSV files and concatenate into single numpy arrays.
    
    NOTE: This function now saves ALL data as one dataset without splitting.
    The train/val/test split should happen in train.py AFTER windowing to avoid
    losing data at split boundaries.
    
    Inputs:
    - data_pth: path to CSV data folder
    - save_pth: path to numpy saving directory
    - train_ratio: not used (kept for backward compatibility)
    - val_ratio: not used (kept for backward compatibility)
    
    Output:
    - all_data.npy: all data concatenated (input features)
    - all_labels.npy: contact labels (decimal encoded)
    - all_foot_velocities.npy: foot velocities in world frame (6D: left xyz + right xyz)
    - all_data_boundaries.npy: indices marking end of each run (to prevent window bleeding)
    """
    
    num_features = 30  # acc(3) + omega(3) + p(6) + v(6) + tau(12) + tau_cmd(12) + cmd_vel(1)
    all_data = np.zeros((0, num_features))
    all_labels = np.zeros((0, 1))
    all_foot_velocities = np.zeros((0, 6))  # World frame foot velocities: left(3) + right(3)
    
    # Track boundaries between different runs to prevent window bleeding
    all_boundaries = []
    
    # Define column names for data extraction
    joint_names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
    ]
    
    # Process all CSV files in the folder
    for data_name in sorted(glob.glob(data_pth + '*.csv')):
        print("loading... ", data_name)
        
        # Load CSV data
        df = pd.read_csv(data_name)
        
        # Detect run boundaries within the file by finding time resets
        # Time resets to ~0.02 indicate a new run
        time_col = df['timestamp'].values if 'timestamp' in df.columns else None
        run_boundaries = [0]  # Start of first run
        
        if time_col is not None:
            # Find where time resets (decreases or goes back to ~0.02)
            for i in range(1, len(time_col)):
                if time_col[i] < time_col[i-1] or (time_col[i] <= 0.03 and time_col[i-1] > 0.1):
                    run_boundaries.append(i)
                    print(f"  Detected run boundary at index {i} (time reset from {time_col[i-1]:.4f} to {time_col[i]:.4f})")
        
        run_boundaries.append(len(df))  # End of last run
        print(f"  Found {len(run_boundaries)-1} runs in file")
        
        # Process each run separately
        for run_idx in range(len(run_boundaries) - 1):
            start_idx = run_boundaries[run_idx]
            end_idx = run_boundaries[run_idx + 1]
            
            df_run = df.iloc[start_idx:end_idx]
            
            # # Extract joint positions (q) - 12 values
            # q_cols = ['joint_pos_' + j for j in joint_names]
            # q = df_run[q_cols].values
            
            # # Extract joint velocities (qd) - 12 values
            # qd_cols = ['joint_vel_' + j for j in joint_names]
            # qd = df_run[qd_cols].values
            
            
            # Extract IMU data in body frame
            imu_acc = df_run[['acc_body_x', 'acc_body_y', 'acc_body_z']].values
            imu_omega = df_run[['gyro_body_x', 'gyro_body_y', 'gyro_body_z']].values
            
            # Extract foot positions from FK
            p = df_run[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z',
                    'fk_right_foot_pos_x', 'fk_right_foot_pos_y', 'fk_right_foot_pos_z']].values
            
            # Extract foot velocities from FK
            v = df_run[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z',
                    'fk_right_foot_vel_x', 'fk_right_foot_vel_y', 'fk_right_foot_vel_z']].values
            
            # Extract joint torques (tau_est) - 12 values
            tau_cols = ['joint_torque_' + j for j in joint_names]
            tau_est = df_run[tau_cols].values

            # tau_cmd_cols = ['joint_action_' + j for j in joint_names]
            # tau_cmd = df_run[tau_cmd_cols].values

            # # Extract command velocity - 1 value
            # cmd_vel = df_run[['cmd_vel_x']].values
            
            # Extract contact labels - binary (2 values)
            contacts = df_run[['lfoot-contact', 'rfoot-contact']].values.astype(int)

            # Calculate foot velocity in world frame by numerical differentiation
            foot_position_world = ['lfoot_pos_x', 'lfoot_pos_y', 'lfoot_pos_z', 
                                   'rfoot_pos_x', 'rfoot_pos_y', 'rfoot_pos_z']
            foot_velocity_world = np.diff(df_run[foot_position_world].values, axis=0) / \
                                  np.diff(df_run['timestamp'].values.reshape(-1, 1), axis=0)
            # Keep size consistent after diff by repeating last velocity
            foot_velocity_world = np.vstack((foot_velocity_world, foot_velocity_world[-1, :]))
            
            # Concatenate current run data: q(12) + qd(12) + acc(3) + omega(3) + p(6) + v(6) + tau(12) + tau_cmd(12) + cmd_vel(1) = 67
            # cur_data = np.concatenate((q, qd, imu_acc, imu_omega, p, v, tau_est, tau_cmd, cmd_vel), axis=1)

            cur_data = np.concatenate((imu_acc, imu_omega, p, v, tau_est), axis=1)  # 43 features (no q/qdot)
            
            # Convert labels from binary to decimal
            cur_label = binary2decimal(contacts).reshape((-1, 1))
            
            # Append to full dataset
            all_data = np.vstack((all_data, cur_data))
            all_labels = np.vstack((all_labels, cur_label))
            all_foot_velocities = np.vstack((all_foot_velocities, foot_velocity_world))
            
            # Record boundary index (end of this run in the full dataset)
            all_boundaries.append(all_data.shape[0])
    
    print(f"\nTotal data collected: {all_data.shape[0]} samples from {len(all_boundaries)} runs")
    
    # Flatten labels to 1D array
    all_labels = all_labels.reshape(-1,)
    
    print("\nSaving data...")
    
    # Save all data as single files - splitting will happen in train.py after windowing
    np.save(save_pth + "all_data.npy", all_data)
    np.save(save_pth + "all_labels.npy", all_labels)
    np.save(save_pth + "all_foot_velocities.npy", all_foot_velocities)
    np.save(save_pth + "all_data_boundaries.npy", np.array(all_boundaries))
    
    print(f"Saved {all_data.shape[0]} samples to all_data.npy")
    print(f"Saved {all_data.shape[0]} contact labels to all_labels.npy")
    print(f"Saved {all_foot_velocities.shape[0]} foot velocity labels to all_foot_velocities.npy")
    print(f"Saved {len(all_boundaries)} run boundaries to all_data_boundaries.npy")
    print("Done!")


def binary2decimal(a, axis=-1):
    """
    Convert binary contact labels to decimal values.
    
    For bipedal robot:
    - [0, 0] -> 0 (both feet in air)
    - [0, 1] -> 1 (left foot in air, right foot on ground)
    - [1, 0] -> 2 (left foot on ground, right foot in air)
    - [1, 1] -> 3 (both feet on ground)
    """
    return np.right_shift(np.packbits(a, axis=axis), 8 - a.shape[axis]).squeeze()


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to numpy.')
    parser.add_argument('--config_name', type=str, 
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../config/mat2numpy_config.yaml')
    parser.add_argument('--mode', type=str, default='train', 
                        help='Mode: train (split data) or inference (single sequence)')
    parser.add_argument('--csv_folder', type=str, default=None,
                        help='Path to CSV folder (overrides config file)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save numpy files (overrides config file)')
    args = parser.parse_args()
    
    # Load config if it exists
    config = {}
    if os.path.exists(args.config_name):
        config = yaml.load(open(args.config_name), Loader=yaml.FullLoader)
    
    # Override with command line arguments if provided
    if args.csv_folder:
        config['csv_folder'] = args.csv_folder
    if args.save_path:
        config['save_path'] = args.save_path
    if args.mode:
        config['mode'] = args.mode
    
    # Set defaults if not in config
    config.setdefault('csv_folder', '../Data/CSVFiles/')
    config.setdefault('save_path', '../Data/NumpyFiles/')
    config.setdefault('mode', 'train')
    config.setdefault('train_ratio', 0.7)
    config.setdefault('val_ratio', 0.15)
    
    print("Using configuration:")
    print(f"  Mode: {config['mode']}")
    print(f"  CSV folder: {config['csv_folder']}")
    print(f"  Save path: {config['save_path']}")
    
    if config['mode'] == 'train':
        print(f"  Train ratio: {config['train_ratio']}")
        print(f"  Val ratio: {config['val_ratio']}")
        csv2numpy_split(config['csv_folder'], config['save_path'], 
                        config['train_ratio'], config['val_ratio'])
    elif config['mode'] == 'inference':
        csv2numpy_one_seq(config['csv_folder'], config['save_path'])
    else:
        print(f"Error: Unknown mode '{config['mode']}'. Use 'train' or 'inference'.")


if __name__ == '__main__':
    main()
