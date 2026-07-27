import cmd
import os
import argparse
import glob
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import yaml


def quaternion_to_rotation_matrix(quaternions):
    """Return body-to-world rotation matrices for `[w, x, y, z]` quaternions."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Body quaternion contains a zero-norm sample.')
    quaternions = quaternions / norms
    w, x, y, z = quaternions.T
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=1).reshape(-1, 3, 3)


def csv2numpy_split(data_pth, save_pth, train_ratio=0.7, val_ratio=0.15, cmd_vel_x_windows=((0.0, 2.0),)):
    """
    Load data from CSV files and concatenate into single numpy arrays.
    
    IMPORTANT: This function saves ALL data as one dataset WITHOUT splitting.
    The train/val/test split happens in train.py BY RUNS (not windows) to prevent
    data leakage from overlapping sliding windows.
    
    Boundaries are saved to mark the end of each run. These are CRITICAL for:
    1. Preventing windows from spanning across different runs
    2. Ensuring all windows from a run go to the same split (train/val/test)
    3. Avoiding data leakage where test windows overlap with training windows
    
    Features are raw sensor data without cmd_vel normalization.
    tau_mse is calculated here from tau_est and included as a feature.
    
    Inputs:
    - data_pth: path to CSV data folder
    - save_pth: path to numpy saving directory
    - train_ratio: not used (kept for backward compatibility)
    - val_ratio: not used (kept for backward compatibility)
    
    Output:
    - all_data.npy: all data concatenated with left-leg features
      NOTE: All features are RAW - no normalization by cmd_vel or any other feature
    - all_labels.npy: left-foot contact labels (shape: N x 1, binary 0/1)
    - all_data_boundaries.npy: indices marking end of each run (CRITICAL for preventing data leakage)
    - all_data_metadata.npy: metadata dict with num_features (SOURCE OF TRUTH for network architecture)
    - all_body_velocities.npy: body-frame body velocities (shape: N x 1 x 3)
    """
    
    # Ensure save directory exists
    os.makedirs(save_pth, exist_ok=True)
    
    # Ensure paths have trailing slashes for proper concatenation
    if not save_pth.endswith('/'):
        save_pth += '/'
    if not data_pth.endswith('/'):
        data_pth += '/'
    
    # num_features will be determined automatically from the actual data shape
    all_data = None  # Will be initialized after first sample
    all_labels = np.zeros((0, 1))
    all_body_velocities = np.zeros((0, 1, 3))
    num_features = None  # Will be set from cur_data.shape[1] after first run
    
    # Track boundaries between different runs to prevent window bleeding
    # CRITICAL: These boundaries are used in train.py to split by RUNS (not windows)
    # This prevents data leakage from overlapping sliding windows
    all_boundaries = []
    
    left_joint_names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    ]
    joint_names = left_joint_names
    
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
            # Split runs when timestamp jump between consecutive samples is large.
            for i in range(1, len(time_col)):
                dt = abs(time_col[i] - time_col[i - 1])
                if dt > 0.025:
                    run_boundaries.append(i)
        
        run_boundaries.append(len(df))  # End of last run
        print(f"  Found {len(run_boundaries)-1} runs in file")
        
        # Process each run separately
        for run_idx in range(len(run_boundaries) - 1):
            start_idx = run_boundaries[run_idx]
            end_idx = run_boundaries[run_idx + 1]
            
            df_run = df.iloc[start_idx:end_idx]
            
            # Skip runs with fewer than 2 samples (can't compute velocity differences)
            if len(df_run) < 2:
                # print(f"  Skipping run {run_idx} (only {len(df_run)} sample)")
                continue
            
            # Keep runs whose maximum cmd_vel_x belongs to any configured window.
            if 'cmd_vel_x' in df_run.columns:
                max_cmd_vel = df_run['cmd_vel_x'].max()
                if not any(low <= max_cmd_vel <= high for low, high in cmd_vel_x_windows):
                    # print(f"  Skipping run {run_idx} (max cmd_vel_x={max_cmd_vel:.2f} outside {cmd_vel_x_windows})")
                    continue
            
            # Extract IMU data in body frame
            imu_acc = df_run[['acc_body_x', 'acc_body_y', 'acc_body_z']].values
            imu_omega = df_run[['gyro_body_x', 'gyro_body_y', 'gyro_body_z']].values

            # Extract left-leg joint positions and velocities.
            q_cols = ['joint_pos_' + j for j in joint_names]
            q = df_run[q_cols].values
            
            qd_cols = ['joint_vel_' + j for j in joint_names]
            qd = df_run[qd_cols].values
            
            p_left = df_run[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z']].values
            p = p_left
            
            v_left = df_run[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z']].values
            v = v_left
            
            tau_cols = ['joint_torque_' + j for j in joint_names]
            tau_est = df_run[tau_cols].values

            # joint_target_cols = ['joint_action_' + j for j in joint_names]
            # joint_target = df_run[joint_target_cols].values

            # Extract command velocity - 1 value
            cmd_vel = df_run[['cmd_vel_x']].values

            tau_mse = np.sum(tau_est ** 2, axis=1, keepdims=True)

            # Concatenate features - num_features is auto-detected from shape
            cur_data = (np.concatenate([q, qd, p, v, tau_est, tau_mse], axis=1))  # Shape: (num_samples, num_features))
            
            # Initialize all_data and capture num_features from actual data shape
            if num_features is None:
                num_features = cur_data.shape[1]
                all_data = np.zeros((0, num_features))
                print(f"\nAuto-detected {num_features} features from data shape")

            # -----------------------------

            # Output extraction
            
            contacts = df_run[['lfoot-contact']].values.astype(int)
            
            body_velocity_world = df_run[['vel_x', 'vel_y', 'vel_z']].values
            body_quaternion = df_run[['quat_w', 'quat_i', 'quat_j', 'quat_k']].values
            rotation_body_to_world = quaternion_to_rotation_matrix(body_quaternion)
            body_velocities = np.einsum(
                'nij,nj->ni', rotation_body_to_world.transpose(0, 2, 1), body_velocity_world
            )[:, np.newaxis, :]


            cur_label = contacts
            
            # Append to full dataset
            all_data = np.vstack((all_data, cur_data))
            all_labels = np.vstack((all_labels, cur_label))
            all_body_velocities = np.vstack((all_body_velocities, body_velocities))
            
            # Record boundary index (end of this run in the full dataset)
            all_boundaries.append(all_data.shape[0])
    
    print(f"\nTotal data collected: {all_data.shape[0]} samples from {len(all_boundaries)} runs")
    
    # DEBUG: Show run sizes to verify proper boundary detection
    print(f"\nDEBUG: Run size distribution:")
    run_sizes = []
    prev_boundary = 0
    for i, boundary in enumerate(all_boundaries):
        run_size = boundary - prev_boundary
        run_sizes.append(run_size)
        if i < 5:  # Show first 5 runs
            print(f"  Run {i}: {run_size} samples")
        prev_boundary = boundary
    if len(all_boundaries) > 5:
        print(f"  ... and {len(all_boundaries) - 5} more runs")
    print(f"  Average run size: {np.mean(run_sizes):.1f} samples")
    print(f"  Min run size: {np.min(run_sizes)} samples")
    print(f"  Max run size: {np.max(run_sizes)} samples")
    
    # CRITICAL WARNING
    if len(all_boundaries) == 1:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: Only 1 run boundary detected!")
    
    print("\nSaving data...")
    
    # Save all data as single files - splitting will happen in train.py after windowing
    np.save(save_pth + "all_data.npy", all_data)
    np.save(save_pth + "all_labels.npy", all_labels)
    np.save(save_pth + "all_body_velocities.npy", all_body_velocities)
    np.save(save_pth + "all_data_boundaries.npy", np.array(all_boundaries))
    
    # Save metadata including num_features (source of truth for network architecture)
    metadata = {
        'num_features': num_features,
        'num_velocity_targets': all_body_velocities.shape[1],
        'num_samples': all_data.shape[0],
        'num_runs': len(all_boundaries)
    }
    np.save(save_pth + "all_data_metadata.npy", metadata)
    
    print(f"Saved {all_data.shape[0]} samples to all_data.npy")
    print(f"Saved {all_labels.shape[0]} left-foot contact labels to all_labels.npy")
    print(f"Saved {all_body_velocities.shape[0]} body-frame [vx/vy/vz] targets to all_body_velocities.npy")
    print(f"Saved {len(all_boundaries)} run boundaries to all_data_boundaries.npy")
    print(f"Saved metadata (num_features={num_features}) to all_data_metadata.npy")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to numpy.')
    parser.add_argument('--config_name', type=str, 
                        default=os.path.dirname(os.path.abspath(__file__)) + '/../config/network_params.yaml')
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
        config['data_folder'] = args.save_path  # Also set data_folder for consistency
    
    # Set defaults if not in config
    config.setdefault('csv_folder', '../Data/CSVFiles/')
    config.setdefault('data_folder', '../Data/NumpyFiles/')
    config.setdefault('save_path', config['data_folder'])  # Use data_folder if save_path not set
    config.setdefault('train_ratio', 0.7)
    config.setdefault('val_ratio', 0.15)
    cmd_vel_x_windows = config.get('cmd_vel_x_windows', [[0.0, 2.0]])
    if not cmd_vel_x_windows or not all(isinstance(window, (list, tuple)) and len(window) == 2 and window[0] <= window[1]
               for window in cmd_vel_x_windows):
        raise ValueError('cmd_vel_x_windows must be a non-empty list of [min, max] windows with min <= max')
    
    print("Using configuration:")
    print(f"  CSV folder: {config['csv_folder']}")
    print(f"  Save path: {config['save_path']}")
    print(f"  Train ratio: {config['train_ratio']}")
    print(f"  Val ratio: {config['val_ratio']}")
    print(f"  Command-velocity windows: {cmd_vel_x_windows}")
    
    csv2numpy_split(config['csv_folder'], config['save_path'],
                    config['train_ratio'], config['val_ratio'],
                    cmd_vel_x_windows)


if __name__ == '__main__':
    main()
