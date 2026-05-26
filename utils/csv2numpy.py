import cmd
import os
import argparse
import glob
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import yaml


def csv2numpy_split(data_pth, save_pth, train_ratio=0.7, val_ratio=0.15):
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
    - all_data.npy: all data concatenated (num_features auto-detected from shape) - LEFT LEG ONLY
      Layout: q[3,4](2) + p[x,z](2) + v[x,z](2) + tau_est[1-5](5) + tau_mse(1) = 12 features
      NOTE: All features are RAW - no normalization by cmd_vel or any other feature
    - all_labels.npy: LEFT leg contact labels only (shape: N x 1, binary 0/1)
    - all_data_boundaries.npy: indices marking end of each run (CRITICAL for preventing data leakage)
    - all_data_metadata.npy: metadata dict with num_features (SOURCE OF TRUTH for network architecture)
    - all_foot_velocities.npy: foot velocity magnitudes for LEFT leg only (shape: N x 1)
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
    all_labels = np.zeros((0, 1))  # LEFT leg only
    all_foot_velocities = np.zeros((0, 1))  # World frame foot velocities: LEFT leg only - magnitudes
    num_features = None  # Will be set from cur_data.shape[1] after first run
    
    # Track boundaries between different runs to prevent window bleeding
    # CRITICAL: These boundaries are used in train.py to split by RUNS (not windows)
    # This prevents data leakage from overlapping sliding windows
    all_boundaries = []
    
    # Define column names for data extraction - LEFT LEG ONLY
    joint_names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
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
                print(f"  Skipping run {run_idx} (only {len(df_run)} sample)")
                continue
            
            # Extract IMU data in body frame
            imu_acc = df_run[['acc_body_x', 'acc_body_y', 'acc_body_z']].values
            imu_omega = df_run[['gyro_body_x', 'gyro_body_y', 'gyro_body_z']].values

            # Extract joint positions (q) - 6 values (LEFT leg only)
            q_cols = ['joint_pos_' + j for j in joint_names]
            q = df_run[q_cols].values
            
            # Extract joint velocities (qd) - 6 values (LEFT leg only)
            qd_cols = ['joint_vel_' + j for j in joint_names]
            qd = df_run[qd_cols].values
            
            # Extract foot positions from FK - LEFT leg only
            p_left = df_run[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z']].values
            # p_right = df_run[['fk_right_foot_pos_x', 'fk_right_foot_pos_y', 'fk_right_foot_pos_z']].values
            p = p_left  # Keep x and z for LEFT leg only, shape: (num_samples, 2)
            
            # Extract foot velocities from FK - LEFT leg only
            v_left = df_run[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z']].values
            # v_right = df_run[['fk_right_foot_vel_x', 'fk_right_foot_vel_y', 'fk_right_foot_vel_z']].values
            v = v_left  # Keep x and z for LEFT leg only, shape: (num_samples, 2)
            
            # Extract joint torques (tau_est) - 6 values (LEFT leg only)
            tau_cols = ['joint_torque_' + j for j in joint_names]
            tau_est = df_run[tau_cols].values

            # joint_target_cols = ['joint_action_' + j for j in joint_names]
            # joint_target = df_run[joint_target_cols].values

            # Extract command velocity - 1 value
            # cmd_vel = df_run[['cmd_vel_x']].values

            # Calculate tau_mse from tau_est for LEFT leg only
            tau_mse = np.sum(tau_est ** 2, axis=1, keepdims=True)  # All 6 left leg joints, shape: (num_samples, 1)

            # Concatenate features - num_features is auto-detected from shape
            cur_data = (np.concatenate([imu_acc, imu_omega, q, qd, p, v, tau_est, tau_mse], axis=1))  # Shape: (num_samples, num_features)
            
            # Initialize all_data and capture num_features from actual data shape
            if num_features is None:
                num_features = cur_data.shape[1]
                all_data = np.zeros((0, num_features))
                print(f"\nAuto-detected {num_features} features from data shape")

            # -----------------------------

            # Output extraction
            
            # Extract contact labels - both legs (binary: 0 or 1)
            contacts_left = df_run[['lfoot-contact']].values.astype(int)   # Shape: (num_samples, 1)
            # contacts_right = df_run[['rfoot-contact']].values.astype(int)  # Shape: (num_samples, 1)
            contacts = contacts_left  # Shape: (num_samples, 1)
            
            # Calculate foot velocity in world frame by numerical differentiation for BOTH legs
            # Left foot velocity
            # lfoot_position_world = ['lfoot_pos_x', 'lfoot_pos_y', 'lfoot_pos_z']
            lfoot_position_world = ['lfoot_pos_x', 'lfoot_pos_y']
            lfoot_velocity_world = np.diff(df_run[lfoot_position_world].values, axis=0) / np.diff(df_run['timestamp'].values.reshape(-1, 1), axis=0)
            lfoot_velocity_world = np.vstack((lfoot_velocity_world, lfoot_velocity_world[-1, :]))  # Keep size consistent
            lfoot_velocity_norm = np.linalg.norm(lfoot_velocity_world, axis=1, keepdims=True) + 1e-8
        
            # Left leg velocity only
            foot_velocities = lfoot_velocity_norm  # Shape: (num_samples, 1)


            # LEFT leg contact labels (already 0 or 1, no conversion needed)
            cur_label = contacts  # Shape: (num_samples, 1) - LEFT leg only
            
            # Append to full dataset
            all_data = np.vstack((all_data, cur_data))
            all_labels = np.vstack((all_labels, cur_label))
            all_foot_velocities = np.vstack((all_foot_velocities, foot_velocities))  # LEFT leg velocities
            
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
    
    # Labels are already 0/1 for left foot (no flattening needed since they're already 1D per sample)
    
    print("\nSaving data...")
    
    # Save all data as single files - splitting will happen in train.py after windowing
    np.save(save_pth + "all_data.npy", all_data)
    np.save(save_pth + "all_labels.npy", all_labels)
    np.save(save_pth + "all_foot_velocities.npy", all_foot_velocities)
    np.save(save_pth + "all_data_boundaries.npy", np.array(all_boundaries))
    
    # Save metadata including num_features (source of truth for network architecture)
    metadata = {
        'num_features': num_features,
        'num_samples': all_data.shape[0],
        'num_runs': len(all_boundaries)
    }
    np.save(save_pth + "all_data_metadata.npy", metadata)
    
    print(f"Saved {all_data.shape[0]} samples to all_data.npy")
    print(f"Saved {all_labels.shape[0]} contact labels (left + right leg) to all_labels.npy")
    print(f"Saved {all_foot_velocities.shape[0]} foot velocity norms (LEFT leg only) to all_foot_velocities.npy")
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
    
    print("Using configuration:")
    print(f"  CSV folder: {config['csv_folder']}")
    print(f"  Save path: {config['save_path']}")
    print(f"  Train ratio: {config['train_ratio']}")
    print(f"  Val ratio: {config['val_ratio']}")
    
    csv2numpy_split(config['csv_folder'], config['save_path'], 
                    config['train_ratio'], config['val_ratio'])


if __name__ == '__main__':
    main()
