import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
import yaml

CANONICAL_LEGS = ('left', 'right')

# Edit these when converting a different Isaac CSV dataset.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FOLDER = os.path.join(PROJECT_ROOT, 'Data/CSVFiles/')
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'Data/NumpyFiles/')

ISAAC_SCHEMA = {
    'time': 'timestamp',
    'command_velocity': 'cmd_vel_x',
    'imu_acceleration': ('acc_body_x', 'acc_body_y', 'acc_body_z'),
    'imu_angular_rate': ('gyro_body_x', 'gyro_body_y', 'gyro_body_z'),
    'joint_position_pattern': 'joint_pos_{joint}',
    'joint_velocity_pattern': 'joint_vel_{joint}',
    'joint_torque_pattern': 'joint_torque_{joint}',
    'foot_position_pattern': 'fk_{leg}_foot_pos_{axis}',
    'foot_velocity_pattern': 'fk_{leg}_foot_vel_{axis}',
    'contact_columns': {'left': 'lfoot-contact', 'right': 'rfoot-contact'},
    'world_velocity': ('vel_x', 'vel_y', 'vel_z'),
    'quaternion': ('quat_w', 'quat_i', 'quat_j', 'quat_k'),
}


def resolve_active_legs(value):
    value = str(value).lower()
    if value == 'both':
        return CANONICAL_LEGS
    if value in CANONICAL_LEGS:
        return (value,)
    raise ValueError("active_legs must be 'left', 'right', or 'both'.")


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


def environment_number_from_filename(data_name):
    """Extract an environment ID from a CSV filename.

    Supported examples include ``robotstate_0.csv``, ``run_env5.csv``, and
    ``run_environment_12.csv``.  For names without an ``env`` marker, the
    final underscore-delimited number is treated as the environment ID.
    """
    stem = os.path.splitext(os.path.basename(data_name))[0]
    match = re.search(r'(?:^|_)(?:env|environment)_?\(?(\d+)\)?(?:_|$)', stem, re.IGNORECASE)
    if match is None:
        match = re.search(r'_(\d+)$', stem)
    return int(match.group(1)) if match else None


def csv2numpy_split(data_pth, save_pth, cmd_vel_x_windows=((0.0, 2.0),),
                    ood_feature='cmd_vel', environment_windows=((0, 0),), legs=CANONICAL_LEGS,
                    schema=ISAAC_SCHEMA):
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
    
    Output:
    - all_data.npy: all data concatenated with body-frame IMU acceleration and
      gyroscope readings, left- then right-leg features, then shared cmd_vel_x
      (57 raw features total when both legs are active)
    - all_labels.npy: left/right foot-contact labels (shape: N x 2, binary 0/1)
    - all_data_boundaries.npy: indices marking end of each run (CRITICAL for preventing data leakage)
    - all_data_metadata.npy: metadata dict with num_features (SOURCE OF TRUTH for network architecture)
    - all_body_velocities.npy: body-frame body velocities repeated per leg
      (shape: N x 2 x 3), so each leg is supervised only during its contact
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
    legs = tuple(legs)
    all_labels = np.zeros((0, len(legs)))
    all_body_velocities = np.zeros((0, len(legs), 3))
    num_features = None  # Will be set from cur_data.shape[1] after first run
    
    # Track boundaries between different runs to prevent window bleeding
    # CRITICAL: These boundaries are used in train.py to split by RUNS (not windows)
    # This prevents data leakage from overlapping sliding windows
    all_boundaries = []
    
    joint_names_by_leg = {
        leg: [
            f'{leg}_hip_pitch_joint', f'{leg}_hip_roll_joint', f'{leg}_hip_yaw_joint',
            f'{leg}_knee_joint', f'{leg}_ankle_pitch_joint', f'{leg}_ankle_roll_joint',
        ]
        for leg in legs
    }
    
    # Process all CSV files in the folder
    for data_name in sorted(glob.glob(data_pth + '*.csv')):
        if ood_feature == 'environment':
            environment_number = environment_number_from_filename(data_name)
            if environment_number is None:
                raise ValueError(
                    f'Cannot determine environment number from CSV filename: {data_name}. '
                    'Use a name ending in _<number>.csv or containing env<number>.')
            if not any(low <= environment_number <= high for low, high in environment_windows):
                print(f"Skipping {data_name} (environment {environment_number} outside {environment_windows})")
                continue

        print("loading... ", data_name)
        
        # Load CSV data
        df = pd.read_csv(data_name)
        
        # Detect run boundaries within the file by finding time resets
        # Time resets to ~0.02 indicate a new run
        time_column = schema['time']
        time_col = df[time_column].values if time_column in df.columns else None
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
            
            # When filtering by command velocity, keep runs whose maximum
            # cmd_vel_x belongs to any configured window.
            command_velocity_column = schema['command_velocity']
            if ood_feature == 'cmd_vel' and command_velocity_column in df_run.columns:
                max_cmd_vel = df_run[command_velocity_column].max()
                if not any(low <= max_cmd_vel <= high for low, high in cmd_vel_x_windows):
                    # print(f"  Skipping run {run_idx} (max cmd_vel_x={max_cmd_vel:.2f} outside {cmd_vel_x_windows})")
                    continue
            
            # Extract IMU data in body frame
            imu_acc = df_run[list(schema['imu_acceleration'])].values
            imu_omega = df_run[list(schema['imu_angular_rate'])].values

            # Extract command velocity - 1 value
            cmd_vel = df_run[[command_velocity_column]].values

            leg_features = []
            for leg in legs:
                joint_names = joint_names_by_leg[leg]
                q = df_run[[schema['joint_position_pattern'].format(joint=name) for name in joint_names]].values
                qd = df_run[[schema['joint_velocity_pattern'].format(joint=name) for name in joint_names]].values
                position = df_run[[schema['foot_position_pattern'].format(leg=leg, axis=axis) for axis in 'xyz']].values
                velocity = df_run[[schema['foot_velocity_pattern'].format(leg=leg, axis=axis) for axis in 'xyz']].values
                torque = df_run[[schema['joint_torque_pattern'].format(joint=name) for name in joint_names]].values
                torque_mse = np.sum(torque ** 2, axis=1, keepdims=True)
                leg_features.extend((q, qd, position, velocity, torque, torque_mse))

            # Feature order: body-frame IMU acceleration/angular rate, leg blocks,
            # then shared command velocity.
            cur_data = np.concatenate([imu_acc, imu_omega, *leg_features, cmd_vel], axis=1)
            
            # Initialize all_data and capture num_features from actual data shape
            if num_features is None:
                num_features = cur_data.shape[1]
                all_data = np.zeros((0, num_features))
                print(f"\nAuto-detected {num_features} features from data shape")

            # -----------------------------

            # Output extraction
            
            contacts = df_run[[schema['contact_columns'][leg] for leg in legs]].values
            if schema.get('contacts_positive'):
                contacts = (contacts > 0).astype(int)
            else:
                contacts = contacts.astype(int)
            
            body_velocity_world = df_run[list(schema['world_velocity'])].values
            body_quaternion = df_run[list(schema['quaternion'])].values
            rotation_body_to_world = quaternion_to_rotation_matrix(body_quaternion)
            body_velocities = np.einsum(
                'nij,nj->ni', rotation_body_to_world.transpose(0, 2, 1), body_velocity_world
            )
            body_velocities = np.repeat(body_velocities[:, np.newaxis, :], len(legs), axis=1)


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
        'legs': list(legs),
        'num_features': num_features,
        'num_velocity_targets': all_body_velocities.shape[1],
        'num_samples': all_data.shape[0],
        'num_runs': len(all_boundaries)
    }
    np.save(save_pth + "all_data_metadata.npy", metadata)
    
    print(f"Saved {all_data.shape[0]} samples to all_data.npy")
    print(f"Saved {all_labels.shape} left/right foot-contact labels to all_labels.npy")
    print(f"Saved {all_body_velocities.shape} body-frame [vx/vy/vz] targets to all_body_velocities.npy")
    print(f"Saved {len(all_boundaries)} run boundaries to all_data_boundaries.npy")
    print(f"Saved metadata (num_features={num_features}) to all_data_metadata.npy")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to numpy.')
    parser.add_argument('--config_name', type=str,
                        default=os.path.join(PROJECT_ROOT, 'config/network_params.yaml'))
    args = parser.parse_args()
    
    with open(args.config_name) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    ood_feature = config.get('ood_feature', 'cmd_vel')
    legs = resolve_active_legs(config.get('active_legs', 'both'))
    if ood_feature not in ('cmd_vel', 'environment'):
        raise ValueError("ood_feature must be either 'cmd_vel' or 'environment'")
    window_key = 'cmd_vel_x_windows' if ood_feature == 'cmd_vel' else 'environment_windows'
    windows = config.get(window_key, [[0.0, 2.0]] if ood_feature == 'cmd_vel' else [[0, 0]])
    if not windows or not all(isinstance(window, (list, tuple)) and len(window) == 2 and window[0] <= window[1]
                              for window in windows):
        raise ValueError(f'{window_key} must be a non-empty list of [min, max] windows with min <= max')
    
    print("Using configuration:")
    print(f"  CSV folder: {CSV_FOLDER}")
    print(f"  Save path: {DATA_FOLDER}")
    print(f"  OOD feature: {ood_feature}")
    print(f"  Active legs: {', '.join(legs)}")
    print(f"  {'Command-velocity' if ood_feature == 'cmd_vel' else 'Environment'} windows: {windows}")
    
    csv2numpy_split(CSV_FOLDER, DATA_FOLDER,
                    cmd_vel_x_windows=windows if ood_feature == 'cmd_vel' else (),
                    ood_feature=ood_feature,
                    environment_windows=windows if ood_feature == 'environment' else (), legs=legs)


if __name__ == '__main__':
    main()
