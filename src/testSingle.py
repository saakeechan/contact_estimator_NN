import argparse
import glob
import os
import sys

sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from contact_cnn import AttentionTCN, ContactCNNWithNormalization, TCN, contact_cnn


# Select one trajectory whose first cmd_vel_x is in this inclusive range.
TEST_CMD_VEL_X_WINDOW = (2.1, 2.2)  # [min, max] in m/s
RANDOM_SEED = 21


def find_random_trajectory(csv_folder, window_size, cmd_vel_x_window, rng):
    """Read CSVs in random order and return one matching timestamp-delimited trajectory."""
    low, high = cmd_vel_x_window
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {csv_folder}')

    for csv_path in rng.permutation(csv_files):
        df = pd.read_csv(csv_path)
        if 'cmd_vel_x' not in df.columns:
            continue

        timestamps = df['timestamp'].to_numpy()
        boundaries = [0] + (np.flatnonzero(np.abs(np.diff(timestamps)) > 0.025) + 1).tolist() + [len(df)]
        for run_index in rng.permutation(len(boundaries) - 1):
            trajectory = df.iloc[boundaries[run_index]:boundaries[run_index + 1]].reset_index(drop=True)
            if len(trajectory) < window_size:
                continue

            start_cmd_vel = trajectory['cmd_vel_x'].iloc[0]
            if low <= start_cmd_vel <= high:
                return trajectory, csv_path, run_index, start_cmd_vel

    raise ValueError(f'No trajectory with at least {window_size} samples starts with cmd_vel_x in [{low}, {high}]')


def make_features(trajectory):
    joint_names = (
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    )
    imu_acc = trajectory[['acc_body_x', 'acc_body_y', 'acc_body_z']].to_numpy()
    imu_omega = trajectory[['gyro_body_x', 'gyro_body_y', 'gyro_body_z']].to_numpy()
    q = trajectory[['joint_pos_' + name for name in joint_names]].to_numpy()
    qd = trajectory[['joint_vel_' + name for name in joint_names]].to_numpy()
    foot_position = trajectory[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z']].to_numpy()
    foot_velocity = trajectory[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z']].to_numpy()
    torque = trajectory[['joint_torque_' + name for name in joint_names]].to_numpy()
    torque_mse = np.sum(torque ** 2, axis=1, keepdims=True)
    return np.concatenate((imu_acc, imu_omega, q, qd, foot_position, foot_velocity, torque, torque_mse), axis=1)


def make_foot_velocity(trajectory):
    position = trajectory[['lfoot_pos_x', 'lfoot_pos_y', 'lfoot_pos_z']].to_numpy()
    dt = np.diff(trajectory['timestamp'].to_numpy())[:, None]
    velocity = np.diff(position, axis=0) / dt
    return np.vstack((velocity, velocity[-1]))


def make_model(config, num_features):
    architecture = config.get('model_architecture', 'vanilla_cnn').lower()
    if architecture == 'attention_tcn':
        base_model = AttentionTCN(
            window_size=config['window_size'], num_features=num_features,
            d_model=config.get('attention_d_model', 64), num_heads=config.get('attention_num_heads', 4),
            tcn_num_channels=config.get('tcn_num_channels', 64), tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5), tcn_dropout=config.get('tcn_dropout', 0.2),
        )
    elif architecture == 'tcn':
        base_model = TCN(
            window_size=config['window_size'], num_features=num_features,
            tcn_num_channels=config.get('tcn_num_channels', 64), tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5), tcn_dropout=config.get('tcn_dropout', 0.2),
        )
    elif architecture == 'vanilla_cnn':
        base_model = contact_cnn(window_size=config['window_size'], num_features=num_features)
    else:
        raise ValueError(f'Unknown model_architecture: {architecture}')
    return ContactCNNWithNormalization(base_model)


def latest_checkpoint():
    logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    for run_dir in sorted(glob.glob(os.path.join(logs_root, '*')), key=os.path.getmtime, reverse=True):
        checkpoint = os.path.join(run_dir, 'model_best_val_velocity.pt')
        if os.path.isfile(checkpoint):
            return checkpoint
    raise FileNotFoundError(f'No model_best_val_velocity.pt found in {logs_root}')


def run_trajectory(model, trajectory, window_size, batch_size, device):
    """Infer every valid sliding window in one trajectory and align final-timestep targets."""
    features = make_features(trajectory)
    num_windows = len(features) - window_size + 1
    predicted_velocity, predicted_variance, contact_probability = [], [], []

    with torch.no_grad():
        for first in range(0, num_windows, batch_size):
            last = min(first + batch_size, num_windows)
            windows = np.stack([features[index:index + window_size] for index in range(first, last)])
            _, velocity, _, variance, contact_logit = model(torch.from_numpy(windows).float().to(device), return_sequence=False)
            predicted_velocity.append(velocity[:, 0].cpu().numpy())
            predicted_variance.append(variance[:, 0].cpu().numpy())
            contact_probability.append(torch.sigmoid(contact_logit[:, 0]).cpu().numpy())

    final_indices = np.arange(window_size - 1, len(trajectory))
    return (
        final_indices,
        np.concatenate(predicted_velocity),
        np.concatenate(predicted_variance),
        np.concatenate(contact_probability),
        trajectory['lfoot-contact'].to_numpy()[final_indices],
        make_foot_velocity(trajectory)[final_indices],
    )


def save_velocity_plot(time, predicted, ground_truth, contact, output_path):
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    contact_changes = np.diff(np.concatenate(([0], contact == 1, [0])))
    for axis_index, axis in enumerate(axes):
        for start, end in zip(np.where(contact_changes == 1)[0], np.where(contact_changes == -1)[0]):
            axis.axvspan(time[start], time[end - 1], color='red', alpha=0.2)
        axis.plot(time, ground_truth[:, axis_index], color='black', label='Ground truth')
        axis.plot(time, predicted[:, axis_index], color='tab:blue', linestyle='--', label='Predicted')
        axis.set_ylabel(f'v{"xyz"[axis_index]} (m/s)')
        axis.grid(True, alpha=0.3)
        axis.legend(loc='upper right')
    axes[0].set_title('Whole-trajectory left-foot velocity: prediction vs ground truth')
    axes[-1].set_xlabel('Trajectory time (s)')
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Run one selected CSV trajectory through the contact network.')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
    args = parser.parse_args()

    with open(args.config_name) as config_file:
        config = yaml.safe_load(config_file)
    low, high = TEST_CMD_VEL_X_WINDOW
    if low > high:
        raise ValueError('TEST_CMD_VEL_X_WINDOW must be (min, max) with min <= max')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_folder = config['csv_folder'] if os.path.isabs(config['csv_folder']) else os.path.join(project_root, config['csv_folder'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trajectory, csv_path, run_index, start_cmd_vel = find_random_trajectory(
        csv_folder, config['window_size'], TEST_CMD_VEL_X_WINDOW, np.random.default_rng(RANDOM_SEED)
    )

    model = make_model(config, make_features(trajectory).shape[1])
    checkpoint_path = latest_checkpoint()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval().to(device)

    indices, predicted, variance, contact_probability, contact, ground_truth = run_trajectory(
        model, trajectory, config['window_size'], config.get('test_batch_size', config['batch_size']), device
    )
    contact_mask = contact == 1
    mae = np.abs(predicted[contact_mask] - ground_truth[contact_mask]).mean() if contact_mask.any() else float('nan')
    mean_variance = variance[contact_mask].mean(axis=0) if contact_mask.any() else np.full(3, np.nan)
    output_path = os.path.join(os.path.dirname(checkpoint_path), f'trajectory_velocity_comparison_seed{RANDOM_SEED}.png')
    save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted, ground_truth, contact, output_path)

    print(f'CSV: {csv_path}')
    print(f'Trajectory: {run_index}, start cmd_vel_x: {start_cmd_vel:.3f} m/s, selected range: [{low}, {high}]')
    print(f'Random seed: {RANDOM_SEED}, trajectory samples: {len(trajectory)}, evaluated windows: {len(indices)}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Contact-final windows: {contact_mask.sum()} / {len(contact)}')
    print(f'Contact-masked velocity MAE: {mae:.6f}')
    print(f'Mean predicted variance on contact [vx, vy, vz]: {mean_variance}')
    print(f'Whole-trajectory velocity plot: {output_path}')


if __name__ == '__main__':
    main()
