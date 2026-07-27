import argparse
import glob
import os
import sys
import warnings

sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from contact_cnn import AttentionTCN, ContactCNNWithNormalization, TCN, contact_cnn
from utils.csv2numpyV1 import quaternion_to_rotation_matrix


# Select one trajectory whose first cmd_vel_x is in this inclusive range.
TEST_CMD_VEL_X_WINDOW = (2.8, 3.0)  # [min, max] in m/s
RANDOM_SEED = 900

RUN_KNN_UMAP = True
KNN_K = 50
OOD_ID_PERCENTILE = 0.95
UMAP_TRAIN_MAX = 5000


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
    # imu_acc = trajectory[['acc_body_x', 'acc_body_y', 'acc_body_z']].to_numpy()
    # imu_omega = trajectory[['gyro_body_x', 'gyro_body_y', 'gyro_body_z']].to_numpy()
    q = trajectory[['joint_pos_' + name for name in joint_names]].to_numpy()
    qd = trajectory[['joint_vel_' + name for name in joint_names]].to_numpy()
    foot_position = trajectory[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z']].to_numpy()
    foot_velocity = trajectory[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z']].to_numpy()
    torque = trajectory[['joint_torque_' + name for name in joint_names]].to_numpy()
    torque_mse = np.sum(torque ** 2, axis=1, keepdims=True)
    cmd_vel_x = trajectory[['cmd_vel_x']].to_numpy()
    return np.concatenate((q, qd, foot_position, foot_velocity, torque, torque_mse, cmd_vel_x), axis=1)


def make_body_velocity(trajectory):
    velocity_world = trajectory[['vel_x', 'vel_y', 'vel_z']].to_numpy()
    quaternion = trajectory[['quat_w', 'quat_i', 'quat_j', 'quat_k']].to_numpy()
    return np.einsum('nij,nj->ni', quaternion_to_rotation_matrix(quaternion).transpose(0, 2, 1), velocity_world)


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
            natpn_flow_layers=config.get('natpn_flow_layers', 8),
            natpn_certainty_budget=config.get('natpn_certainty_budget', 'normal'),
        )
    elif architecture == 'vanilla_cnn':
        base_model = contact_cnn(window_size=config['window_size'], num_features=num_features)
    else:
        raise ValueError(f'Unknown model_architecture: {architecture}')
    return ContactCNNWithNormalization(base_model)


def latest_checkpoint():
    logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    for run_dir in sorted(glob.glob(os.path.join(logs_root, '*')), key=os.path.getmtime, reverse=True):
        for filename in ('model_natpn_finetuned.pt', 'model_best_val_velocity.pt'):
            checkpoint = os.path.join(run_dir, filename)
            if os.path.isfile(checkpoint):
                return checkpoint
    raise FileNotFoundError(f'No NatPN-finetuned or best-velocity checkpoint found in {logs_root}')


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
        make_body_velocity(trajectory)[final_indices],
    )


@torch.no_grad()
def natpn_uncertainty_for_window(model, features, window_start, window_size, device):
    """Return NatPN aleatoric and epistemic variance for one final-timestep window."""
    window = torch.from_numpy(features[window_start:window_start + window_size]).float().unsqueeze(0).to(device)
    *_, posteriors = model(window, return_sequence=False, return_posteriors=True)
    aleatoric = np.array([
        (posterior.beta / (posterior.alpha - 1.0).clamp_min(1e-6)).item()
        for posterior in posteriors
    ])
    epistemic = np.array([
        (posterior.beta / ((posterior.alpha - 1.0).clamp_min(1e-6) * posterior.lambd)).item()
        for posterior in posteriors
    ])
    return aleatoric, epistemic


def get_training_window_starts(data_folder, window_size, config):
    """Recreate train.py's run-level split and return its valid window starts."""
    boundaries = np.load(os.path.join(data_folder, 'all_data_boundaries.npy'))
    run_starts = np.concatenate(([0], boundaries[:-1]))
    valid_run_ids = np.flatnonzero(boundaries - run_starts >= window_size)
    if len(valid_run_ids) < 3:
        raise ValueError(f'Need at least 3 runs with {window_size} samples for the training split.')

    run_ids = valid_run_ids.copy()
    if config.get('shuffle', True):
        np.random.RandomState(config.get('random_seed', 42)).shuffle(run_ids)
    train_count = int(config.get('train_ratio', 0.7) * len(run_ids))
    if train_count == 0:
        train_count = 1

    return np.concatenate([
        np.arange(run_starts[run_id], boundaries[run_id] - window_size + 1)
        for run_id in run_ids[:train_count]
    ])


def collect_final_tcn_latents(model, raw_data, window_starts, window_size, batch_size, device):
    """Run raw windows through the wrapper and return their final TCN latents."""
    backbone = model.base_model.tcn_backbone
    latents = []
    hook = backbone.register_forward_hook(lambda _, __, output: latents.append(output[:, :, -1].detach().cpu()))
    offsets = np.arange(window_size)
    try:
        with torch.no_grad():
            for first in range(0, len(window_starts), batch_size):
                starts = window_starts[first:first + batch_size]
                windows = raw_data[starts[:, None] + offsets]
                model(torch.from_numpy(windows).float().to(device), return_sequence=False)
    finally:
        hook.remove()
    return torch.cat(latents).numpy()


def l2_normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def save_knn_umap(training_latents, trajectory_latents, knn_distances, ood_mask, knn_k, threshold, output_path):
    """Plot sampled training latents and the selected contact trajectory windows in one UMAP."""
    from umap import UMAP

    rng = np.random.default_rng(RANDOM_SEED)
    train_indices = rng.choice(len(training_latents), size=min(UMAP_TRAIN_MAX, len(training_latents)), replace=False)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='n_jobs value .* overridden.*')
        reducer = UMAP(n_components=2, random_state=RANDOM_SEED)
        train_embedding = reducer.fit_transform(training_latents[train_indices])
        trajectory_embedding = reducer.transform(trajectory_latents)

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(train_embedding[:, 0], train_embedding[:, 1], s=4, color='lightgray', alpha=0.45, label='Training windows')
    points = axis.scatter(
        trajectory_embedding[:, 0], trajectory_embedding[:, 1], c=knn_distances,
        cmap='viridis', s=18, edgecolors='black', linewidths=0.2, label='Selected trajectory windows'
    )
    if ood_mask.any():
        axis.scatter(
            trajectory_embedding[ood_mask, 0], trajectory_embedding[ood_mask, 1],
            s=44, facecolors='none', edgecolors='red', linewidths=1.2, label='OOD window'
        )
    figure.colorbar(points, ax=axis, label=f'Mean distance to {knn_k} nearest latent neighbors')
    axis.set(title=f'Training-window UMAP (OOD threshold: {threshold:.4f})', xlabel='UMAP 1', ylabel='UMAP 2')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)


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
    axes[0].set_title('Whole-trajectory body-frame body velocity: prediction vs ground truth')
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

    trajectory_features = make_features(trajectory)
    model = make_model(config, trajectory_features.shape[1])
    checkpoint_path = latest_checkpoint()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval().to(device)

    indices, predicted, variance, contact_probability, contact, ground_truth = run_trajectory(
        model, trajectory, config['window_size'], config.get('test_batch_size', config['batch_size']), device
    )
    contact_mask = contact == 1
    candidate_positions = np.flatnonzero(contact_mask)
    random_contact_window = len(candidate_positions) > 0
    if not random_contact_window:
        candidate_positions = np.arange(len(contact_mask))
    random_position = int(np.random.default_rng(RANDOM_SEED).choice(candidate_positions))
    random_aleatoric, random_epistemic = natpn_uncertainty_for_window(
        model, trajectory_features, random_position, config['window_size'], device
    )
    mae = np.abs(predicted[contact_mask] - ground_truth[contact_mask]).mean() if contact_mask.any() else float('nan')
    mean_variance = variance[contact_mask].mean(axis=0) if contact_mask.any() else np.full(3, np.nan)
    output_path = os.path.join(os.path.dirname(checkpoint_path), f'trajectory_velocity_comparison_seed{RANDOM_SEED}.png')
    save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted, ground_truth, contact, output_path)

    if RUN_KNN_UMAP:
        data_folder = config['data_folder'] if os.path.isabs(config['data_folder']) else os.path.join(project_root, config['data_folder'])
        training_data = np.load(os.path.join(data_folder, 'all_data.npy'))
        training_starts = get_training_window_starts(data_folder, config['window_size'], config)
        training_latents = l2_normalize(collect_final_tcn_latents(
            model, training_data, training_starts, config['window_size'], config['batch_size'], device
        ))
        trajectory_starts = np.arange(len(trajectory) - config['window_size'] + 1)
        trajectory_latents = l2_normalize(collect_final_tcn_latents(
            model, trajectory_features, trajectory_starts, config['window_size'], config['batch_size'], device
        ))
        if len(trajectory_latents) != len(contact_mask):
            raise RuntimeError('Trajectory latent/contact-window alignment failed.')
        trajectory_latents = trajectory_latents[contact_mask]
        if len(trajectory_latents) == 0:
            raise ValueError('Selected trajectory has no contact-final windows for KNN evaluation.')
        from sklearn.neighbors import NearestNeighbors
        if len(training_latents) < 2:
            raise ValueError('Need at least two training windows for KNN OOD detection.')
        knn_k = min(KNN_K, len(training_latents) - 1)
        neighbors = NearestNeighbors(n_neighbors=knn_k).fit(training_latents)
        knn_distances = neighbors.kneighbors(trajectory_latents, return_distance=True)[0].mean(axis=1)
        train_distances = NearestNeighbors(n_neighbors=knn_k + 1).fit(training_latents).kneighbors(
            training_latents, return_distance=True
        )[0][:, 1:].mean(axis=1)
        ood_threshold = np.quantile(train_distances, OOD_ID_PERCENTILE)
        ood_mask = knn_distances > ood_threshold
        umap_output_path = os.path.join(os.path.dirname(checkpoint_path), f'contact_trajectory_knn_umap_seed{RANDOM_SEED}.png')
        save_knn_umap(training_latents, trajectory_latents, knn_distances, ood_mask, knn_k, ood_threshold, umap_output_path)

    print(f'CSV: {csv_path}')
    print(f'Trajectory: {run_index}, start cmd_vel_x: {start_cmd_vel:.3f} m/s, selected range: [{low}, {high}]')
    print(f'Random seed: {RANDOM_SEED}, trajectory samples: {len(trajectory)}, evaluated windows: {len(indices)}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Contact-final windows: {contact_mask.sum()} / {len(contact)}')
    print(f'Contact-masked velocity MAE: {mae:.6f}')
    print(f'Mean predicted variance on contact [vx, vy, vz]: {mean_variance}')
    print(
        f'Random {"contact " if random_contact_window else ""}window: final sample {indices[random_position]} | '
        f'aleatoric variance [vx, vy, vz]: {random_aleatoric} | '
        f'epistemic variance [vx, vy, vz]: {random_epistemic}'
    )
    print(f'Whole-trajectory velocity plot: {output_path}')
    if RUN_KNN_UMAP:
        print(f'KNN reference windows: {len(training_latents)}, K: {knn_k}')
        print(f'Mean distance to {knn_k} nearest latent neighbors: mean {knn_distances.mean():.6f}, max {knn_distances.max():.6f}')
        print(f'ID threshold ({OOD_ID_PERCENTILE:.0%} training quantile): {ood_threshold:.6f}')
        print(f'OOD contact trajectory windows: {ood_mask.sum()} / {len(ood_mask)}')
        print(f'Training/contact-trajectory KNN UMAP: {umap_output_path}')


if __name__ == '__main__':
    main()
