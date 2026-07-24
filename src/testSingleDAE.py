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

from contact_cnn import DenoisingTCNAutoencoder
from trainEncoder import VariationalTCNAutoencoder


# Select one trajectory whose first cmd_vel_x is in this inclusive range.
TEST_CMD_VEL_X_WINDOW = (1.3, 1.8)  # [min, max] in m/s
RANDOM_SEED = 28

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


def make_model(config, num_features):
    model_class = DenoisingTCNAutoencoder if config['encoder_type'] == 'DAE' else VariationalTCNAutoencoder
    return model_class(
        window_size=config['window_size'], num_features=num_features,
        tcn_num_channels=config.get('tcn_num_channels', 64),
        tcn_kernel_size=config.get('tcn_kernel_size', 3),
        tcn_num_blocks=config.get('tcn_num_blocks', 3),
        tcn_dropout=config.get('tcn_dropout', 0.2),
    )


def latest_checkpoint(encoder_type):
    logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logsEncoder')
    checkpoint_name = 'dae_best_val_mse.pt' if encoder_type == 'DAE' else 'vae_best_val_loss.pt'
    for run_dir in sorted(glob.glob(os.path.join(logs_root, '*')), key=os.path.getmtime, reverse=True):
        checkpoint = os.path.join(run_dir, checkpoint_name)
        if os.path.isfile(checkpoint):
            return checkpoint
    raise FileNotFoundError(f'No {checkpoint_name} found in {logs_root}')


def run_trajectory(model, trajectory, window_size, batch_size, device):
    """Reconstruct every valid sliding window and return its raw-window MSE."""
    features = make_features(trajectory)
    num_windows = len(features) - window_size + 1
    reconstruction_mse = []

    with torch.no_grad():
        for first in range(0, num_windows, batch_size):
            last = min(first + batch_size, num_windows)
            windows = np.stack([features[index:index + window_size] for index in range(first, last)])
            raw_windows = torch.from_numpy(windows).float().to(device)
            reconstruction = model(raw_windows)
            if isinstance(reconstruction, tuple):
                reconstruction = reconstruction[0]
            reconstruction_mse.append((reconstruction - raw_windows).square().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(reconstruction_mse)


def get_training_window_starts(data_folder, window_size, config):
    """Recreate train.py's run-level split and return its valid window starts."""
    boundaries = np.load(os.path.join(data_folder, 'all_data_boundaries.npy'))
    run_starts = np.concatenate(([0], boundaries[:-1]))
    valid_run_ids = np.flatnonzero(boundaries - run_starts >= window_size)
    if len(valid_run_ids) < 2:
        raise ValueError(f'Need at least 2 runs with {window_size} samples for the training split.')

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
    """Run raw windows through the DAE and return their final TCN latents."""
    backbone = model.tcn_backbone
    latents = []
    hook = backbone.register_forward_hook(lambda _, __, output: latents.append(output[:, :, -1].detach().cpu()))
    offsets = np.arange(window_size)
    try:
        with torch.no_grad():
            for first in range(0, len(window_starts), batch_size):
                starts = window_starts[first:first + batch_size]
                windows = raw_data[starts[:, None] + offsets]
                model(torch.from_numpy(windows).float().to(device))
    finally:
        hook.remove()
    return torch.cat(latents).numpy()


def l2_normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def save_knn_umap(training_latents, trajectory_latents, knn_distances, ood_mask, knn_k, threshold, output_path):
    """Plot sampled training latents and every selected trajectory window in one UMAP."""
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
    figure.colorbar(points, ax=axis, label=f'{knn_k}-th nearest-neighbor latent distance')
    axis.set(title=f'Training-window UMAP (OOD threshold: {threshold:.4f})', xlabel='UMAP 1', ylabel='UMAP 2')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Run one selected CSV trajectory through the configured autoencoder.')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
    args = parser.parse_args()

    with open(args.config_name) as config_file:
        config = yaml.safe_load(config_file)
    config['encoder_type'] = config.get('encoder_type', 'DAE').upper()
    if config['encoder_type'] not in {'DAE', 'VAE'}:
        raise ValueError("encoder_type must be 'DAE' or 'VAE'.")
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
    checkpoint_path = latest_checkpoint(config['encoder_type'])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval().to(device)

    reconstruction_mse = run_trajectory(
        model, trajectory, config['window_size'], config.get('test_batch_size', config['batch_size']), device
    )

    if RUN_KNN_UMAP:
        data_folder = config['data_folder'] if os.path.isabs(config['data_folder']) else os.path.join(project_root, config['data_folder'])
        training_data = np.load(os.path.join(data_folder, 'all_data.npy'))
        training_starts = get_training_window_starts(data_folder, config['window_size'], config)
        training_latents = l2_normalize(collect_final_tcn_latents(
            model, training_data, training_starts, config['window_size'], config['batch_size'], device
        ))
        trajectory_starts = np.arange(len(trajectory) - config['window_size'] + 1)
        trajectory_latents = l2_normalize(collect_final_tcn_latents(
            model, make_features(trajectory), trajectory_starts, config['window_size'], config['batch_size'], device
        ))
        from sklearn.neighbors import NearestNeighbors
        if len(training_latents) < 2:
            raise ValueError('Need at least two training windows for KNN OOD detection.')
        knn_k = min(KNN_K, len(training_latents) - 1)
        neighbors = NearestNeighbors(n_neighbors=knn_k).fit(training_latents)
        knn_distances = neighbors.kneighbors(trajectory_latents, return_distance=True)[0][:, -1]
        train_distances = NearestNeighbors(n_neighbors=knn_k + 1).fit(training_latents).kneighbors(
            training_latents, return_distance=True
        )[0][:, -1]
        ood_threshold = np.quantile(train_distances, OOD_ID_PERCENTILE)
        ood_mask = knn_distances > ood_threshold
        umap_output_path = os.path.join(os.path.dirname(checkpoint_path), f'trajectory_knn_umap_seed{RANDOM_SEED}.png')
        save_knn_umap(training_latents, trajectory_latents, knn_distances, ood_mask, knn_k, ood_threshold, umap_output_path)

    print(f'CSV: {csv_path}')
    print(f'Trajectory: {run_index}, start cmd_vel_x: {start_cmd_vel:.3f} m/s, selected range: [{low}, {high}]')
    print(f'Random seed: {RANDOM_SEED}, trajectory samples: {len(trajectory)}, evaluated windows: {len(reconstruction_mse)}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Reconstruction MSE: mean {reconstruction_mse.mean():.8f}, max {reconstruction_mse.max():.8f}')
    if RUN_KNN_UMAP:
        print(f'KNN reference windows: {len(training_latents)}, K: {knn_k}')
        print(f'{knn_k}-th nearest-neighbor latent distance: mean {knn_distances.mean():.6f}, max {knn_distances.max():.6f}')
        print(f'ID threshold ({OOD_ID_PERCENTILE:.0%} training quantile): {ood_threshold:.6f}')
        print(f'OOD trajectory windows: {ood_mask.sum()} / {len(ood_mask)}')
        print(f'Training/trajectory KNN UMAP: {umap_output_path}')


if __name__ == '__main__':
    main()
