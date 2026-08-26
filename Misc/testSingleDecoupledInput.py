import argparse
import glob
import json
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

from contact_cnn import ContactCNNWithNormalization, TCN, contact_cnn
from tests.base_test import BaseSingleTest
from utils.csv2numpyV1 import quaternion_to_rotation_matrix
from utils.ood_selection import csv_matches_environment_windows, validate_ood_selection


# Select one trajectory whose first cmd_vel_x is in this inclusive range.
TEST_CMD_VEL_X_WINDOW = (1.0, 2.0)  # [min, max] in m/s
RANDOM_SEED = 203

RUN_KNN_UMAP = True
KNN_K = 20
OOD_ID_PERCENTILE = 0.95
UMAP_TRAIN_MAX = 5000


def find_random_trajectory(csv_folder, window_size, cmd_vel_x_window, rng,
                           ood_feature='cmd_vel', environment_windows=()):
    """Read CSVs in random order and return one trajectory matching the configured OOD feature."""
    low, high = cmd_vel_x_window
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {csv_folder}')
    if ood_feature == 'environment':
        csv_files = [path for path in csv_files if csv_matches_environment_windows(path, environment_windows)[0]]
        if not csv_files:
            raise ValueError(f'No CSV files in {csv_folder} match environment windows {environment_windows}')

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
            if ood_feature == 'environment' or low <= start_cmd_vel <= high:
                return trajectory, csv_path, run_index, start_cmd_vel

    selection = f'environment windows {environment_windows}' if ood_feature == 'environment' else f'cmd_vel_x in [{low}, {high}]'
    raise ValueError(f'No trajectory with at least {window_size} samples matches {selection}')


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
    if architecture == 'tcn':
        base_model = TCN(
            window_size=config['window_size'], num_features=num_features,
            tcn_num_channels=config.get('tcn_num_channels', 64), tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5), tcn_dropout=config.get('tcn_dropout', 0.2),
            natpn_flow_layers=config.get('natpn_flow_layers', 8),
            natpn_certainty_budget=config.get('natpn_certainty_budget', 'normal'),
            natpn_evidence_source=config.get('natpn_evidence_source', 'task'),
            input_natpn_checkpoint=config.get('input_natpn_checkpoint'),
            input_epistemic_scale=config.get('input_epistemic_scale', 1.0),
        )
    elif architecture == 'vanilla_cnn':
        base_model = contact_cnn(window_size=config['window_size'], num_features=num_features)
    else:
        raise ValueError(f'Unknown model_architecture: {architecture}')
    return ContactCNNWithNormalization(base_model)


def latest_checkpoint(num_features, evidence_source):
    """Select the newest task checkpoint matching the active input contract and NatPN mode."""
    logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logsNatPN')
    for run_dir in sorted(glob.glob(os.path.join(logs_root, '*')), key=os.path.getmtime, reverse=True):
        for filename in ('model_natpn_finetuned.pt', 'model_best_val_velocity.pt'):
            checkpoint = os.path.join(run_dir, filename)
            if not os.path.isfile(checkpoint):
                continue
            state = torch.load(checkpoint, map_location='cpu')['model_state_dict']
            checkpoint_features = state['base_model.input_proj.weight'].shape[1]
            checkpoint_uses_input_evidence = any(key.startswith('base_model.input_density.') for key in state)
            if checkpoint_features == num_features and checkpoint_uses_input_evidence == (evidence_source == 'input'):
                return checkpoint
    raise FileNotFoundError(
        f'No {evidence_source}-evidence checkpoint with {num_features} input features found in {logs_root}. '
        'Train that configuration first.'
    )


def latest_input_natpn_checkpoint(num_features):
    """Select the newest input-density artifact compatible with the current CSV features."""
    logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logsEncoder')
    for checkpoint_path in sorted(
        glob.glob(os.path.join(logs_root, '*', 'encoder_input_natpn.pt')),
        key=os.path.getmtime,
        reverse=True,
    ):
        if torch.load(checkpoint_path, map_location='cpu')['num_features'] == num_features:
            return checkpoint_path
    raise FileNotFoundError(
        f'No encoder_input_natpn.pt with {num_features} input features found in {logs_root}. '
        'Run trainEncoder.py first.'
    )


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
        getattr(
            posterior, 'epistemic_variance',
            posterior.beta / ((posterior.alpha - 1.0).clamp_min(1e-6) * posterior.lambd)
        ).item()
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


def format_vector(values):
    return '[' + ', '.join(f'{float(value):.4e}' for value in values) + ']'


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


def run_evaluation(args, context):
    config = context['config']
    ood_feature = context['ood_feature']
    environment_windows = context['environment_windows']
    low, high = context['cmd_vel_x_window']
    project_root = context['project_root']
    device = context['device']
    trajectory = context['trajectory']
    csv_path = context['csv_path']
    run_index = context['run_index']
    start_cmd_vel = context['start_cmd_vel']
    environment_id = context['environment_id']

    trajectory_features = make_features(trajectory)
    if config.get('natpn_evidence_source', 'task') == 'input' and not config.get('input_natpn_checkpoint'):
        config['input_natpn_checkpoint'] = latest_input_natpn_checkpoint(trajectory_features.shape[1])
        print(f"Input NatPN checkpoint: {config['input_natpn_checkpoint']}")
    model = make_model(config, trajectory_features.shape[1])
    checkpoint_path = latest_checkpoint(
        trajectory_features.shape[1], config.get('natpn_evidence_source', 'task')
    )
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
    uncertainty_final_timestep_gt_contact = bool(contact[random_position] == 1)
    random_aleatoric, random_epistemic = natpn_uncertainty_for_window(
        model, trajectory_features, random_position, config['window_size'], device
    )
    mae = np.abs(predicted[contact_mask] - ground_truth[contact_mask]).mean() if contact_mask.any() else float('nan')
    mean_variance = variance[contact_mask].mean(axis=0) if contact_mask.any() else np.full(3, np.nan)
    output_path = None
    if not args.skip_plots:
        output_path = os.path.join(os.path.dirname(checkpoint_path), f'trajectory_velocity_comparison_seed{RANDOM_SEED}.png')
        save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted, ground_truth, contact, output_path)

    if RUN_KNN_UMAP:
        cache_matches_checkpoint = False
        if args.knn_cache and os.path.isfile(args.knn_cache):
            with np.load(args.knn_cache, allow_pickle=False) as cache:
                cache_matches_checkpoint = (
                    cache['checkpoint_path'].item() == os.path.abspath(checkpoint_path)
                    and cache['checkpoint_mtime_ns'].item() == os.stat(checkpoint_path).st_mtime_ns
                    and cache['window_size'].item() == config['window_size']
                    and cache['num_features'].item() == trajectory_features.shape[1]
                )
                if cache_matches_checkpoint:
                    training_latents = cache['training_latents']
                    knn_k = int(cache['knn_k'].item())
                    ood_threshold = float(cache['ood_threshold'].item())
                    print(f'Reused kNN reference cache: {args.knn_cache}')
        if not cache_matches_checkpoint:
            data_folder = config['data_folder'] if os.path.isabs(config['data_folder']) else os.path.join(project_root, config['data_folder'])
            training_data = np.load(os.path.join(data_folder, 'all_data.npy'))
            training_starts = get_training_window_starts(data_folder, config['window_size'], config)
            training_latents = l2_normalize(collect_final_tcn_latents(
                model, training_data, training_starts, config['window_size'], config['batch_size'], device
            ))
            if len(training_latents) < 2:
                raise ValueError('Need at least two training windows for KNN OOD detection.')
            knn_k = min(KNN_K, len(training_latents) - 1)
            from sklearn.neighbors import NearestNeighbors
            train_distances = NearestNeighbors(n_neighbors=knn_k + 1).fit(training_latents).kneighbors(
                training_latents, return_distance=True
            )[0][:, 1:].mean(axis=1)
            ood_threshold = np.quantile(train_distances, OOD_ID_PERCENTILE)
            if args.knn_cache:
                os.makedirs(os.path.dirname(os.path.abspath(args.knn_cache)), exist_ok=True)
                np.savez(
                    args.knn_cache, training_latents=training_latents, knn_k=knn_k, ood_threshold=ood_threshold,
                    checkpoint_path=os.path.abspath(checkpoint_path), checkpoint_mtime_ns=os.stat(checkpoint_path).st_mtime_ns,
                    window_size=config['window_size'], num_features=trajectory_features.shape[1],
                )
                print(f'Built kNN reference cache: {args.knn_cache}')
        trajectory_starts = np.arange(len(trajectory) - config['window_size'] + 1)
        trajectory_latents = l2_normalize(collect_final_tcn_latents(
            model, trajectory_features, trajectory_starts, config['window_size'], config['batch_size'], device
        ))
        if len(trajectory_latents) != len(contact_mask):
            raise RuntimeError('Trajectory latent/contact-window alignment failed.')
        knn_query_mask = contact_mask if contact_mask.any() else np.ones_like(contact_mask, dtype=bool)
        if not contact_mask.any():
            print('No contact-final windows; using all evaluated windows for kNN.')
        trajectory_latents = trajectory_latents[knn_query_mask]
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=knn_k).fit(training_latents)
        knn_distances = neighbors.kneighbors(trajectory_latents, return_distance=True)[0].mean(axis=1)
        ood_mask = knn_distances > ood_threshold
        umap_output_path = None
        if args.save_umap and not args.skip_umap:
            umap_output_path = os.path.join(os.path.dirname(checkpoint_path), f'contact_trajectory_knn_umap_seed{RANDOM_SEED}.png')
            save_knn_umap(training_latents, trajectory_latents, knn_distances, ood_mask, knn_k, ood_threshold, umap_output_path)

    print(f'CSV: {csv_path}')
    selection = f'environment {environment_id}, windows: {environment_windows}' if ood_feature == 'environment' else f'cmd_vel_x range: [{low:.4e}, {high:.4e}]'
    print(f'Trajectory: {run_index}, start cmd_vel_x: {start_cmd_vel:.4e} m/s, selected by {selection}')
    print(f'Random seed: {RANDOM_SEED}, trajectory samples: {len(trajectory)}, evaluated windows: {len(indices)}')
    print(f'Contact-final windows: {contact_mask.sum()} / {len(contact)}')
    print(f'Contact-masked velocity MAE: {mae:.4e}')
    print(f'Mean predicted variance on contact [vx, vy, vz]: {format_vector(mean_variance)}')
    print(
        f'Random {"contact " if random_contact_window else ""}window: final sample {indices[random_position]} | '
        f'GT contact: {int(uncertainty_final_timestep_gt_contact)} | '
        f'aleatoric variance [vx, vy, vz]: {format_vector(random_aleatoric)} | '
        f'epistemic variance [vx, vy, vz]: {format_vector(random_epistemic)}'
    )
    if output_path:
        print(f'Whole-trajectory velocity plot: {output_path}')
    if RUN_KNN_UMAP:
        print(f'KNN reference windows: {len(training_latents)}, K: {knn_k}')
        print(f'Mean distance to {knn_k} nearest latent neighbors: mean {knn_distances.mean():.4e}, max {knn_distances.max():.4e}')
        print(f'ID threshold ({OOD_ID_PERCENTILE:.0%} training quantile): {ood_threshold:.4e}')
        print(f'OOD {"contact" if contact_mask.any() else "evaluated"} windows / all evaluated windows: {ood_mask.sum()} / {len(contact)} ({100.0 * ood_mask.sum() / len(contact):.4e}%)')

    if args.metrics_json:
        metrics = {
            'seed': RANDOM_SEED,
            'ood_feature': ood_feature,
            'environment': environment_id,
            'cmd_vel_x': float(start_cmd_vel),
            'velocity_mae': float(mae),
            'uncertainty_final_timestep_gt_contact': uncertainty_final_timestep_gt_contact,
            'aleatoric_variance': random_aleatoric.tolist(),
            'epistemic_variance': random_epistemic.tolist(),
            'knn_ood_windows': int(ood_mask.sum()) if RUN_KNN_UMAP else None,
            'knn_total_windows': len(contact) if RUN_KNN_UMAP else None,
            'knn_ood_percentage_total_windows': float(ood_mask.sum() / len(contact)) if RUN_KNN_UMAP else None,
        }
        with open(args.metrics_json, 'w') as metrics_file:
            json.dump(metrics, metrics_file)


class InputNatPNSingleTest(BaseSingleTest):
    default_seed = RANDOM_SEED
    default_cmd_vel_x_window = TEST_CMD_VEL_X_WINDOW
    model_config_name = 'NatPN_params.yaml'
    find_trajectory = staticmethod(find_random_trajectory)

    def add_model_arguments(self, parser):
        parser.add_argument('--knn-cache', help='Optional .npz cache for the invariant training-latent kNN reference.')
        parser.add_argument('--skip-plots', action='store_true', help='Do not save the per-trajectory velocity plot.')

    def configure(self, config):
        config['natpn_evidence_source'] = 'input'
        return config

    def set_runtime_values(self, args):
        global RANDOM_SEED, TEST_CMD_VEL_X_WINDOW
        RANDOM_SEED, TEST_CMD_VEL_X_WINDOW = args.seed, tuple(args.cmd_vel_x_window)

    def run(self, args, context):
        return run_evaluation(args, context)


def main():
    InputNatPNSingleTest().main()


if __name__ == '__main__':
    main()
