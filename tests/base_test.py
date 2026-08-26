"""Shared command-line setup for model-specific single-trajectory tests."""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from utils.ood_selection import csv_matches_environment_windows, validate_ood_selection


class BaseSingleTest:
    """Select one evaluation trajectory, then delegate inference to a model subclass."""

    description = 'Run one selected CSV trajectory through the contact network.'
    default_cmd_vel_x_window = (2.0, 3.0)
    model_config_name = None

    def build_parser(self):
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
        parser.add_argument('--seed', type=int, default=self.default_seed)
        parser.add_argument('--cmd-vel-x-window', type=float, nargs=2, metavar=('MIN', 'MAX'),
                            default=self.default_cmd_vel_x_window)
        parser.add_argument('--ood-feature', choices=('cmd_vel', 'environment'), help='Override ood_feature from the config.')
        parser.add_argument('--environment-window', type=int, nargs=2, action='append', metavar=('MIN', 'MAX'),
                            help='Inclusive environment window; repeat this option for disjoint windows.')
        parser.add_argument('--metrics-json', help='Optional path for machine-readable evaluation metrics.')
        parser.add_argument('--skip-umap', action='store_true', help='Compute kNN/OOD metrics without saving a UMAP figure.')
        parser.add_argument('--save-umap', action='store_true', help='Save the kNN UMAP figure (disabled by default).')
        parser.add_argument('--skip-knn', action='store_true', help='Do not run kNN/OOD evaluation.')
        self.add_model_arguments(parser)
        return parser

    def add_model_arguments(self, parser):
        """Add arguments used only by a concrete model evaluator."""

    def load_config(self, config_name):
        config = {}
        if self.model_config_name:
            model_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', self.model_config_name)
            with open(model_config_path) as config_file:
                config.update(yaml.safe_load(config_file) or {})
        with open(config_name) as config_file:
            config.update(yaml.safe_load(config_file) or {})
        return self.configure(config)

    def configure(self, config):
        return config

    def prepare(self, args):
        self.set_runtime_values(args)
        config = self.load_config(args.config_name)
        ood_feature = args.ood_feature or config.get('ood_feature', 'cmd_vel')
        environment_windows = args.environment_window or config.get('environment_windows', [])
        validate_ood_selection(ood_feature, environment_windows)
        cmd_vel_x_window = tuple(args.cmd_vel_x_window)
        low, high = cmd_vel_x_window
        if low > high:
            raise ValueError('TEST_CMD_VEL_X_WINDOW must be (min, max) with min <= max')

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_folder = config['csv_folder'] if os.path.isabs(config['csv_folder']) else os.path.join(project_root, config['csv_folder'])
        trajectory, csv_path, run_index, start_cmd_vel = self.find_trajectory(
            csv_folder, config['window_size'], cmd_vel_x_window, np.random.default_rng(args.seed),
            ood_feature, environment_windows,
        )
        environment_id = csv_matches_environment_windows(csv_path, environment_windows)[1] if ood_feature == 'environment' else None
        return {
            'config': config, 'ood_feature': ood_feature, 'environment_windows': environment_windows,
            'cmd_vel_x_window': cmd_vel_x_window, 'project_root': project_root,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'trajectory': trajectory, 'csv_path': csv_path, 'run_index': run_index,
            'start_cmd_vel': start_cmd_vel, 'environment_id': environment_id,
        }

    def main(self):
        args = self.build_parser().parse_args()
        return self.run(args, self.prepare(args))

    def set_runtime_values(self, args):
        """Update legacy module-level defaults used by plotting helpers."""

    def find_trajectory(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, args, context):
        result = self.evaluate(args, context)
        metrics = result[0] if isinstance(result, tuple) else result
        if args.metrics_json:
            with open(args.metrics_json, 'w') as metrics_file:
                json.dump(metrics, metrics_file)
        return result

    @staticmethod
    def l2_normalize(features):
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    def run_knn_ood(self, args, context, model, checkpoint_path, trajectory_features,
                    contact_mask, get_training_starts, extract_latents, save_plot,
                    knn_k=20, id_percentile=0.95):
        """Run cacheable kNN/OOD evaluation using model-specific latent extraction."""
        if args.skip_knn:
            return None

        config, project_root, device = context['config'], context['project_root'], context['device']
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
                    active_knn_k = int(cache['knn_k'].item())
                    threshold = float(cache['ood_threshold'].item())
                    print(f'Reused kNN reference cache: {args.knn_cache}')

        if not cache_matches_checkpoint:
            data_folder = config['data_folder'] if os.path.isabs(config['data_folder']) else os.path.join(project_root, config['data_folder'])
            training_data = np.load(os.path.join(data_folder, 'all_data.npy'))
            training_starts = get_training_starts(data_folder, config['window_size'], config)
            training_latents = self.l2_normalize(extract_latents(
                model, training_data, training_starts, config['window_size'], config['batch_size'], device
            ))
            if len(training_latents) < 2:
                raise ValueError('Need at least two training windows for kNN OOD detection.')
            active_knn_k = min(knn_k, len(training_latents) - 1)
            from sklearn.neighbors import NearestNeighbors
            train_distances = NearestNeighbors(n_neighbors=active_knn_k + 1).fit(training_latents).kneighbors(
                training_latents, return_distance=True
            )[0][:, 1:].mean(axis=1)
            threshold = np.quantile(train_distances, id_percentile)
            if args.knn_cache:
                os.makedirs(os.path.dirname(os.path.abspath(args.knn_cache)), exist_ok=True)
                np.savez(
                    args.knn_cache, training_latents=training_latents, knn_k=active_knn_k, ood_threshold=threshold,
                    checkpoint_path=os.path.abspath(checkpoint_path), checkpoint_mtime_ns=os.stat(checkpoint_path).st_mtime_ns,
                    window_size=config['window_size'], num_features=trajectory_features.shape[1],
                )
                print(f'Built kNN reference cache: {args.knn_cache}')

        trajectory_starts = np.arange(len(trajectory_features) - config['window_size'] + 1)
        trajectory_latents = self.l2_normalize(extract_latents(
            model, trajectory_features, trajectory_starts, config['window_size'], config['batch_size'], device
        ))
        if len(trajectory_latents) != len(contact_mask):
            raise RuntimeError('Trajectory latent/contact-window alignment failed.')
        query_mask = contact_mask if contact_mask.any() else np.ones_like(contact_mask, dtype=bool)
        if not contact_mask.any():
            print('No contact-final windows; using all evaluated windows for kNN.')
        from sklearn.neighbors import NearestNeighbors
        distances = NearestNeighbors(n_neighbors=active_knn_k).fit(training_latents).kneighbors(
            trajectory_latents[query_mask], return_distance=True
        )[0].mean(axis=1)
        ood_mask = distances > threshold
        if args.save_umap and not args.skip_umap:
            save_plot(training_latents, trajectory_latents[query_mask], distances, ood_mask, active_knn_k, threshold)
        return {'training_latents': training_latents, 'knn_k': active_knn_k, 'threshold': threshold,
                'distances': distances, 'ood_mask': ood_mask, 'query_mask': query_mask}

    def evaluate(self, args, context):
        raise NotImplementedError


class ProbabilisticVelocitySingleTest(BaseSingleTest):
    """Shared evaluator for one probabilistic velocity model and one trajectory."""

    knn_k = 20
    ood_id_percentile = 0.95

    @staticmethod
    def find_random_trajectory(csv_folder, window_size, cmd_vel_x_window, rng,
                               ood_feature='cmd_vel', environment_windows=()):
        import glob
        import pandas as pd

        low, high = cmd_vel_x_window
        csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
        if not csv_files:
            raise FileNotFoundError(f'No CSV files found in {csv_folder}')
        if ood_feature == 'environment':
            csv_files = [path for path in csv_files if csv_matches_environment_windows(path, environment_windows)[0]]
        for csv_path in rng.permutation(csv_files):
            dataframe = pd.read_csv(csv_path)
            if 'cmd_vel_x' not in dataframe:
                continue
            timestamps = dataframe['timestamp'].to_numpy()
            boundaries = [0] + (np.flatnonzero(np.abs(np.diff(timestamps)) > 0.025) + 1).tolist() + [len(dataframe)]
            for run_index in rng.permutation(len(boundaries) - 1):
                trajectory = dataframe.iloc[boundaries[run_index]:boundaries[run_index + 1]].reset_index(drop=True)
                if len(trajectory) >= window_size and (ood_feature == 'environment' or low <= trajectory['cmd_vel_x'].iloc[0] <= high):
                    return trajectory, csv_path, run_index, trajectory['cmd_vel_x'].iloc[0]
        raise ValueError('No trajectory matches the requested evaluation selection.')

    @staticmethod
    def make_features(trajectory):
        joint_names = (
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        )
        q = trajectory[['joint_pos_' + name for name in joint_names]].to_numpy()
        qd = trajectory[['joint_vel_' + name for name in joint_names]].to_numpy()
        foot_position = trajectory[['fk_left_foot_pos_x', 'fk_left_foot_pos_y', 'fk_left_foot_pos_z']].to_numpy()
        foot_velocity = trajectory[['fk_left_foot_vel_x', 'fk_left_foot_vel_y', 'fk_left_foot_vel_z']].to_numpy()
        torque = trajectory[['joint_torque_' + name for name in joint_names]].to_numpy()
        torque_mse = np.sum(torque ** 2, axis=1, keepdims=True)
        return np.concatenate((q, qd, foot_position, foot_velocity, torque, torque_mse, trajectory[['cmd_vel_x']].to_numpy()), axis=1)

    @staticmethod
    def make_body_velocity(trajectory):
        from utils.csv2numpyV1 import quaternion_to_rotation_matrix

        velocity_world = trajectory[['vel_x', 'vel_y', 'vel_z']].to_numpy()
        quaternion = trajectory[['quat_w', 'quat_i', 'quat_j', 'quat_k']].to_numpy()
        return np.einsum('nij,nj->ni', quaternion_to_rotation_matrix(quaternion).transpose(0, 2, 1), velocity_world)

    def build_model(self, config, num_features):
        raise NotImplementedError

    def find_checkpoint(self, num_features, config):
        raise NotImplementedError

    def uncertainty_for_window(self, model, features, window_start, window_size, device):
        raise NotImplementedError

    @staticmethod
    def get_training_window_starts(data_folder, window_size, config):
        boundaries = np.load(os.path.join(data_folder, 'all_data_boundaries.npy'))
        run_starts = np.concatenate(([0], boundaries[:-1]))
        valid_run_ids = np.flatnonzero(boundaries - run_starts >= window_size)
        if len(valid_run_ids) < 3:
            raise ValueError(f'Need at least 3 runs with {window_size} samples for the training split.')
        run_ids = valid_run_ids.copy()
        if config.get('shuffle', True):
            np.random.RandomState(config.get('random_seed', 42)).shuffle(run_ids)
        train_count = max(1, int(config.get('train_ratio', 0.7) * len(run_ids)))
        return np.concatenate([np.arange(run_starts[run_id], boundaries[run_id] - window_size + 1) for run_id in run_ids[:train_count]])

    @staticmethod
    def collect_final_tcn_latents(model, raw_data, window_starts, window_size, batch_size, device):
        latents = []
        hook = model.base_model.tcn_backbone.register_forward_hook(lambda _, __, output: latents.append(output[:, :, -1].detach().cpu()))
        offsets = np.arange(window_size)
        try:
            with torch.no_grad():
                for first in range(0, len(window_starts), batch_size):
                    starts = window_starts[first:first + batch_size]
                    model(torch.from_numpy(raw_data[starts[:, None] + offsets]).float().to(device), return_sequence=False)
        finally:
            hook.remove()
        return torch.cat(latents).numpy()

    def run_trajectory(self, model, trajectory, window_size, batch_size, device):
        features = self.make_features(trajectory)
        predictions, variances, contact_probabilities = [], [], []
        with torch.no_grad():
            for first in range(0, len(features) - window_size + 1, batch_size):
                last = min(first + batch_size, len(features) - window_size + 1)
                windows = torch.from_numpy(np.stack([features[i:i + window_size] for i in range(first, last)])).float().to(device)
                _, velocity, _, variance, contact_logit = model(windows, return_sequence=False)
                predictions.append(velocity[:, 0].cpu().numpy())
                variances.append(variance[:, 0].cpu().numpy())
                contact_probabilities.append(torch.sigmoid(contact_logit[:, 0]).cpu().numpy())
        final_indices = np.arange(window_size - 1, len(trajectory))
        return final_indices, np.concatenate(predictions), np.concatenate(variances), np.concatenate(contact_probabilities), trajectory['lfoot-contact'].to_numpy()[final_indices], self.make_body_velocity(trajectory)[final_indices]

    @staticmethod
    def save_velocity_plot(time, predicted, ground_truth, contact, output_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

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
        axes[-1].set_xlabel('Trajectory time (s)')
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(figure)

    @staticmethod
    def save_knn_umap(training_latents, trajectory_latents, distances, ood_mask, knn_k, threshold, output_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from umap import UMAP

        sampled = np.random.default_rng(203).choice(len(training_latents), size=min(5000, len(training_latents)), replace=False)
        reducer = UMAP(n_components=2, random_state=203)
        training_embedding = reducer.fit_transform(training_latents[sampled])
        trajectory_embedding = reducer.transform(trajectory_latents)
        figure, axis = plt.subplots(figsize=(10, 8))
        axis.scatter(training_embedding[:, 0], training_embedding[:, 1], s=4, color='lightgray', alpha=0.45)
        points = axis.scatter(trajectory_embedding[:, 0], trajectory_embedding[:, 1], c=distances, cmap='viridis', s=18)
        axis.scatter(trajectory_embedding[ood_mask, 0], trajectory_embedding[ood_mask, 1], s=44, facecolors='none', edgecolors='red')
        figure.colorbar(points, ax=axis, label=f'Mean distance to {knn_k} nearest latent neighbors')
        axis.set(title=f'Training-window UMAP (OOD threshold: {threshold:.4f})', xlabel='UMAP 1', ylabel='UMAP 2')
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(figure)

    def evaluate(self, args, context, model=None, checkpoint_path=None):
        config, device, trajectory = context['config'], context['device'], context['trajectory']
        features = self.make_features(trajectory)
        if model is None:
            model = self.build_model(config, features.shape[1]).to(device)
            checkpoint_path = self.find_checkpoint(features.shape[1], config)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
            model.eval()
        indices, predicted, variance, _, contact, ground_truth = self.run_trajectory(model, trajectory, config['window_size'], config.get('test_batch_size', config['batch_size']), device)
        contact_mask = contact == 1
        candidates = np.flatnonzero(contact_mask) if contact_mask.any() else np.arange(len(contact))
        position = int(np.random.default_rng(args.seed).choice(candidates))
        aleatoric, epistemic, *extra = self.uncertainty_for_window(model, features, position, config['window_size'], device)
        output_path = None
        if not args.skip_plots:
            output_path = os.path.join(os.path.dirname(checkpoint_path), f'trajectory_velocity_comparison_seed{args.seed}.png')
            self.save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted, ground_truth, contact, output_path)
        def save_plot(*knn_args):
            if args.save_umap and not args.skip_umap:
                self.save_knn_umap(*knn_args, os.path.join(os.path.dirname(checkpoint_path), f'contact_trajectory_knn_umap_seed{args.seed}.png'))
        knn = self.run_knn_ood(args, context, model, checkpoint_path, features, contact_mask,
                               self.get_training_window_starts, self.collect_final_tcn_latents, save_plot,
                               self.knn_k, self.ood_id_percentile)
        mae = float(np.abs(predicted[contact_mask] - ground_truth[contact_mask]).mean()) if contact_mask.any() else float('nan')
        print(f"CSV: {context['csv_path']}")
        print(f"Random seed: {args.seed}, trajectory samples: {len(trajectory)}, evaluated windows: {len(indices)}")
        print(f'Contact-final windows: {contact_mask.sum()} / {len(contact)}')
        print(f'Contact-masked velocity MAE: {mae:.4e}')
        print(f'Aleatoric variance [vx, vy, vz]: {aleatoric.tolist()}')
        print(f'Epistemic variance [vx, vy, vz]: {epistemic.tolist()}')
        if knn:
            print(f"KNN reference windows: {len(knn['training_latents'])}, K: {knn['knn_k']}, OOD windows: {knn['ood_mask'].sum()}")
        metrics = {
            'seed': args.seed, 'ood_feature': context['ood_feature'], 'environment': context['environment_id'],
            'cmd_vel_x': float(context['start_cmd_vel']),
            'velocity_mae': mae,
            'uncertainty_final_timestep_gt_contact': bool(contact[position] == 1),
            'aleatoric_variance': aleatoric.tolist(), 'epistemic_variance': epistemic.tolist(),
            'knn_ood_windows': int(knn['ood_mask'].sum()) if knn else None,
            'knn_total_windows': len(contact) if knn else None,
            'knn_ood_percentage_total_windows': float(knn['ood_mask'].sum() / len(contact)) if knn else None,
        }
        if extra:
            metrics['negative_log_density'] = float(extra[0])
        return metrics, model, checkpoint_path
