"""Deep-ensemble architecture and checkpoint hooks for the shared evaluator."""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

from contact_cnn import ContactCNNWithNormalization, EnsembleTCN
from tests.base_test import ProbabilisticVelocitySingleTest


class EnsembleSingleTest(ProbabilisticVelocitySingleTest):
    default_seed = 203
    default_cmd_vel_x_window = (2.0, 3.0)
    model_config_name = 'Ensemble_params.yaml'
    find_trajectory = staticmethod(ProbabilisticVelocitySingleTest.find_random_trajectory)

    def add_model_arguments(self, parser):
        parser.add_argument('--skip-plots', action='store_true', help='Do not save the per-trajectory velocity plot.')

    @torch.no_grad()
    def run_ensemble_trajectory(self, models, trajectory, window_size, batch_size, device):
        features = self.make_features(trajectory)
        member_means, member_variances = [], []
        for first in range(0, len(features) - window_size + 1, batch_size):
            last = min(first + batch_size, len(features) - window_size + 1)
            windows = torch.from_numpy(np.stack([features[i:i + window_size] for i in range(first, last)])).float().to(device)
            outputs = [model(windows, return_sequence=False) for model in models]
            member_means.append(torch.stack([output[1][:, 0] for output in outputs]))
            member_variances.append(torch.stack([output[3][:, 0] for output in outputs]))
        means = torch.cat(member_means, dim=1)
        aleatoric = torch.cat(member_variances, dim=1).mean(dim=0)
        epistemic = means.var(dim=0, unbiased=False)
        final_indices = np.arange(window_size - 1, len(trajectory))
        return final_indices, means.mean(dim=0).cpu().numpy(), (aleatoric + epistemic).cpu().numpy(), aleatoric.cpu().numpy(), epistemic.cpu().numpy(), trajectory['lfoot-contact'].to_numpy()[final_indices], self.make_body_velocity(trajectory)[final_indices]

    def evaluate(self, args, context, models=None, checkpoint_paths=None):
        config, device, trajectory = context['config'], context['device'], context['trajectory']
        features = self.make_features(trajectory)
        if models is None:
            checkpoint_paths = self.find_member_checkpoints(config, features.shape[1])
            models = []
            for checkpoint_path in checkpoint_paths:
                model = self.build_member(config, features.shape[1]).to(device)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
                models.append(model.eval())
        indices, predicted, total, aleatoric, epistemic, contact, ground_truth = self.run_ensemble_trajectory(models, trajectory, config['window_size'], config.get('test_batch_size', config['batch_size']), device)
        contact_mask = contact == 1
        selected_mask = contact_mask if contact_mask.any() else np.ones(len(contact), dtype=bool)
        position = int(np.random.default_rng(args.seed).choice(np.flatnonzero(selected_mask)))
        if not args.skip_plots:
            output_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_paths[0])), f'ensemble_trajectory_seed{args.seed}.png')
            self.save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted, ground_truth, contact, output_path)
        mae = float(np.abs(predicted[contact_mask] - ground_truth[contact_mask]).mean()) if contact_mask.any() else float('nan')
        metrics = {
            'seed': args.seed, 'ood_feature': context['ood_feature'], 'environment': context['environment_id'],
            'cmd_vel_x': float(context['start_cmd_vel']), 'velocity_mae': mae,
            'uncertainty_final_timestep_gt_contact': bool(contact[position] == 1),
            'num_ensemble_members': len(models), 'aleatoric_variance': aleatoric[position].tolist(),
            'epistemic_variance': epistemic[position].tolist(), 'total_variance': total[position].tolist(),
            'mean_epistemic_variance_on_contact': epistemic[selected_mask].mean(axis=0).tolist(),
            'knn_ood_windows': None, 'knn_total_windows': None, 'knn_ood_percentage_total_windows': None,
        }
        return metrics, models, checkpoint_paths

    def build_member(self, config, num_features):
        if config.get('model_architecture', 'tcn').lower() != 'tcn':
            raise ValueError("Deep-ensemble inference is implemented only for model_architecture: 'tcn'.")
        return ContactCNNWithNormalization(EnsembleTCN(
            window_size=config['window_size'], num_features=num_features,
            tcn_num_channels=config.get('tcn_num_channels', 64),
            tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5), tcn_dropout=config.get('tcn_dropout', 0.2),
        ))

    def find_member_checkpoints(self, config, num_features):
        expected_members = int(config['num_ensemble_members'])
        logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'logsEnsemble')
        for run_dir in sorted(glob.glob(os.path.join(logs_root, 'run_*')), key=os.path.getmtime, reverse=True):
            checkpoints = sorted(glob.glob(os.path.join(run_dir, 'member_*', 'model_best_val_velocity.pt')))
            if len(checkpoints) == expected_members:
                state = torch.load(checkpoints[0], map_location='cpu')['model_state_dict']
                if state['base_model.input_proj.weight'].shape[1] == num_features:
                    return checkpoints
        raise FileNotFoundError(f'No complete {expected_members}-member ensemble with {num_features} input features found in {logs_root}.')


def run_evaluation(args, context, models=None, checkpoint_paths=None):
    return EnsembleSingleTest().evaluate(args, context, models, checkpoint_paths)


if __name__ == '__main__':
    EnsembleSingleTest().main()
