"""MC-dropout-specific hooks for the shared probabilistic evaluator."""
import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

import numpy as np
import torch
import torch.nn as nn

from contact_cnn import ContactCNNWithNormalization, MCDropoutTCN
from tests.base_test import ProbabilisticVelocitySingleTest


class MCDropoutSingleTest(ProbabilisticVelocitySingleTest):
    default_seed = 203
    default_cmd_vel_x_window = (2.0, 3.0)
    model_config_name = 'MCdropout_params.yaml'
    find_trajectory = staticmethod(ProbabilisticVelocitySingleTest.find_random_trajectory)

    def add_model_arguments(self, parser):
        parser.add_argument('--knn-cache', help='Optional .npz cache for the invariant training-latent kNN reference.')
        parser.add_argument('--skip-plots', action='store_true', help='Do not save the per-trajectory velocity plot.')
        parser.add_argument('--mc-samples', type=int, help='Override mc_dropout_samples from the config.')

    def set_runtime_values(self, args):
        self.mc_samples_override = args.mc_samples

    def configure(self, config):
        samples = int(getattr(self, 'mc_samples_override', None) or config['mc_dropout_samples'])
        if samples < 2:
            raise ValueError('mc_dropout_samples must be at least 2.')
        config['mc_dropout_samples'] = samples
        return config

    def build_model(self, config, num_features):
        if config.get('model_architecture', 'tcn').lower() != 'tcn':
            raise ValueError("MC-dropout inference is implemented only for model_architecture: 'tcn'.")
        return ContactCNNWithNormalization(MCDropoutTCN(
            config['window_size'], num_features, config.get('tcn_num_channels', 64),
            config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), float(config['mc_dropout_rate']),
        ))

    def find_checkpoint(self, num_features, config):
        logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logsMCDropout')
        for run_dir in sorted(glob.glob(os.path.join(logs_root, 'run_*')), key=os.path.getmtime, reverse=True):
            for filename in ('model_best_val_velocity.pt', 'model_final_epoch.pt'):
                checkpoint = os.path.join(run_dir, filename)
                if os.path.isfile(checkpoint):
                    state = torch.load(checkpoint, map_location='cpu')['model_state_dict']
                    if state['base_model.input_proj.weight'].shape[1] == num_features:
                        return checkpoint
        raise FileNotFoundError(f'No MC-dropout checkpoint with {num_features} input features found in {logs_root}.')

    @staticmethod
    @torch.no_grad()
    def stochastic_outputs(model, windows, samples):
        model.eval()
        dropout_modules = [module for module in model.modules() if isinstance(module, nn.Dropout)]
        for module in dropout_modules:
            module.train()
        try:
            outputs = [model(windows, return_sequence=False) for _ in range(samples)]
        finally:
            for module in dropout_modules:
                module.eval()
        means = torch.stack([output[1][:, 0] for output in outputs])
        aleatoric = torch.stack([output[3][:, 0] for output in outputs]).mean(dim=0)
        epistemic = means.var(dim=0, unbiased=False)
        return means.mean(dim=0), aleatoric, epistemic

    def run_trajectory(self, model, trajectory, window_size, batch_size, device):
        features = self.make_features(trajectory)
        predicted, aleatoric, epistemic = [], [], []
        for first in range(0, len(features) - window_size + 1, batch_size):
            last = min(first + batch_size, len(features) - window_size + 1)
            windows = torch.from_numpy(np.stack([features[i:i + window_size] for i in range(first, last)])).float().to(device)
            mean, noise, disagreement = self.stochastic_outputs(model, windows, self.mc_samples)
            predicted.append(mean.cpu().numpy())
            aleatoric.append(noise.cpu().numpy())
            epistemic.append(disagreement.cpu().numpy())
        final_indices = np.arange(window_size - 1, len(trajectory))
        contact = trajectory['lfoot-contact'].to_numpy()[final_indices]
        return final_indices, np.concatenate(predicted), np.concatenate(aleatoric), np.concatenate(epistemic), contact, self.make_body_velocity(trajectory)[final_indices]

    def evaluate(self, args, context, model=None, checkpoint_path=None):
        self.mc_samples = int(context['config']['mc_dropout_samples'])
        config, device, trajectory = context['config'], context['device'], context['trajectory']
        features = self.make_features(trajectory)
        if model is None:
            model = self.build_model(config, features.shape[1]).to(device)
            checkpoint_path = self.find_checkpoint(features.shape[1], config)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
        model.eval()
        indices, predicted, aleatoric, epistemic, contact, ground_truth = self.run_trajectory(
            model, trajectory, config['window_size'], config.get('test_batch_size', config['batch_size']), device
        )
        total = aleatoric + epistemic
        contact_mask = contact == 1
        candidates = np.flatnonzero(contact_mask) if contact_mask.any() else np.arange(len(contact))
        position = int(np.random.default_rng(args.seed).choice(candidates))
        if not args.skip_plots:
            output_path = os.path.join(os.path.dirname(checkpoint_path), f'mc_dropout_trajectory_seed{args.seed}.png')
            self.save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted, ground_truth, contact, output_path)

        def save_plot(*knn_args):
            if args.save_umap and not args.skip_umap:
                self.save_knn_umap(*knn_args, os.path.join(os.path.dirname(checkpoint_path), f'mc_dropout_knn_umap_seed{args.seed}.png'))

        knn = self.run_knn_ood(args, context, model, checkpoint_path, features, contact_mask,
                               self.get_training_window_starts, self.collect_final_tcn_latents, save_plot,
                               self.knn_k, self.ood_id_percentile)
        selected_mask = contact_mask if contact_mask.any() else np.ones(len(contact), dtype=bool)
        mae = float(np.abs(predicted[contact_mask] - ground_truth[contact_mask]).mean()) if contact_mask.any() else float('nan')
        return {
            'seed': args.seed, 'ood_feature': context['ood_feature'], 'environment': context['environment_id'],
            'cmd_vel_x': float(context['start_cmd_vel']), 'velocity_mae': mae,
            'uncertainty_final_timestep_gt_contact': bool(contact[position] == 1),
            'mc_dropout_samples': self.mc_samples,
            'aleatoric_variance': aleatoric[position].tolist(),
            'epistemic_variance': epistemic[position].tolist(),
            'total_variance': total[position].tolist(),
            'mean_epistemic_variance_on_contact': epistemic[selected_mask].mean(axis=0).tolist(),
            'knn_ood_windows': int(knn['ood_mask'].sum()) if knn else None,
            'knn_total_windows': len(contact) if knn else None,
            'knn_ood_percentage_total_windows': float(knn['ood_mask'].sum() / len(contact)) if knn else None,
        }, model, checkpoint_path

    def uncertainty_for_window(self, model, features, window_start, window_size, device):
        window = torch.from_numpy(features[window_start:window_start + window_size]).float().unsqueeze(0).to(device)
        _, aleatoric, epistemic = self.stochastic_outputs(model, window, self.mc_samples)
        return aleatoric.squeeze(0).cpu().numpy(), epistemic.squeeze(0).cpu().numpy()


def run_evaluation(args, context, model=None, checkpoint_path=None):
    return MCDropoutSingleTest().evaluate(args, context, model, checkpoint_path)


if __name__ == '__main__':
    MCDropoutSingleTest().main()
