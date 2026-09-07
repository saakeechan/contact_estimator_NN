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

from mc_dropout import MCDropoutTCN
from normalization import ContactCNNWithNormalization
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
            config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), float(config['mc_dropout_rate']), legs=config['legs'],
        ))

    def find_checkpoint(self, num_features, config):
        logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'logsMCDropout')
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
        means = torch.stack([output[1] for output in outputs])
        aleatoric = torch.stack([output[3] for output in outputs]).mean(dim=0)
        epistemic = means.var(dim=0, unbiased=False)
        return means.mean(dim=0), aleatoric, epistemic

    def run_trajectory(self, model, trajectory, window_size, batch_size, device):
        features = self.make_features(trajectory, model.base_model.legs)
        predicted, aleatoric, epistemic = [], [], []
        for first in range(0, len(features) - window_size + 1, batch_size):
            last = min(first + batch_size, len(features) - window_size + 1)
            windows = torch.from_numpy(np.stack([features[i:i + window_size] for i in range(first, last)])).float().to(device)
            mean, noise, disagreement = self.stochastic_outputs(model, windows, self.mc_samples)
            predicted.append(mean.cpu().numpy())
            aleatoric.append(noise.cpu().numpy())
            epistemic.append(disagreement.cpu().numpy())
        final_indices = np.arange(window_size - 1, len(trajectory))
        contacts = np.stack([trajectory[f'{leg[0]}foot-contact'].to_numpy()[final_indices] for leg in model.base_model.legs], axis=1)
        body_velocity = self.make_body_velocity(trajectory)[final_indices]
        return final_indices, np.concatenate(predicted), np.concatenate(aleatoric), np.concatenate(epistemic), contacts, np.repeat(body_velocity[:, None, :], len(model.base_model.legs), axis=1)

    def evaluate(self, args, context, model=None, checkpoint_path=None):
        self.mc_samples = int(context['config']['mc_dropout_samples'])
        config, device, trajectory = context['config'], context['device'], context['trajectory']
        features = self.make_features(trajectory, config['legs'])
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
        union_contact_mask = contact_mask.any(axis=1)
        candidates = np.flatnonzero(union_contact_mask) if union_contact_mask.any() else np.arange(len(contact))
        position = int(np.random.default_rng(args.seed).choice(candidates))
        if not args.skip_plots:
            output_path = os.path.join(os.path.dirname(checkpoint_path), f'mc_dropout_trajectory_seed{args.seed}.png')
            for leg_index, leg in enumerate(config['legs']):
                self.save_velocity_plot(trajectory['timestamp'].to_numpy()[indices], predicted[:, leg_index], ground_truth[:, leg_index], contact[:, leg_index], output_path.replace('.png', f'_{leg}.png'))

        def save_plot(*knn_args):
            if args.save_umap and not args.skip_umap:
                self.save_knn_umap(*knn_args, os.path.join(os.path.dirname(checkpoint_path), f'mc_dropout_knn_umap_seed{args.seed}.png'))

        knn = self.run_knn_ood(args, context, model, checkpoint_path, features, union_contact_mask,
                               self.get_training_window_starts, self.collect_final_tcn_latents, save_plot,
                               self.knn_k, self.ood_id_percentile)
        selected_mask = union_contact_mask if union_contact_mask.any() else np.ones(len(contact), dtype=bool)
        per_leg_mae = {leg: float(np.abs(predicted[:, index][contact_mask[:, index]] - ground_truth[:, index][contact_mask[:, index]]).mean()) if contact_mask[:, index].any() else float('nan') for index, leg in enumerate(config['legs'])}
        mae = float(np.nanmean(list(per_leg_mae.values())))
        return {
            'seed': args.seed, 'ood_feature': context['ood_feature'], 'environment': context['environment_id'],
            'cmd_vel_x': float(context['start_cmd_vel']), 'velocity_mae': mae,
            'uncertainty_final_timestep_gt_contact': {leg: bool(contact[position, index] == 1) for index, leg in enumerate(config['legs'])}, 'per_leg_velocity_mae': per_leg_mae,
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
