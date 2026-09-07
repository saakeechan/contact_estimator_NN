"""DER-specific model hooks for the shared probabilistic evaluator."""
import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

import numpy as np
import torch

from contact_cnn import ContactCNNWithNormalization, DERTCN
from tests.base_test import ProbabilisticVelocitySingleTest


class DERSingleTest(ProbabilisticVelocitySingleTest):
    default_seed = 203
    default_cmd_vel_x_window = (2.0, 3.0)
    model_config_name = 'DER_params.yaml'
    find_trajectory = staticmethod(ProbabilisticVelocitySingleTest.find_random_trajectory)

    def add_model_arguments(self, parser):
        parser.add_argument('--knn-cache', help='Optional .npz cache for the invariant training-latent kNN reference.')
        parser.add_argument('--skip-plots', action='store_true', help='Do not save the per-trajectory velocity plot.')

    def build_model(self, config, num_features):
        if config.get('model_architecture', 'tcn').lower() != 'tcn':
            raise ValueError("DER inference is implemented only for model_architecture: 'tcn'.")
        return ContactCNNWithNormalization(DERTCN(
            window_size=config['window_size'], num_features=num_features,
            tcn_num_channels=config.get('tcn_num_channels', 64),
            tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5), tcn_dropout=config.get('tcn_dropout', 0.2),
        ))

    def find_checkpoint(self, num_features, config):
        logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'logsDER')
        for run_dir in sorted(glob.glob(os.path.join(logs_root, '*')), key=os.path.getmtime, reverse=True):
            for filename in ('model_best_val_velocity.pt', 'model_final_epoch.pt'):
                checkpoint = os.path.join(run_dir, filename)
                if os.path.isfile(checkpoint):
                    state = torch.load(checkpoint, map_location='cpu')['model_state_dict']
                    if state['base_model.input_proj.weight'].shape[1] == num_features:
                        return checkpoint
        raise FileNotFoundError(f'No DER checkpoint with {num_features} input features found in {logs_root}.')

    @torch.no_grad()
    def uncertainty_for_window(self, model, features, window_start, window_size, device):
        window = torch.from_numpy(features[window_start:window_start + window_size]).float().unsqueeze(0).to(device)
        *_, (_, nu, alpha, beta) = model(window, return_sequence=False, return_posteriors=True)
        aleatoric = (beta / (alpha - 1.0).clamp_min(1e-6)).squeeze().cpu().numpy()
        return aleatoric, aleatoric / nu.squeeze().cpu().numpy()


def run_evaluation(args, context, model=None, checkpoint_path=None):
    return DERSingleTest().evaluate(args, context, model, checkpoint_path)


if __name__ == '__main__':
    DERSingleTest().main()
