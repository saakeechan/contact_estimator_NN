"""Train a TCN for Monte-Carlo-dropout uncertainty estimation."""
import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

import torch
import torch.nn.functional as F
import yaml

from contact_cnn import ContactCNNWithNormalization, MCDropoutTCN
from train.base_train import BaseTrainer, load_training_data


class MCDropoutTrainer(BaseTrainer):
    logs_dir = 'logsMCDropout'
    loss_description = 'Gaussian negative log-likelihood (dropout enabled during training)'

    def velocity_loss(self, outputs, velocity, dense):
        velocity_prediction, covariance = outputs[0], outputs[2]
        if dense:
            velocity_prediction = velocity_prediction.permute(0, 3, 1, 2)
            covariance = covariance.permute(0, 3, 1, 2)
        else:
            velocity_prediction, covariance = outputs[1], outputs[3]
        return F.gaussian_nll_loss(velocity_prediction, velocity, covariance, reduction='none')


def main():
    parser = argparse.ArgumentParser(description='Train an MC-dropout TCN')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
    args = parser.parse_args()
    with open(args.config_name) as file:
        config = yaml.safe_load(file) or {}
    with open(os.path.join(os.path.dirname(__file__), '../config/MCdropout_params.yaml')) as file:
        config.update(yaml.safe_load(file) or {})
    if config.get('model_architecture', 'tcn').lower() != 'tcn':
        raise ValueError("MC-dropout training requires model_architecture: 'tcn'.")
    dropout_rate = float(config['mc_dropout_rate'])
    if not 0.0 < dropout_rate < 1.0:
        raise ValueError('mc_dropout_rate must be in (0, 1).')
    if int(config['mc_dropout_samples']) < 2:
        raise ValueError('mc_dropout_samples must be at least 2.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features, mean, std, train_loader, val_loader = load_training_data(config, device)
    model = ContactCNNWithNormalization(MCDropoutTCN(
        config['window_size'], num_features, config.get('tcn_num_channels', 64),
        config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), dropout_rate,
    ), global_mean=mean, global_std=std).to(device)
    MCDropoutTrainer(model, config).train(train_loader, val_loader)


if __name__ == '__main__':
    main()
