import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
import torch
import yaml

from contact_cnn import ContactCNNWithNormalization, DERTCN
from train.base_train import BaseTrainer, load_training_data


def nig_loss(parameters, target, regularizer_weight):
    gamma, nu, alpha, beta = parameters
    two_beta_lambda = 2.0 * beta * (1.0 + nu)
    nll = (.5 * torch.log(torch.pi / nu) - alpha * torch.log(two_beta_lambda)
           + (alpha + .5) * torch.log(nu * (target - gamma).square() + two_beta_lambda)
           + torch.lgamma(alpha) - torch.lgamma(alpha + .5))
    return nll + regularizer_weight * (target - gamma).abs() * (2.0 * nu + alpha)


class DERTrainer(BaseTrainer):
    logs_dir = 'logsDER'

    def __init__(self, model, config):
        super().__init__(model, config)
        self.regularizer_weight = float(config.get('der_regularizer_weight', 1e-2))
        self.loss_description = f'NIG negative log-likelihood + {self.regularizer_weight:g} evidence regularizer'

    def forward(self, inputs, dense):
        return self.model(inputs, return_sequence=dense, return_posteriors=True)

    def velocity_loss(self, outputs, velocity, dense):
        parameters = outputs[5]
        if dense:
            parameters = tuple(value.permute(0, 3, 1, 2) for value in parameters)
        return nig_loss(parameters, velocity, self.regularizer_weight)


def main():
    parser = argparse.ArgumentParser(description='Train DER model')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(__file__), '../config/DER_params.yaml')) as file:
        config = yaml.safe_load(file) or {}
    with open(args.config_name) as file:
        config.update(yaml.safe_load(file) or {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features, mean, std, train_loader, val_loader = load_training_data(config, device)
    if config.get('model_architecture', 'tcn').lower() != 'tcn':
        raise ValueError("DER training requires model_architecture: 'tcn'.")
    model = ContactCNNWithNormalization(DERTCN(
        config['window_size'], num_features, config.get('tcn_num_channels', 64),
        config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), config.get('tcn_dropout', .2),
    ), global_mean=mean, global_std=std).to(device)
    DERTrainer(model, config).train(train_loader, val_loader)


if __name__ == '__main__':
    main()
