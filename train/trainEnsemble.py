import argparse
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from contact_cnn import ContactCNNWithNormalization, EnsembleTCN
from train.base_train import BaseTrainer, load_training_data


class EnsembleTrainer(BaseTrainer):
    loss_description = 'Gaussian negative log-likelihood'

    def velocity_loss(self, outputs, velocity, dense):
        velocity_prediction, covariance = outputs[0], outputs[2]
        if dense:
            velocity_prediction, covariance = velocity_prediction.permute(0, 3, 1, 2), covariance.permute(0, 3, 1, 2)
        else:
            velocity_prediction, covariance = outputs[1], outputs[3]
        return F.gaussian_nll_loss(velocity_prediction, velocity, covariance, reduction='none')


def main():
    parser = argparse.ArgumentParser(description='Train a deep ensemble')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
    args = parser.parse_args()
    with open(args.config_name) as file:
        config = yaml.safe_load(file) or {}
    with open(os.path.join(os.path.dirname(__file__), '../config/Ensemble_params.yaml')) as file:
        config.update(yaml.safe_load(file) or {})
    count = int(config.get('num_ensemble_members', 5))
    if count < 2:
        raise ValueError('num_ensemble_members must be at least 2.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.get('model_architecture', 'tcn').lower() != 'tcn':
        raise ValueError("Ensemble training requires model_architecture: 'tcn'.")
    root_dir = os.path.join('logsEnsemble', f"run_{__import__('datetime').datetime.now():%Y-%m-%d_%H-%M-%S}")
    num_features, mean, std, train_loader, val_loader = load_training_data(config, device)
    train_dataset = train_loader.dataset
    for index in range(count):
        seed = int(config.get('random_seed', 42)) + index
        random.seed(seed); torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
            generator=torch.Generator().manual_seed(seed),
        )
        model = ContactCNNWithNormalization(EnsembleTCN(
            config['window_size'], num_features, config.get('tcn_num_channels', 64),
            config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), config.get('tcn_dropout', .2),
        ), global_mean=mean, global_std=std).to(device)
        EnsembleTrainer(model, {**config, 'member_index': index, 'member_seed': seed}, os.path.join(root_dir, f'member_{index:02d}')).train(train_loader, val_loader)


if __name__ == '__main__':
    main()
