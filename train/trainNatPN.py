import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from natpn.nn import BayesianLoss

from contact_cnn import ContactCNNWithNormalization, TCN
from train.base_train import BaseTrainer, load_training_data


def natpn_loss(posteriors, target, loss_fn):
    return torch.stack([loss_fn(posterior, target[..., 0, axis]) for axis, posterior in enumerate(posteriors)], dim=-1).unsqueeze(-2)


class NatPNTrainer(BaseTrainer):
    logs_dir = 'logsNatPN'

    def __init__(self, model, config):
        super().__init__(model, config)
        self.loss_type = config.get('velocity_loss', 'bayesian')
        if self.loss_type not in {'bayesian', 'gaussian_nll'}:
            raise ValueError("velocity_loss must be 'bayesian' or 'gaussian_nll'.")
        self.loss_fn = BayesianLoss(float(config.get('natpn_entropy_weight', 1e-5)), reduction='none')
        self.mse_warmup_epochs = int(config.get('mse_warmup_epochs', 0))
        self.use_mse_warmup = False
        self.loss_description = 'NatPN Bayesian loss' if self.loss_type == 'bayesian' else 'Gaussian NLL on NatPN predictive mean and variance'

    def forward(self, inputs, dense):
        return self.model(inputs, return_sequence=dense, return_posteriors=True)

    def velocity_loss(self, outputs, velocity, dense):
        mean, variance, posteriors = (outputs[0], outputs[2], outputs[5]) if dense else (outputs[1], outputs[3], outputs[5])
        if dense:
            mean, variance = mean.permute(0, 3, 1, 2), variance.permute(0, 3, 1, 2)
        return F.gaussian_nll_loss(mean, velocity, variance, reduction='none') if self.loss_type == 'gaussian_nll' else natpn_loss(posteriors, velocity, self.loss_fn)

    def on_epoch_start(self, epoch):
        self.use_mse_warmup = self.loss_type == 'gaussian_nll' and epoch < self.mse_warmup_epochs

    def training_velocity_loss(self, outputs, velocity, dense):
        if not self.use_mse_warmup:
            return self.velocity_loss(outputs, velocity, dense)
        mean = outputs[0] if dense else outputs[1]
        if dense:
            mean = mean.permute(0, 3, 1, 2)
        return F.mse_loss(mean, velocity, reduction='none')

    def _fit_flows(self, dataloader, epochs, label):
        if epochs <= 0: return
        optimizer = optim.Adam(self.model.base_model.velocity_heads.flow.parameters(), lr=self.config['init_lr'])
        for epoch in range(epochs):
            for sample in dataloader:
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(sample['data'], return_sequence=self.use_dense_supervision, return_flow_nll=True)
                loss.backward(); optimizer.step()
            print(f'{label} {epoch + 1}/{epochs}')

    def before_training(self, train_dataloader):
        self._fit_flows(train_dataloader, int(self.config.get('natpn_warmup_epochs', 3)), 'NatPN warmup')

    def after_training(self, train_dataloader):
        self._fit_flows(train_dataloader, int(self.config.get('natpn_finetune_epochs', self.config['num_epoch'])), 'NatPN fine-tune')

    def after_final_checkpoint(self, checkpoint_path):
        shutil.copyfile(checkpoint_path, self.config['model_save_path'] + '_natpn_finetuned.pt')


def main():
    parser = argparse.ArgumentParser(description='Train NatPN model')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '../config/network_params.yaml'))
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(__file__), '../config/NatPN_params.yaml')) as file:
        config = yaml.safe_load(file) or {}
    with open(args.config_name) as file:
        config.update(yaml.safe_load(file) or {})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features, mean, std, train_loader, val_loader = load_training_data(config, device)
    if config.get('model_architecture', 'tcn').lower() != 'tcn':
        raise ValueError("NatPN training requires model_architecture: 'tcn'.")
    model = ContactCNNWithNormalization(TCN(
        config['window_size'], num_features, config.get('tcn_num_channels', 64),
        config.get('tcn_kernel_size', 3), config.get('tcn_num_blocks', 5), config.get('tcn_dropout', .2),
        config.get('natpn_flow_type', 'radial'), config.get('natpn_flow_layers', 8), config.get('natpn_certainty_budget', 'normal'),
    ), global_mean=mean, global_std=std).to(device)
    NatPNTrainer(model, config).train(train_loader, val_loader)


if __name__ == '__main__':
    main()
