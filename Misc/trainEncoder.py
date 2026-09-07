import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append('.')

from contact_cnn import DenoisingTCNAutoencoder
from natpn.nn.flow import RadialFlow
from natpn.nn.scaler import EvidenceScaler
from utils.data_handler import contact_dataset


class VariationalTCNAutoencoder(DenoisingTCNAutoencoder):
    """TCN VAE with one Gaussian latent vector per input timestep."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = self.input_proj.out_channels
        self.mu = nn.Conv1d(channels, channels, kernel_size=1)
        self.logvar = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = (x - self.global_mean) / (self.global_std + self.eps)
        features = self.tcn_backbone(self.input_proj(x.permute(0, 2, 1)))
        mu, logvar = self.mu(features), self.logvar(features)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        reconstruction = self.decoder(latent.permute(0, 2, 1))
        return reconstruction * (self.global_std + self.eps) + self.global_mean, mu, logvar


class InputLatentNatPNDensity(nn.Module):
    """NatPN's density/evidence component over a frozen DAE/VAE latent."""
    def __init__(self, latent_dim, flow_layers, certainty_budget):
        super().__init__()
        self.flow = RadialFlow(latent_dim, flow_layers)
        self.scaler = EvidenceScaler(latent_dim, certainty_budget)

    def forward(self, latent):
        log_prob = self.flow(latent)
        return log_prob, self.scaler(log_prob)


def encoder_latent(model, window):
    """Return the deterministic TCN latent used by the DAE/VAE density model."""
    normalized = (window - model.global_mean) / (model.global_std + model.eps)
    features = model.tcn_backbone(model.input_proj(normalized.permute(0, 2, 1)))
    return model.mu(features) if isinstance(model, VariationalTCNAutoencoder) else features


def input_natpn_nll(encoder, density, windows):
    """Negative log-density of final input latents; encoder remains frozen."""
    with torch.no_grad():
        latent = encoder_latent(encoder, windows)[:, :, -1]
    log_prob, _ = density(latent)
    return -log_prob.mean()


def fit_input_natpn_density(encoder, train_dataloader, val_dataloader, config, run_dir):
    """Fit a radial-flow input density after reconstruction training has finished."""
    epochs = int(config['input_natpn_epochs'])
    latent_dim = encoder.input_proj.out_channels
    density = InputLatentNatPNDensity(
        latent_dim, config['input_natpn_flow_layers'], config['input_natpn_certainty_budget']
    ).to(next(encoder.parameters()).device)
    optimizer = optim.Adam(density.flow.parameters(), lr=config['input_natpn_lr'])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    for epoch in range(1, epochs + 1):
        density.train()
        train_nll_sum = 0.0
        for sample in tqdm(train_dataloader, desc=f'Input NatPN flow {epoch}/{epochs}'):
            optimizer.zero_grad(set_to_none=True)
            loss = input_natpn_nll(encoder, density, sample['data'])
            loss.backward()
            optimizer.step()
            train_nll_sum += loss.item()

        density.eval()
        with torch.no_grad():
            val_nll = sum(
                input_natpn_nll(encoder, density, sample['data']).item() for sample in val_dataloader
            ) / len(val_dataloader)
        print(f'Input NatPN flow {epoch:03d} | train_nll={train_nll_sum / len(train_dataloader):.4f} | val_nll={val_nll:.4f}')

    checkpoint_path = os.path.join(run_dir, 'encoder_input_natpn.pt')
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'input_natpn_state_dict': density.state_dict(),
        'encoder_type': config['encoder_type'],
        'num_features': encoder.num_features,
        'latent_dim': latent_dim,
        'input_natpn_flow_layers': config['input_natpn_flow_layers'],
        'input_natpn_certainty_budget': config['input_natpn_certainty_budget'],
        'encoder_config': {
            key: config.get(key) for key in ('window_size', 'tcn_kernel_size', 'tcn_num_blocks', 'tcn_dropout')
        },
    }, checkpoint_path)
    print(f'Frozen encoder and input-latent NatPN flow saved to: {checkpoint_path}')
    return checkpoint_path


def train_task_from_input_density(config, run_dir, input_natpn_checkpoint):
    """Start task training with the encoder-flow artifact from this run."""
    task_config = dict(config)
    task_config.update(
        natpn_evidence_source='input',
        input_natpn_checkpoint=os.path.abspath(input_natpn_checkpoint),
        use_dense_supervision=False,
        velocity_loss=config['input_velocity_loss'],
        mse_warmup_epochs=int(config['num_epoch']) // 2,
        input_epistemic_scale=float(config.get('input_epistemic_scale', 1.0)),
    )
    task_config_path = os.path.join(run_dir, 'task_network_params.yaml')
    with open(task_config_path, 'w') as config_file:
        yaml.safe_dump(task_config, config_file, default_flow_style=False, sort_keys=False)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'Starting task training with input-latent NatPN: {task_config_path}')
    subprocess.run(
        [sys.executable, os.path.join(project_root, 'src', 'trainNatPN.py'), '--config_name', task_config_path],
        cwd=project_root,
        check=True,
    )


def reconstruction_loss(model, clean_window, config):
    """Return reconstruction loss; only the DAE corrupts its input."""
    if config['encoder_type'] == 'DAE':
        noisy_window = clean_window + torch.randn_like(clean_window) * model.global_std * config['dae_noise_std']
        return F.mse_loss(model(noisy_window), clean_window)

    reconstruction, mu, logvar = model(clean_window)
    kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return F.mse_loss(reconstruction, clean_window) + config['vae_beta'] * kl_loss


def save_tcn_last_timestep_umap(dataloader, model, output_path, random_seed=42, max_samples=5000):
    """Save final-timestep TCN latents colored by contact, velocity, and cmd_vel_x."""
    from umap import UMAP
    import matplotlib.pyplot as plt

    latents, contacts, velocity_norms, command_velocities = [], [], [], []
    hook = model.tcn_backbone.register_forward_hook(
        lambda _, __, output: latents.append(output[:, :, -1].detach().cpu())
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sample in dataloader:
                model(sample['data'])
                contacts.append(sample['label'].detach().cpu())
                velocity_norms.append(sample['velocity'][:, -1].norm(dim=-1).detach().cpu())
                command_velocities.append(sample['data'][:, -1, -1].detach().cpu())
                if sum(latent.shape[0] for latent in latents) >= max_samples:
                    break
    finally:
        hook.remove()
        model.train(was_training)

    latent_array = torch.cat(latents)[:max_samples].numpy()
    contact_array = torch.cat(contacts).reshape(-1)[:max_samples].numpy()
    velocity_norm_array = torch.cat(velocity_norms).reshape(-1)[:max_samples].numpy()
    command_velocity_array = torch.cat(command_velocities).reshape(-1)[:max_samples].numpy()
    if len(latent_array) < 3:
        print('Skipping TCN UMAP: need at least 3 validation windows.')
        return

    embedding = UMAP(n_components=2, random_state=random_seed).fit_transform(latent_array)
    figure, axis = plt.subplots(figsize=(8, 6))
    for state, color, label in ((0, 'tab:blue', 'No contact'), (1, 'tab:orange', 'Contact')):
        mask = contact_array == state
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.7, color=color, label=label)
    axis.set(title='Final-timestep DAE TCN latent UMAP', xlabel='UMAP 1', ylabel='UMAP 2')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    velocity_norm_output_path = os.path.splitext(output_path)[0] + '_dv_norm.png'
    figure, axis = plt.subplots(figsize=(8, 6))
    points = axis.scatter(embedding[:, 0], embedding[:, 1], c=velocity_norm_array, cmap='viridis', s=8, alpha=0.7)
    figure.colorbar(points, ax=axis, label='Ground-truth body-velocity norm (m/s)')
    axis.set(title='Final-timestep DAE TCN latent UMAP', xlabel='UMAP 1', ylabel='UMAP 2')
    figure.tight_layout()
    figure.savefig(velocity_norm_output_path, dpi=150)
    plt.close(figure)

    command_velocity_output_path = os.path.splitext(output_path)[0] + '_cmd_vel_x.png'
    figure, axis = plt.subplots(figsize=(8, 6))
    points = axis.scatter(embedding[:, 0], embedding[:, 1], c=command_velocity_array, cmap='viridis', s=8, alpha=0.7)
    figure.colorbar(points, ax=axis, label='Command velocity x (m/s)')
    axis.set(title='Final-timestep DAE TCN latent UMAP', xlabel='UMAP 1', ylabel='UMAP 2')
    figure.tight_layout()
    figure.savefig(command_velocity_output_path, dpi=150)
    plt.close(figure)
    print(f'TCN latent UMAPs saved to: {output_path}, {velocity_norm_output_path}, and {command_velocity_output_path}')


def train(model, train_dataloader, val_dataloader, config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    encoder_type = config['encoder_type'].lower()
    run_dir = _ROOT / "logs" / "logsEncoder" / f"{encoder_type}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "network_params.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    best_checkpoint = 'dae_best_val_mse.pt' if config['encoder_type'] == 'DAE' else 'vae_best_val_loss.pt'
    final_checkpoint = 'dae_final.pt' if config['encoder_type'] == 'DAE' else 'vae_final.pt'
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    best_val_loss = float('inf')

    print(f"Training {config['encoder_type']} in: {run_dir}")
    if config['encoder_type'] == 'DAE':
        print(f"Raw-input noise: {config['dae_noise_std']} × training feature standard deviation")
    for epoch in range(config['num_epoch']):
        model.train()
        train_loss_sum = 0.0
        for sample in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_epoch']}"):
            clean_window = sample['data']
            optimizer.zero_grad()
            loss = reconstruction_loss(model, clean_window, config)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for sample in val_dataloader:
                clean_window = sample['data']
                val_loss_sum += reconstruction_loss(model, clean_window, config).item()

        train_loss = train_loss_sum / len(train_dataloader)
        val_loss = val_loss_sum / len(val_dataloader)
        metric = 'mse' if config['encoder_type'] == 'DAE' else 'loss'
        writer.add_scalar(f'reconstruction/train_{metric}', train_loss, epoch)
        writer.add_scalar(f'reconstruction/val_{metric}', val_loss, epoch)
        print(f"Epoch {epoch + 1}/{config['num_epoch']}: train {metric} {train_loss:.8f}, val {metric} {val_loss:.8f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_mse': train_loss,
                'val_mse': val_loss,
            }, os.path.join(run_dir, best_checkpoint))

    torch.save({
        'epoch': config['num_epoch'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mse': val_loss,
    }, os.path.join(run_dir, final_checkpoint))
    # Density must be fitted on the best reconstruction encoder, not on the final epoch by default.
    model.load_state_dict(torch.load(os.path.join(run_dir, best_checkpoint), map_location=next(model.parameters()).device)['model_state_dict'])
    input_natpn_checkpoint = fit_input_natpn_density(model, train_dataloader, val_dataloader, config, run_dir)
    writer.close()
    save_tcn_last_timestep_umap(
        val_dataloader, model, os.path.join(run_dir, 'tcn_last_timestep_umap.png'), config.get('random_seed', 42)
    )
    print(f"Best validation reconstruction {'MSE' if config['encoder_type'] == 'DAE' else 'loss'}: {best_val_loss:.8f}")
    if config['train_task_after_encoder']:
        train_task_from_input_density(config, run_dir, input_natpn_checkpoint)


def main():
    parser = argparse.ArgumentParser(description='Train a DAE or VAE TCN autoencoder')
    parser.add_argument('--config_name', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'network_params.yaml'))
    args = parser.parse_args()
    natpn_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'NatPN_params.yaml')
    with open(natpn_config_path) as config_file:
        natpn_config = yaml.safe_load(config_file) or {}
    with open(args.config_name) as config_file:
        config = {**natpn_config, **(yaml.safe_load(config_file) or {})}
    config['encoder_type'] = config.get('encoder_type', 'DAE').upper()
    if config['encoder_type'] not in {'DAE', 'VAE'}:
        raise ValueError("encoder_type must be 'DAE' or 'VAE'.")
    config['dae_noise_std'] = float(config.get('dae_noise_std', 0.05))
    config['vae_beta'] = float(config.get('vae_beta', 0.001))
    config['input_natpn_flow_layers'] = int(config.get('input_natpn_flow_layers', 8))
    config['input_natpn_certainty_budget'] = config.get('input_natpn_certainty_budget', 'normal')
    config['input_natpn_epochs'] = int(config.get('input_natpn_epochs', 3))
    config['input_natpn_lr'] = float(config.get('input_natpn_lr', config['init_lr']))
    config['train_task_after_encoder'] = bool(config.get('train_task_after_encoder', True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    torch.manual_seed(config.get('random_seed', 42))
    np.random.seed(config.get('random_seed', 42))

    metadata_path = os.path.join(config['data_folder'], 'all_data_metadata.npy')
    if os.path.exists(metadata_path):
        num_features = np.load(metadata_path, allow_pickle=True).item()['num_features']
    else:
        num_features = config.get('num_features', 25)
        print(f"Warning: metadata not found; using num_features={num_features}")

    dataset = contact_dataset(
        data_path=os.path.join(config['data_folder'], 'all_data.npy'),
        label_path=os.path.join(config['data_folder'], 'all_labels.npy'),
        window_size=config['window_size'],
        device=device,
    )
    run_ids = np.unique(dataset.window_to_run_id)
    if len(run_ids) < 2:
        raise ValueError('Need at least two runs for train/validation reconstruction.')
    if config.get('shuffle', True):
        np.random.shuffle(run_ids)

    train_run_count = max(1, int(config.get('train_ratio', 0.8) * len(run_ids)))
    train_run_count = min(train_run_count, len(run_ids) - 1)
    train_indices = dataset.get_windows_by_run_ids(set(run_ids[:train_run_count]))
    val_indices = dataset.get_windows_by_run_ids(set(run_ids[train_run_count:]))
    if not train_indices or not val_indices:
        raise ValueError('Train/validation split produced no windows.')

    train_windows = torch.cat([dataset[index]['data'].cpu() for index in train_indices])
    lower = torch.quantile(train_windows, 0.01, dim=0, keepdim=True)
    upper = torch.quantile(train_windows, 0.99, dim=0, keepdim=True)
    train_windows = torch.clamp(train_windows, min=lower, max=upper)
    global_mean = train_windows.mean(dim=0, keepdim=True).unsqueeze(0).to(device)
    global_std = train_windows.std(dim=0, keepdim=True).unsqueeze(0).to(device)
    global_std = torch.where(global_std == 0, torch.ones_like(global_std), global_std)

    train_dataloader = DataLoader(Subset(dataset, train_indices), batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(Subset(dataset, val_indices), batch_size=config['batch_size'], shuffle=False)
    model_class = DenoisingTCNAutoencoder if config['encoder_type'] == 'DAE' else VariationalTCNAutoencoder
    model = model_class(
        window_size=config['window_size'],
        num_features=num_features,
        tcn_num_channels=config.get('tcn_num_channels', 64),
        tcn_kernel_size=config.get('tcn_kernel_size', 3),
        tcn_num_blocks=config.get('tcn_num_blocks', 3),
        tcn_dropout=config.get('tcn_dropout', 0.2),
        global_mean=global_mean,
        global_std=global_std,
    ).to(device)
    train(model, train_dataloader, val_dataloader, config)


if __name__ == '__main__':
    main()
