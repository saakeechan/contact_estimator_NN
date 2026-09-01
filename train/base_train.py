"""Shared training workflow for contact-velocity models."""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.plot_loss import generate_training_summary
from utils.data_handler import contact_dataset


class ONNXInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, velocity, _, covariance, contact = self.model(x)
        return velocity, covariance, contact


def save_onnx_model(model, checkpoint_path, window_size):
    was_training = model.training
    try:
        onnx_path = checkpoint_path.replace('.pt', '.onnx')
        device = next(model.parameters()).device
        model.eval()
        example_input = torch.randn(1, window_size, model.base_model.num_features, device=device)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings(
                'ignore', message=r'`isinstance\(treespec, LeafSpec\)` is deprecated.*',
                category=FutureWarning,
            )
            torch.onnx.export(
                ONNXInferenceWrapper(model), example_input, onnx_path, export_params=True,
                opset_version=18, input_names=['input'],
                output_names=['velocity_output', 'covariance_output', 'contact_output'],
                dynamic_axes={name: {0: 'batch_size'} for name in (
                    'input', 'velocity_output', 'covariance_output', 'contact_output'
                )}, verbose=False,
            )
        print(f'  ✓ ONNX inference model saved: {onnx_path}')
    except Exception as error:
        print(f'  ⚠ Warning: Failed to save ONNX model: {error}')
    finally:
        model.train(was_training)


def save_tcn_last_timestep_umap(dataloader, model, output_path, random_seed=42, max_samples=5000):
    base_model = getattr(model, 'base_model', model)
    if not hasattr(base_model, 'tcn_backbone'):
        print('Skipping TCN UMAP: selected model has no TCN backbone.')
        return
    from umap import UMAP
    import matplotlib.pyplot as plt

    latents, contacts, velocity_norms = [], [], []
    hook = base_model.tcn_backbone.register_forward_hook(
        lambda _, __, output: latents.append(output[:, :, -1].detach().cpu())
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sample in dataloader:
                model(sample['data'], return_sequence=False)
                contacts.append(sample['label'].detach().cpu())
                velocity_norms.append(sample['velocity'][:, -1].norm(dim=-1).detach().cpu())
                if sum(latent.shape[0] for latent in latents) >= max_samples:
                    break
    finally:
        hook.remove()
        model.train(was_training)
    if not latents:
        print('Skipping TCN UMAP: no validation samples.')
        return
    latent_array = torch.cat(latents)[:max_samples].numpy()
    if len(latent_array) < 3:
        print('Skipping TCN UMAP: need at least 3 validation samples.')
        return
    contact_array = torch.cat(contacts).reshape(-1)[:max_samples].numpy()
    velocity_norm_array = torch.cat(velocity_norms).reshape(-1)[:max_samples].numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='n_jobs value .* overridden.*')
        embedding = UMAP(n_components=2, random_state=random_seed).fit_transform(latent_array)
    for suffix, colors, title in (
        ('', contact_array, 'Final-timestep TCN latent UMAP'),
        ('_dv_norm', velocity_norm_array, 'Final-timestep TCN latent UMAP'),
    ):
        figure, axis = plt.subplots(figsize=(8, 6))
        if suffix:
            points = axis.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='viridis', s=8, alpha=0.7)
            figure.colorbar(points, ax=axis, label='Ground-truth body-velocity norm (m/s)')
        else:
            for state, color, label in ((0, 'tab:blue', 'No contact'), (1, 'tab:orange', 'Contact')):
                mask = colors == state
                axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.7, color=color, label=label)
            axis.legend()
        axis.set(title=title, xlabel='UMAP 1', ylabel='UMAP 2')
        figure.tight_layout()
        path = os.path.splitext(output_path)[0] + suffix + '.png'
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f'TCN UMAP saved to: {path}')


class BaseTrainer:
    """Model-independent epoch, evaluation, checkpoint, and reporting workflow."""

    logs_dir = 'logs'
    loss_description = 'Gaussian negative log-likelihood'

    def __init__(self, model, config, run_dir=None):
        self.model, self.config = model, config
        self.run_dir = run_dir or os.path.join(self.logs_dir, f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}")
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, 'network_params.yaml'), 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
        self.config['model_save_path'] = os.path.join(self.run_dir, 'model')
        self.config['log_writer_path'] = os.path.join(self.run_dir, 'tensorboard')
        self.contact_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
        self.contact_weight = float(config.get('contact_weight', 1.0))
        self.velocity_weight = float(config.get('velocity_weight', 1.0))
        self.use_dense_supervision = config.get('use_dense_supervision', False)

    def forward(self, inputs, dense):
        return self.model(inputs, return_sequence=dense)

    def to_model_device(self, sample):
        """Accept ordinary CPU DataLoaders as well as preloaded device datasets."""
        device = next(self.model.parameters()).device
        return {name: value.to(device) if torch.is_tensor(value) else value for name, value in sample.items()}

    def velocity_loss(self, outputs, velocity, dense):
        raise NotImplementedError

    def before_training(self, train_dataloader):
        pass

    def on_epoch_start(self, epoch):
        pass

    def training_velocity_loss(self, outputs, velocity, dense):
        return self.velocity_loss(outputs, velocity, dense)

    def after_training(self, train_dataloader):
        pass

    def after_final_checkpoint(self, checkpoint_path):
        pass

    def evaluate(self, dataloader):
        was_training = self.model.training
        self.model.eval()
        try:
            return self._evaluate(dataloader)
        finally:
            self.model.train(was_training)

    def _evaluate(self, dataloader):
        totals = dict(correct=0, data=0, contact_loss=0.0, velocity_loss=0.0, squared_error=0.0, variance=0.0)
        error_sum = contact_count = None
        with torch.no_grad():
            for sample in dataloader:
                sample = self.to_model_device(sample)
                velocity = sample['velocity'][:, -1]
                outputs = self.forward(sample['data'], False)
                _, velocity_output, _, covariance_output, contact_output = outputs[:5]
                contact_mask = (sample['label'] == 1).float().unsqueeze(-1)
                totals['contact_loss'] += self.contact_criterion(contact_output, sample['label']).item()
                loss = self.velocity_loss(outputs, velocity, False)
                totals['velocity_loss'] += (loss * contact_mask).sum().item() / (contact_mask.sum().item() * velocity.shape[-1] + 1e-8)
                if contact_mask.sum() > 0:
                    errors = (velocity_output - velocity).abs()
                    totals['squared_error'] += ((velocity_output - velocity).square() * contact_mask).sum().item()
                    totals['variance'] += (covariance_output * contact_mask).sum().item()
                    if error_sum is None:
                        error_sum, contact_count = torch.zeros_like(errors[0]), torch.zeros_like(sample['label'][0])
                    error_sum += (errors * contact_mask).sum(dim=0)
                    contact_count += contact_mask.squeeze(-1).sum(dim=0)
                prediction = (contact_output > 0).float()
                totals['correct'] += (prediction == sample['label']).sum().item()
                totals['data'] += sample['label'].numel()
        batches = max(len(dataloader), 1)
        components = np.zeros((1, 3)) if error_sum is None else (error_sum / (contact_count[:, None] + 1e-8)).cpu().tolist()
        mae = 0.0 if error_sum is None else error_sum.sum().item() / (contact_count.sum().item() * 3 + 1e-8)
        return {
            'contact_acc': totals['correct'] / totals['data'] if totals['data'] else 0.0,
            'contact_loss': totals['contact_loss'] / batches, 'velocity_loss': totals['velocity_loss'] / batches,
            'velocity_mae': mae, 'velocity_mae_components': components,
            'total_variance_calibration_ratio': totals['squared_error'] / (totals['variance'] + 1e-8),
        }

    def _checkpoint(self, epoch, train_loss, train_metrics, val_metrics, suffix):
        path = f"{self.config['model_save_path']}_{suffix}.pt"
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss': train_loss,
                    'contact_acc': train_metrics['contact_acc'], 'velocity_mae': train_metrics['velocity_mae'],
                    'val_loss': val_metrics['contact_loss'] + val_metrics['velocity_loss'],
                    'val_contact_acc': val_metrics['contact_acc'], 'val_velocity_mae': val_metrics['velocity_mae']}, path)
        save_onnx_model(self.model, path, self.config['window_size'])
        return path

    def train(self, train_dataloader, val_dataloader):
        writer, started = SummaryWriter(self.config['log_writer_path']), time.time()
        print(f'VELOCITY LOSS: {self.loss_description}')
        self.before_training(train_dataloader)
        best = {'loss': float('inf'), 'velocity': float('inf'), 'contact': 0.0}
        for epoch in range(self.config['num_epoch']):
            self.on_epoch_start(epoch)
            self.model.train()
            sums = dict(total=0.0, contact=0.0, velocity=0.0)
            progress = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{self.config["num_epoch"]}', unit='batch')
            for index, sample in enumerate(progress):
                sample = self.to_model_device(sample)
                velocity = sample['velocity'] if self.use_dense_supervision else sample['velocity'][:, -1]
                contact = sample['label_seq'] if self.use_dense_supervision else sample['label']
                outputs = self.forward(sample['data'], self.use_dense_supervision)
                contact_loss = self.contact_criterion(outputs[4], sample['label'])
                velocity_loss = self.training_velocity_loss(outputs, velocity, self.use_dense_supervision)
                mask = (contact == 1).float().unsqueeze(-1)
                velocity_loss = (velocity_loss * mask).sum() / (mask.sum() * velocity.shape[-1] + 1e-8)
                loss = self.contact_weight * contact_loss + self.velocity_weight * velocity_loss
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                for key, value in (('total', loss), ('contact', contact_loss), ('velocity', velocity_loss)):
                    sums[key] += value.item()
                progress.set_postfix(loss=f'{loss.item():.8f}')
            train_metrics, val_metrics = self.evaluate(train_dataloader), self.evaluate(val_dataloader)
            averages = {key: value / len(train_dataloader) for key, value in sums.items()}
            for scope, metrics in (('training', {'total_loss': averages['total'], 'contact_loss': averages['contact'], 'velocity_loss': averages['velocity'], 'contact_accuracy': train_metrics['contact_acc'], 'velocity_mae': train_metrics['velocity_mae']}), ('validation', {'total_loss': val_metrics['contact_loss'] + val_metrics['velocity_loss'], 'contact_loss': val_metrics['contact_loss'], 'velocity_loss': val_metrics['velocity_loss'], 'contact_accuracy': val_metrics['contact_acc'], 'velocity_mae': val_metrics['velocity_mae']})):
                for name, value in metrics.items(): writer.add_scalar(f'{scope}/{name}', value, epoch)
            writer.add_scalar('validation/total_variance_calibration_ratio', val_metrics['total_variance_calibration_ratio'], epoch)
            val_loss = val_metrics['contact_loss'] + val_metrics['velocity_loss']
            for key, value, suffix, comparison in (('contact', val_metrics['contact_acc'], 'best_val_contact_acc', lambda a, b: a > b), ('velocity', val_metrics['velocity_mae'], 'best_val_velocity', lambda a, b: a < b), ('loss', val_loss, 'best_val_loss', lambda a, b: a < b)):
                if comparison(value, best[key]): best[key] = value; self._checkpoint(epoch, averages['total'], train_metrics, val_metrics, suffix)
            print(f"Finished epoch {epoch + 1}/{self.config['num_epoch']} | train MAE {train_metrics['velocity_mae']:.4f} | val MAE {val_metrics['velocity_mae']:.4f}")
        self.after_training(train_dataloader)
        final_path = self._checkpoint(epoch, averages['total'], train_metrics, val_metrics, 'final_epoch')
        self.after_final_checkpoint(final_path)
        writer.close()
        elapsed = time.time() - started
        if self.config.get('save_umap_visualization', False):
            save_tcn_last_timestep_umap(val_dataloader, self.model, os.path.join(self.run_dir, 'tcn_last_timestep_umap.png'), self.config.get('random_seed', 42))
        generate_training_summary(run_dir=self.run_dir, config=self.config,
            train_metrics={'train_loss': averages['total'], 'train_contact_acc': train_metrics['contact_acc'], 'train_velocity_mae': train_metrics['velocity_mae'], 'val_loss': val_metrics['contact_loss'] + val_metrics['velocity_loss'], 'val_contact_acc': val_metrics['contact_acc'], 'val_velocity_mae': val_metrics['velocity_mae']},
            val_metrics=val_metrics, best_metrics={'best_val_loss': best['loss'], 'best_val_contact_acc': best['contact'], 'best_val_velocity_mae': best['velocity']}, train_time_seconds=elapsed,
            checkpoint_paths={'best_val_loss': f"{self.config['model_save_path']}_best_val_loss.pt", 'best_val_contact_acc': f"{self.config['model_save_path']}_best_val_contact_acc.pt", 'best_val_velocity': f"{self.config['model_save_path']}_best_val_velocity.pt", 'final_epoch': final_path})


def load_training_data(config, device, seed=None):
    """Load run-disjoint splits and normalization statistics from training windows only."""
    metadata_path = os.path.join(config['data_folder'], 'all_data_metadata.npy')
    num_features = np.load(metadata_path, allow_pickle=True).item()['num_features'] if os.path.exists(metadata_path) else config.get('num_features', 25)
    dataset = contact_dataset(os.path.join(config['data_folder'], 'all_data.npy'), os.path.join(config['data_folder'], 'all_labels.npy'), config['window_size'], device=device)
    run_ids = np.unique(dataset.window_to_run_id)
    if len(run_ids) < 3:
        raise ValueError(f'Need at least 3 runs for train/val/test split, but only have {len(run_ids)}.')
    if config.get('shuffle'):
        np.random.default_rng(config.get('random_seed', 42)).shuffle(run_ids)
    train_count = max(1, int(config.get('train_ratio', .7) * len(run_ids)))
    val_count = max(1, int(config.get('val_ratio', .15) * len(run_ids)))
    train_indices = dataset.get_windows_by_run_ids(set(run_ids[:train_count]))
    val_indices = dataset.get_windows_by_run_ids(set(run_ids[train_count:train_count + val_count]))
    if not train_indices or not val_indices:
        raise ValueError(
            f'Run split produced {len(train_indices)} training and {len(val_indices)} validation windows. '
            'Adjust the split ratios or collect longer runs.'
        )
    train_data = torch.cat([dataset[index]['data'].cpu() for index in train_indices])
    low, high = torch.quantile(train_data, .01, dim=0, keepdim=True), torch.quantile(train_data, .99, dim=0, keepdim=True)
    clipped = train_data.clamp(low, high)
    mean = clipped.mean(dim=0, keepdim=True).unsqueeze(0).to(device)
    std = clipped.std(dim=0, keepdim=True).unsqueeze(0).clamp_min(1e-8).to(device)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=config['batch_size'], shuffle=config['shuffle'], generator=generator)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=config['batch_size'], shuffle=False)
    return num_features, mean, std, train_loader, val_loader
