import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import warnings
from datetime import datetime
import time

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *
from utils.plot_loss import generate_training_summary
from natpn.nn import BayesianLoss

from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*LeafSpec.*"
)


def natpn_loss(posteriors, target, loss_fn):
    """Return per-axis NatPN Bayesian losses in the existing velocity tensor layout."""
    losses = [loss_fn(posterior, target[..., 0, axis]) for axis, posterior in enumerate(posteriors)]
    return torch.stack(losses, dim=-1).unsqueeze(-2)


def supervised_velocity_loss(posteriors, mean, variance, target, loss_type, bayesian_loss, use_aleatoric_variance=False):
    if loss_type == 'mse':
        return torch.nn.functional.mse_loss(mean, target, reduction='none')
    if loss_type == 'gaussian_nll':
        if use_aleatoric_variance:
            variance = torch.stack([
                posterior.beta / (posterior.alpha - 1.0).clamp_min(1e-6) for posterior in posteriors
            ], dim=-1).unsqueeze(-2)
        return torch.nn.functional.gaussian_nll_loss(mean, target, variance, reduction='none')
    return natpn_loss(posteriors, target, bayesian_loss)


def optimize_natpn_flows(model, dataloader, config, epochs, label):
    """The input-density flow is pretrained and frozen by trainEncoder.py."""
    if epochs <= 0 or model.base_model.natpn_evidence_source == 'input':
        if epochs > 0:
            print(f'{label}: using frozen input-latent NatPN flow.')
        return
    flows = model.base_model.velocity_heads.models
    optimizer = optim.Adam(
        (parameter for natpn_model in flows for parameter in natpn_model.flow.parameters()),
        lr=config['init_lr'],
    )
    use_dense_supervision = config.get('use_dense_supervision', False)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for samples in tqdm(dataloader, desc=f'{label} flow {epoch}/{epochs}'):
            optimizer.zero_grad(set_to_none=True)
            flow_nll = model(
                samples['data'], return_sequence=use_dense_supervision, return_flow_nll=True
            )
            flow_nll.backward()
            optimizer.step()
            loss_sum += flow_nll.item()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'{label} {epoch:03d} | flow_nll={loss_sum / len(dataloader):.4f}')


def save_tcn_last_timestep_umap(dataloader, model, output_path, random_seed=42, max_samples=5000):
    """Save final-timestep TCN UMAPs colored by contact and ground-truth body-velocity norm."""
    base_model = getattr(model, 'base_model', model)
    if not hasattr(base_model, 'tcn_backbone'):
        print("Skipping TCN UMAP: selected model has no TCN backbone.")
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
                velocity_norms.append(sample['velocity'][:, -1, :, :].norm(dim=-1).detach().cpu())
                if sum(latent.shape[0] for latent in latents) >= max_samples:
                    break
    finally:
        hook.remove()
        model.train(was_training)

    if len(latents) == 0:
        print("Skipping TCN UMAP: no validation samples.")
        return

    latent_array = torch.cat(latents)[:max_samples].numpy()
    contact_array = torch.cat(contacts).reshape(-1)[:max_samples].numpy()
    velocity_norm_array = torch.cat(velocity_norms).reshape(-1)[:max_samples].numpy()
    if len(latent_array) < 3:
        print("Skipping TCN UMAP: need at least 3 validation samples.")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='n_jobs value .* overridden.*')
        embedding = UMAP(n_components=2, random_state=random_seed).fit_transform(latent_array)
    figure, axis = plt.subplots(figsize=(8, 6))
    for contact_state, color, label in ((0, 'tab:blue', 'No contact'), (1, 'tab:orange', 'Contact')):
        mask = contact_array == contact_state
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.7, color=color, label=label)
    axis.set(title='Final-timestep TCN latent UMAP', xlabel='UMAP 1', ylabel='UMAP 2')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"TCN UMAP saved to: {output_path}")

    velocity_norm_output_path = os.path.splitext(output_path)[0] + "_dv_norm.png"
    figure, axis = plt.subplots(figsize=(8, 6))
    points = axis.scatter(
        embedding[:, 0], embedding[:, 1], c=velocity_norm_array, cmap='viridis', s=8, alpha=0.7
    )
    figure.colorbar(points, ax=axis, label='Ground-truth body-velocity norm (m/s)')
    axis.set(title='Final-timestep TCN latent UMAP', xlabel='UMAP 1', ylabel='UMAP 2')
    figure.tight_layout()
    figure.savefig(velocity_norm_output_path, dpi=150)
    plt.close(figure)
    print(f"TCN velocity-norm UMAP saved to: {velocity_norm_output_path}")


def compute_accuracy(dataloader, model, contact_criterion=None, velocity_loss_fn=None, velocity_loss_type='bayesian'):
    """
    Compute left-foot contact and body-velocity metrics at the last timestep.
    
    Args:
        dataloader: DataLoader to evaluate
        model: The neural network model
        contact_criterion: Optional contact loss criterion (BCEWithLogitsLoss)
        velocity_loss_fn: Optional elementwise Gaussian NLL function
    
    Returns:
        dict with metrics: contact_acc, contact_loss, velocity_mae, velocity_loss
    """
    num_correct = 0
    num_data = 0
    contact_loss_sum = 0
    velocity_loss_sum = 0
    velocity_abs_error_sum = None
    velocity_contact_count = None
    total_variance_sum = 0.0
    squared_error_sum = 0.0
    
    # Track prediction distribution to detect bias
    num_pred_contact = 0
    num_pred_no_contact = 0
    num_gt_contact = 0
    num_gt_no_contact = 0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # [B, 1] left foot
            gt_velocity_seq = sample['velocity']  # [B, T, 1, 3]
            gt_velocity = gt_velocity_seq[:, -1, :, :]  # [B, 1, 3]

            velocity_seq, velocity_output, covariance_seq, covariance_output, contact_output, posteriors = model(
                input_data, return_sequence=False, return_posteriors=True
            )
            contact_prediction = (contact_output > 0).float()  # Binary predictions

            # Compute contact loss if criterion provided
            if contact_criterion is not None:
                contact_loss = contact_criterion(contact_output, gt_contact)
                contact_loss_sum += contact_loss.item()
            
            # Velocity loss masked to contact samples only
            contact_mask = (gt_contact == 1).float().unsqueeze(-1)  # [B, 1, 1]
            
            if velocity_loss_fn is not None:
                # Velocity loss on last timestep only, masked to contact samples only
                velocity_loss_each = supervised_velocity_loss(
                    posteriors, velocity_output, covariance_output, gt_velocity, velocity_loss_type, velocity_loss_fn,
                    model.base_model.natpn_evidence_source == 'input'
                )
                
                velocity_loss_masked = (
                    velocity_loss_each * contact_mask
                ).sum() / (contact_mask.sum() * gt_velocity.shape[-1] + 1e-8)
                
                velocity_loss_sum += velocity_loss_masked.item()
            
            # MAE for velocity (only on contact samples, last timestep)
            if contact_mask.sum() > 0:
                velocity_errors = torch.abs(velocity_output - gt_velocity)  # [B, 1, 3]
                squared_error_sum += ((velocity_output - gt_velocity).square() * contact_mask).sum().item()
                total_variance_sum += (covariance_output * contact_mask).sum().item()
                if velocity_abs_error_sum is None:
                    velocity_abs_error_sum = torch.zeros_like(velocity_errors[0])
                    velocity_contact_count = torch.zeros_like(gt_contact[0])
                velocity_abs_error_sum += (velocity_errors * contact_mask).sum(dim=0)
                velocity_contact_count += contact_mask.squeeze(-1).sum(dim=0)

            # Contact accuracy
            num_correct += (contact_prediction == gt_contact).sum().item()
            num_data += gt_contact.numel()
            
            # Track prediction distribution
            num_pred_contact += contact_prediction.sum().item()
            num_pred_no_contact += (1 - contact_prediction).sum().item()
            num_gt_contact += gt_contact.sum().item()
            num_gt_no_contact += (1 - gt_contact).sum().item()

    # Safeguard against empty dataloaders (no validation windows)
    contact_acc = (num_correct / num_data) if num_data > 0 else 0.0
    num_batches = len(dataloader) if len(dataloader) > 0 else 1

    if velocity_abs_error_sum is None:
        velocity_mae_components = np.zeros((1, 3))
        velocity_mae = 0.0
    else:
        velocity_mae_components = (velocity_abs_error_sum / (velocity_contact_count[:, None] + 1e-8)).cpu().tolist()
        velocity_mae = float(velocity_abs_error_sum.sum().item() / (velocity_contact_count.sum().item() * 3 + 1e-8))

    metrics = {
        'contact_acc': contact_acc,
        'contact_loss': contact_loss_sum / num_batches if contact_criterion else 0,
        'velocity_mae': velocity_mae,
        'velocity_loss': velocity_loss_sum / num_batches if velocity_loss_fn else 0,
        'velocity_mae_components': velocity_mae_components,
        'num_pred_contact': num_pred_contact,
        'num_gt_contact': num_gt_contact,
        'total_variance_calibration_ratio': (
            squared_error_sum / (total_variance_sum + 1e-8)
            if model.base_model.natpn_evidence_source == 'input' else None
        ),
    }
    
    return metrics


class ONNXInferenceWrapper(nn.Module):
    """Expose only tensor-valued inference outputs to the ONNX exporter."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, velocity, _, covariance, contact = self.model(x)
        return velocity, covariance, contact


def save_onnx_model(model, checkpoint_path, window_size):
    """
    Save ONNX version of the model for C++ deployment.
    The inference model has three outputs:
        - velocity_output: (batch, 1, 3) - last-timestep body-velocity prediction
        - covariance_output: (batch, 1, 3) - diagonal last-timestep body-velocity variances
        - contact_output: (batch, 1) - left-foot contact logit
    """
    try:
        import warnings
        
        # Create ONNX path (replace .pt with .onnx)
        onnx_path = checkpoint_path.replace('.pt', '.onnx')
        
        device = next(model.parameters()).device
        model.eval()
        
        # num_features is read from model's base_model attribute
        num_features = model.base_model.num_features
        example_input = torch.randn(1, window_size, num_features).to(device)
        
        # Suppress warnings and use standard export
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            torch.onnx.export(
                ONNXInferenceWrapper(model),
                example_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                input_names=['input'],
                output_names=['velocity_output', 'covariance_output', 'contact_output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'velocity_output': {0: 'batch_size'},
                    'covariance_output': {0: 'batch_size'},
                    'contact_output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print(f"  ✓ ONNX inference model saved: {onnx_path}")
        
    except Exception as e:
        print(f"  ⚠ Warning: Failed to save ONNX model: {e}")


def train(model, train_dataloader, val_dataloader, config):
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config copy to run directory
    config_copy_path = os.path.join(run_dir, "network_params.yaml")
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n{'='*60}")
    print(f"Training Run Directory: {run_dir}")
    print(f"Config saved to: {config_copy_path}")
    print(f"{'='*60}\n")
    
    # Set paths to use run directory
    config['model_save_path'] = os.path.join(run_dir, "model")
    config['log_writer_path'] = os.path.join(run_dir, "tensorboard")
    
    # Track training start time
    train_start_time = time.time()

    # Initialize TensorBoard writer for scalar metrics (loss, MAE plots)
    writer = SummaryWriter(config['log_writer_path'])

    # Loss functions
    # Contact: BCEWithLogitsLoss (binary classification)
    contact_criterion = nn.BCEWithLogitsLoss()
    
    velocity_loss_type = config.get('velocity_loss', 'bayesian')
    if velocity_loss_type not in {'bayesian', 'gaussian_nll'}:
        raise ValueError("velocity_loss must be 'bayesian' or 'gaussian_nll'.")
    mse_warmup_epochs = int(config.get('mse_warmup_epochs', 0))

    # Used only when velocity_loss is bayesian; input-evidence Gaussian NLL uses task aleatoric variance.
    velocity_loss_fn = BayesianLoss(float(config.get('natpn_entropy_weight', 1e-5)), reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    
    # Get loss weighting parameters
    contact_weight = float(config.get('contact_weight', 1.0))  # Weight for contact loss
    velocity_weight = float(config.get('velocity_weight', 1.0))  # Weight for velocity loss
    derivative_weight = float(config.get('derivative_weight', 0.0))  # Weight for derivative matching loss
    temporal_weight_power = float(config.get('temporal_weight_power', 0.0))  # Temporal weighting exponent
    use_dense_supervision = config.get('use_dense_supervision', False)
    
    # Print supervision mode
    print(f"\n{'='*60}")
    if use_dense_supervision:
        print(f"DENSE SUPERVISION MODE: Loss on full sequence")

    else:
        print(f"LAST TIMESTEP ONLY MODE: Loss on final output only")
    loss_description = (
        f'NatPN Bayesian loss (entropy weight={velocity_loss_fn.entropy_weight:g})'
        if velocity_loss_type == 'bayesian' else 'Gaussian NLL on NatPN predictive mean and variance'
    )
    print(f'VELOCITY LOSS: {loss_description}')
    if mse_warmup_epochs:
        print(f'VELOCITY LOSS WARMUP: MSE for the first {mse_warmup_epochs} epochs')
    natpn_warmup_epochs = int(config.get('natpn_warmup_epochs', 3))
    natpn_finetune_epochs = config.get('natpn_finetune_epochs')
    natpn_finetune_epochs = config['num_epoch'] if natpn_finetune_epochs is None else int(natpn_finetune_epochs)
    optimize_natpn_flows(model, train_dataloader, config, natpn_warmup_epochs, 'NatPN warmup')


    # best_acc = 0
    best_loss = 1000000000
    best_velocity_mae = 1000000000
    best_contact_acc = 0
    
    # Pre-compute temporal weights ONCE (before training loop) to avoid recomputation every batch
    device = next(model.parameters()).device
    if use_dense_supervision and temporal_weight_power > 0:
        T = config['window_size']
        timesteps = torch.arange(1, T + 1, dtype=torch.float32, device=device)
        temporal_weights = ((timesteps / T) ** temporal_weight_power).view(1, T, 1, 1)  # [1, T, 1, 1]
        
        # Print temporal weights info
        print(f"\n{'='*60}")
        print(f"Temporal weighting enabled (power={temporal_weight_power})")
        print(f"Weights for each timestep (oldest → newest):")
        weights_1d = temporal_weights.squeeze().cpu().numpy()
        print(f"  {weights_1d}")
        print(f"  First timestep weight: {weights_1d[0]:.4f}")
        print(f"  Last timestep weight: {weights_1d[-1]:.4f}")
        print(f"  Ratio (last/first): {weights_1d[-1]/weights_1d[0]:.2f}x")
        print(f"{'='*60}\n")
    else:
        temporal_weights = 1.0  # Uniform weighting
    
    for epoch in range(config['num_epoch']):
        epoch_loss_type = 'mse' if velocity_loss_type == 'gaussian_nll' and epoch < mse_warmup_epochs else velocity_loss_type
        print(f"Velocity loss: {'MSE warmup' if epoch_loss_type == 'mse' else loss_description}")
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved  = torch.cuda.memory_reserved(device)  / 1024**2
            print(f"[GPU] Epoch {epoch+1}: {allocated:.0f} MB allocated / {reserved:.0f} MB reserved")
        running_loss = 0.0  # For periodic printing
        loss_sum = 0.0  # For epoch average
        contact_loss_sum = 0.0
        velocity_loss_sum = 0.0
        derivative_loss_sum = 0.0
        
        model.train()
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            contact_label = samples['label']  # [B, 1] left foot
            contact_label_seq = samples['label_seq']  # [B, T, 1]
            velocity_label_seq = samples['velocity']  # [B, T, 1, 3]
            velocity_label = velocity_label_seq[:, -1, :, :]  # [B, 1, 3]
        

            optimizer.zero_grad()
            
            velocity_seq, velocity_output, covariance_seq, covariance_output, contact_output, posteriors = model(
                input_data, return_sequence=use_dense_supervision, return_posteriors=True
            )
            
            # Contact loss (binary classification at last timestep)
            contact_loss = contact_criterion(contact_output, contact_label)
            
            if use_dense_supervision:
                # DENSE SUPERVISION: Compute velocity loss on FULL SEQUENCE masked to contact only
                velocity_seq_permuted = velocity_seq.permute(0, 3, 1, 2)  # [B, T, 1, 3]
                
                covariance_seq_permuted = covariance_seq.permute(0, 3, 1, 2)  # [B, T, 1, 3]
                velocity_loss_elementwise = supervised_velocity_loss(
                    posteriors, velocity_seq_permuted, covariance_seq_permuted,
                    velocity_label_seq, epoch_loss_type, velocity_loss_fn,
                    model.base_model.natpn_evidence_source == 'input'
                )
                
                # Use full contact sequence for masking (dense supervision only at contact timesteps)
                contact_mask_seq = (contact_label_seq == 1).float().unsqueeze(-1)  # [B, T, 1, 1]
                
                # Apply temporal weighting (pre-computed before training loop)
                velocity_loss = (
                    velocity_loss_elementwise * contact_mask_seq * temporal_weights
                ).sum() / (((contact_mask_seq * temporal_weights).sum() * velocity_label_seq.shape[-1]) + 1e-8)
                
                # Derivative matching loss: match temporal dynamics (slopes/changes)
                # Only available in dense mode (requires full sequence)
                derivative_loss = torch.tensor(0.0, device=velocity_output.device)
                if derivative_weight > 0:
                    # Compute derivatives: Δy_t = y_t - y_{t-1}
                    pred_derivative = velocity_seq_permuted[:, 1:, :, :] - velocity_seq_permuted[:, :-1, :, :]
                    gt_derivative = velocity_label_seq[:, 1:, :, :] - velocity_label_seq[:, :-1, :, :]
                    
                    # Compute derivative loss using Huber loss (robust to outliers)
                    derivative_loss_elementwise = F.smooth_l1_loss(
                        pred_derivative,
                        gt_derivative,
                        reduction='none',
                        beta=0.05
                    )  # [B, T-1, 1, 3]
                    
                    # Mask to contact timesteps - both t and t+1 must be in contact
                    # (derivative spans from timestep t to t+1)
                    contact_mask_derivative = contact_mask_seq[:, 1:, :, :] * contact_mask_seq[:, :-1, :, :]
                    
                    derivative_loss = (
                        derivative_loss_elementwise * contact_mask_derivative
                    ).sum() / (contact_mask_derivative.sum() * velocity_label_seq.shape[-1] + 1e-8)
            else:
                # LAST TIMESTEP ONLY: Compute velocity loss only on final output (simpler, faster)
                velocity_loss_elementwise = supervised_velocity_loss(
                    posteriors, velocity_output, covariance_output,
                    velocity_label, epoch_loss_type, velocity_loss_fn,
                    model.base_model.natpn_evidence_source == 'input'
                )
                
                # Mask to contact samples only (last timestep)
                contact_mask = (contact_label == 1).float().unsqueeze(-1)  # [B, 1, 1]
                
                velocity_loss = (
                    velocity_loss_elementwise * contact_mask
                ).sum() / (contact_mask.sum() * velocity_label.shape[-1] + 1e-8)
                
                # No derivative loss in last-timestep-only mode
                derivative_loss = torch.tensor(0.0, device=velocity_output.device)
            
            loss = contact_weight * contact_loss + velocity_weight * velocity_loss + derivative_weight * derivative_loss
            
            # Add Elastic Net regularization (L1 + L2) if specified
            l1_lambda = float(config.get('l1_lambda', 0.0))
            l2_lambda = float(config.get('l2_lambda', 0.0))
            
            if l1_lambda > 0 or l2_lambda > 0:
                l1_norm = torch.tensor(0.0, device=velocity_output.device)
                l2_norm = torch.tensor(0.0, device=velocity_output.device)
                
                for p in model.parameters():
                    if l1_lambda > 0:
                        l1_norm = l1_norm + p.abs().sum()
                    if l2_lambda > 0:
                        l2_norm = l2_norm + (p ** 2).sum()
                
                loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
            
            # NOTE: Temporal consistency loss removed - requires sequential data, not shuffled batches
            # If needed, implement by grouping consecutive windows from same trajectory
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_sum += loss.item()
            contact_loss_sum += contact_loss.item()
            velocity_loss_sum += velocity_loss.item()
            if derivative_weight > 0 and use_dense_supervision:
                derivative_loss_sum += derivative_loss.item()

            if i % config['print_every'] == 0:
                derivative_str = f", derivative: {derivative_loss.item():.6f}" if (derivative_weight > 0 and use_dense_supervision) else ""
                if use_dense_supervision:
                    contact_count = int(contact_mask_seq.sum().item())
                else:
                    contact_count = int(contact_mask.sum().item())
                print("epoch %d / %d, iteration %d / %d, loss: %.8f (contact: %.6f, velocity masked: %.6f%s, contact samples: %d)" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), 
                     running_loss/config['print_every'],
                     contact_loss.item(), velocity_loss.item(), derivative_str, contact_count))
                running_loss = 0.0

        # calculate training and validation metrics
        model.eval()
        train_metrics = compute_accuracy(
            train_dataloader, model, contact_criterion=contact_criterion,
            velocity_loss_fn=velocity_loss_fn, velocity_loss_type=velocity_loss_type
        )
        val_metrics = compute_accuracy(
            val_dataloader, model, contact_criterion=contact_criterion,
            velocity_loss_fn=velocity_loss_fn, velocity_loss_type=velocity_loss_type
        )
        
        train_loss_avg = loss_sum / len(train_dataloader)
        contact_loss_avg = contact_loss_sum / len(train_dataloader)
        velocity_loss_avg = velocity_loss_sum / len(train_dataloader)
        derivative_loss_avg = derivative_loss_sum / len(train_dataloader) if (derivative_weight > 0 and use_dense_supervision) else 0.0

        # log down info in tensorboard
        writer.add_scalar('training/total_loss', train_loss_avg, epoch)
        writer.add_scalar('training/contact_loss', contact_loss_avg, epoch)
        writer.add_scalar('training/velocity_loss', velocity_loss_avg, epoch)
        if derivative_weight > 0 and use_dense_supervision:
            writer.add_scalar('training/derivative_loss', derivative_loss_avg, epoch)
        writer.add_scalar('training/contact_accuracy', train_metrics['contact_acc'], epoch)
        writer.add_scalar('training/velocity_mae', train_metrics['velocity_mae'], epoch)
        
        writer.add_scalar('validation/total_loss', val_metrics['contact_loss'] + val_metrics['velocity_loss'], epoch)
        writer.add_scalar('validation/contact_loss', val_metrics['contact_loss'], epoch)
        writer.add_scalar('validation/velocity_loss', val_metrics['velocity_loss'], epoch)
        writer.add_scalar('validation/contact_accuracy', val_metrics['contact_acc'], epoch)
        writer.add_scalar('validation/velocity_mae', val_metrics['velocity_mae'], epoch)
        if val_metrics['total_variance_calibration_ratio'] is not None:
            writer.add_scalar('validation/total_variance_calibration_ratio', val_metrics['total_variance_calibration_ratio'], epoch)

        # if we achieve best contact accuracy, save the model
        if val_metrics['contact_acc'] > best_contact_acc:
            best_contact_acc = val_metrics['contact_acc']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'contact_acc': train_metrics['contact_acc'],
                    'velocity_mae': train_metrics['velocity_mae'],
                    'val_loss': val_metrics['contact_loss'] + val_metrics['velocity_loss'],
                    'val_contact_acc': val_metrics['contact_acc'],
                    'val_velocity_mae': val_metrics['velocity_mae']}
        
            checkpoint_path = config['model_save_path']+'_best_val_contact_acc.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
        
        # if we achieve best velocity MAE, save the model.
        if val_metrics['velocity_mae'] < best_velocity_mae:
            best_velocity_mae = val_metrics['velocity_mae']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'contact_acc': train_metrics['contact_acc'],
                    'velocity_mae': train_metrics['velocity_mae'],
                    'val_loss': val_metrics['contact_loss'] + val_metrics['velocity_loss'],
                    'val_contact_acc': val_metrics['contact_acc'],
                    'val_velocity_mae': val_metrics['velocity_mae']}

            checkpoint_path = config['model_save_path']+'_best_val_velocity.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])

        # if we achieve best val loss, save the model
        val_total_loss = val_metrics['contact_loss'] + val_metrics['velocity_loss']
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'contact_acc': train_metrics['contact_acc'],
                    'velocity_mae': train_metrics['velocity_mae'],
                    'val_loss': val_total_loss,
                    'val_contact_acc': val_metrics['contact_acc'],
                    'val_velocity_mae': val_metrics['velocity_mae']}

            checkpoint_path = config['model_save_path']+'_best_val_loss.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
            

        train_left_mae = train_metrics['velocity_mae_components'][0]
        val_left_mae = val_metrics['velocity_mae_components'][0]
        print("Finished epoch %d / %d" % (epoch + 1, config['num_epoch']))
        print(
                "  Train - Contact Acc: %.4f, Body Velocity MAE: %.4f [vx/vy/vz: %.4f/%.4f/%.4f]" % (
                train_metrics['contact_acc'],
                train_metrics['velocity_mae'],
                *train_left_mae
            )
        )
        print(
                "  Val   - Contact Acc: %.4f, Body Velocity MAE: %.4f [vx/vy/vz: %.4f/%.4f/%.4f]" % (
                val_metrics['contact_acc'],
                val_metrics['velocity_mae'],
                *val_left_mae
            )
        )
        if val_metrics['total_variance_calibration_ratio'] is not None:
            print(f"  Val   - Total variance calibration ratio: {val_metrics['total_variance_calibration_ratio']:.4f} (target: 1.0)")
    
    optimize_natpn_flows(model, train_dataloader, config, natpn_finetune_epochs, 'NatPN fine-tune')

    # save final model
    state = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_avg,
            'contact_acc': train_metrics['contact_acc'],
            'velocity_mae': train_metrics['velocity_mae'],
            'val_loss': val_total_loss,
            'val_contact_acc': val_metrics['contact_acc'],
            'val_velocity_mae': val_metrics['velocity_mae']}

    checkpoint_path = config['model_save_path']+'_final_epoch.pt'
    torch.save(state, checkpoint_path)
    torch.save(state, config['model_save_path']+'_natpn_finetuned.pt')
    # Also save ONNX version for faster C++ deployment
    save_onnx_model(model, checkpoint_path, config['window_size'])

    writer.close()
    
    # Calculate total training time
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'='*60}\n")

    save_tcn_last_timestep_umap(
        val_dataloader,
        model,
        os.path.join(run_dir, "tcn_last_timestep_umap.png"),
        config.get('random_seed', 42),
    )

    # Generate comprehensive training summary with plots
    generate_training_summary(
        run_dir=run_dir,
        config=config,
        train_metrics={
            "train_loss": float(train_loss_avg),
            "train_contact_acc": float(train_metrics['contact_acc']),
            "train_velocity_mae": float(train_metrics['velocity_mae']),
            "val_loss": float(val_total_loss),
            "val_contact_acc": float(val_metrics['contact_acc']),
            "val_velocity_mae": float(val_metrics['velocity_mae']),
        },
        val_metrics=val_metrics,
        best_metrics={
            "best_val_loss": float(best_loss),
            "best_val_contact_acc": float(best_contact_acc),
            "best_val_velocity_mae": float(best_velocity_mae),
        },
        train_time_seconds=total_train_time,
        checkpoint_paths={
            "best_val_loss": checkpoint_path.replace('_final_epoch.pt', '_best_val_loss.pt'),
            "best_val_contact_acc": checkpoint_path.replace('_final_epoch.pt', '_best_val_contact_acc.pt'),
            "best_val_velocity": checkpoint_path.replace('_final_epoch.pt', '_best_val_velocity.pt'),
            "final_epoch": checkpoint_path,
        }
    )

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name), Loader=yaml.FullLoader)
    evidence_source = config.get('natpn_evidence_source', 'task')
    if evidence_source not in {'task', 'input'}:
        raise ValueError("natpn_evidence_source must be 'task' or 'input'.")
    if evidence_source == 'input' and config.get('use_dense_supervision', False):
        raise ValueError('Input-latent NatPN evidence currently supports only use_dense_supervision: false.')
    if evidence_source == 'input' and not config.get('input_natpn_checkpoint'):
        raise ValueError('Set input_natpn_checkpoint to the encoder_input_natpn.pt made by trainEncoder.py.')


    # Load num_features from data metadata (source of truth)
    metadata_path = config['data_folder'] + "all_data_metadata.npy"
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        num_features = metadata['num_features']
        print(f"Loaded num_features={num_features} from data metadata")
    else:
        num_features = config.get('num_features', 25)
        print(f"⚠️  Warning: metadata file not found, using config num_features={num_features}")


    
    # Load ALL data (not pre-split) - windowing happens first, then splitting
    all_dataset = contact_dataset(data_path=config['data_folder']+"all_data.npy",\
                                  label_path=config['data_folder']+"all_labels.npy",\
                                  window_size=config['window_size'],device=device)
    
    # Split by RUNS (not windows) to prevent data leakage from overlapping sliding windows
    # Get all unique run IDs
    all_run_ids = np.unique(all_dataset.window_to_run_id)
    num_runs = len(all_run_ids)
    
    # Validate sufficient runs for splitting
    if num_runs < 3:
        raise ValueError(f"Need at least 3 runs for train/val/test split, but only have {num_runs}. Collect more data or use fewer splits.")
    
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    
    train_num_runs = int(train_ratio * num_runs)
    val_num_runs = int(val_ratio * num_runs)
    test_num_runs = num_runs - train_num_runs - val_num_runs
    
    # Ensure at least 1 run per split
    if train_num_runs == 0:
        train_num_runs = 1
        val_num_runs = max(1, (num_runs - train_num_runs) // 2)
        test_num_runs = num_runs - train_num_runs - val_num_runs
        print(f"Warning: Adjusted split to ensure at least 1 run per split.")
    
    # Shuffle run IDs if specified (NOT windows - this prevents leakage)
    if config['shuffle']:
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(all_run_ids)
    
    # Split run IDs into train/val/test
    train_run_ids = set(all_run_ids[:train_num_runs])
    val_run_ids = set(all_run_ids[train_num_runs:train_num_runs + val_num_runs])
    test_run_ids = set(all_run_ids[train_num_runs + val_num_runs:])
    
    # Get all windows belonging to each split (all windows from a run go to same split)
    train_indices = all_dataset.get_windows_by_run_ids(train_run_ids)
    val_indices = all_dataset.get_windows_by_run_ids(val_run_ids)
    test_indices = all_dataset.get_windows_by_run_ids(test_run_ids)
    
    # Create Subset datasets for deterministic sampling
    from torch.utils.data import Subset
    train_dataset = Subset(all_dataset, train_indices)
    val_dataset = Subset(all_dataset, val_indices)
    
    # Compute global normalization statistics from TRAINING WINDOWS only
    # Extract actual data from training windows to compute unbiased statistics
    # NOTE: train_indices are window indices, not raw data indices
    # Compute stats on CPU (quantile is faster/more stable on CPU) then move to GPU
    print(f"Computing normalization statistics from {len(train_indices)} training windows...")
    train_windows_data = []
    for window_idx in train_indices:
        # Get the actual window data (shape: window_size x num_features), move to CPU for stats
        window = all_dataset[window_idx]['data'].cpu()
        train_windows_data.append(window)
    
    # Stack all training windows and flatten to get all training samples
    # Shape: (num_train_windows * window_size, num_features)
    train_data_flat = torch.cat(train_windows_data, dim=0)  # CPU tensor
    
    # Compute 1st and 99th percentiles for each feature to clip outliers
    percentile_1 = torch.quantile(train_data_flat, 0.01, dim=0, keepdim=True)  # (1, num_features) CPU
    percentile_99 = torch.quantile(train_data_flat, 0.99, dim=0, keepdim=True)  # (1, num_features) CPU
    
    # Clip training data to percentile bounds
    train_data_clipped = torch.clamp(train_data_flat, min=percentile_1, max=percentile_99)
    
    # Compute mean and std per feature from clipped data
    global_mean = train_data_clipped.mean(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, num_features)
    global_std = train_data_clipped.std(dim=0, keepdim=True).unsqueeze(0)    # Shape: (1, 1, num_features)
    
    # Handle features with zero std (constant values) to avoid division by zero
    global_std = torch.where(global_std == 0, torch.ones_like(global_std), global_std)
    
    # Move normalization stats to GPU so they stay on device inside the model
    global_mean = global_mean.to(device)
    global_std = global_std.to(device)
    print(f"Normalization stats computed and moved to {device}")

    
    # Create dataloaders
    # Train: shuffle each epoch for better training (but with reproducible seed from config)
    # Val: don't shuffle - we want consistent validation across epochs
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],\
                                  shuffle=config['shuffle'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'],\
                                shuffle=False)

    # init network with built-in normalization using global training statistics
    # num_features loaded from metadata at the start of main()
    # Select model architecture based on config
    model_arch = config.get('model_architecture', 'vanilla_cnn').lower()
    if model_arch != 'tcn':
        raise ValueError("NatPN velocity training is currently implemented only for model_architecture: 'tcn'.")
    
    if model_arch == 'attention_tcn':

        from contact_cnn import AttentionTCN
        base_model = AttentionTCN(
            window_size=config['window_size'],
            num_features=num_features,
            d_model=config.get('attention_d_model', 64),
            num_heads=config.get('attention_num_heads', 4),
            tcn_num_channels=config.get('tcn_num_channels', 64),
            tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5),
            tcn_dropout=config.get('tcn_dropout', 0.2)
        )

    elif model_arch == 'tcn':
        from contact_cnn import TCN
        base_model = TCN(
            window_size=config['window_size'],
            num_features=num_features,
            tcn_num_channels=config.get('tcn_num_channels', 64),
            tcn_kernel_size=config.get('tcn_kernel_size', 3),
            tcn_num_blocks=config.get('tcn_num_blocks', 5),
            tcn_dropout=config.get('tcn_dropout', 0.2),
            natpn_flow_layers=config.get('natpn_flow_layers', 8),
            natpn_certainty_budget=config.get('natpn_certainty_budget', 'normal'),
            natpn_evidence_source=evidence_source,
            input_natpn_checkpoint=config['input_natpn_checkpoint'],
            input_epistemic_scale=config.get('input_epistemic_scale', 1.0),
        )

    elif model_arch == 'vanilla_cnn':
        base_model = contact_cnn(
            window_size=config['window_size'],
            num_features=num_features
        )
    else:
        raise ValueError(f"Unknown model_architecture: {model_arch}. Options: 'attention_tcn', 'tcn', 'vanilla_cnn'")
    
    from contact_cnn import ContactCNNWithNormalization
    model = ContactCNNWithNormalization(base_model, global_mean=global_mean, global_std=global_std)
    model = model.to(device)

    train(model, train_dataloader, val_dataloader, config)

   
if __name__ == '__main__':
    main()
