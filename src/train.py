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

from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*LeafSpec.*"
)


def compute_accuracy(dataloader, model, contact_criterion=None, velocity_criterion=None):
    """
    Compute accuracy and losses for left leg contact and velocity at last timestep.
    
    Args:
        dataloader: DataLoader to evaluate
        model: The neural network model
        contact_criterion: Optional contact loss criterion (BCEWithLogitsLoss)
        velocity_criterion: Optional velocity loss criterion
    
    Returns:
        dict with metrics: contact_acc, contact_loss, velocity_mae, velocity_loss
    """
    num_correct = 0
    num_data = 0
    contact_loss_sum = 0
    velocity_loss_sum = 0
    velocity_mae_sum = 0
    
    # Track prediction distribution to detect bias
    num_pred_contact = 0
    num_pred_no_contact = 0
    num_gt_contact = 0
    num_gt_no_contact = 0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # Shape: (batch, 1) - binary labels for left leg (last timestep)
            gt_velocity_seq = sample['velocity']  # Shape: (batch, window_size, 1) - full velocity sequence from dataset
            gt_velocity = gt_velocity_seq[:, -1, :]  # Extract last timestep: (batch, 1)

            velocity_seq, velocity_output, contact_output = model(input_data)  # contact: (batch, 1), velocity: (batch, 1)
            contact_prediction = (contact_output > 0).float()  # Binary predictions

            # Compute contact loss if criterion provided
            if contact_criterion is not None:
                contact_loss = contact_criterion(contact_output, gt_contact)
                contact_loss_sum += contact_loss.item()
            
            # Velocity loss masked to contact samples only
            contact_mask = (gt_contact == 1).float()  # [B, 1]
            
            if velocity_criterion is not None:
                # Velocity loss on last timestep only, masked to contact samples only
                velocity_loss_each = velocity_criterion(
                    velocity_output, gt_velocity
                )  # [B, 1], requires reduction="none"
                
                velocity_loss_masked = (
                    velocity_loss_each * contact_mask
                ).sum() / (contact_mask.sum() + 1e-8)
                
                velocity_loss_sum += velocity_loss_masked.item()
            
            # MAE for velocity (only on contact samples, last timestep)
            if contact_mask.sum() > 0:
                velocity_errors = torch.abs(velocity_output - gt_velocity)  # [B, 1]
                
                velocity_mae = (
                    velocity_errors * contact_mask
                ).sum() / (contact_mask.sum() + 1e-8)
                
                velocity_mae_sum += velocity_mae.item()

            # Contact accuracy
            num_correct += (contact_prediction == gt_contact).sum().item()
            num_data += input_data.size(0)
            
            # Track prediction distribution
            num_pred_contact += contact_prediction.sum().item()
            num_pred_no_contact += (1 - contact_prediction).sum().item()
            num_gt_contact += gt_contact.sum().item()
            num_gt_no_contact += (1 - gt_contact).sum().item()

    metrics = {
        'contact_acc': num_correct / num_data,
        'contact_loss': contact_loss_sum / len(dataloader) if contact_criterion else 0,
        'velocity_mae': velocity_mae_sum / len(dataloader),
        'velocity_loss': velocity_loss_sum / len(dataloader) if velocity_criterion else 0,
        'num_pred_contact': num_pred_contact,
        'num_gt_contact': num_gt_contact
    }
    
    return metrics


def save_onnx_model(model, checkpoint_path, window_size):
    """
    Save ONNX version of the model for C++ deployment.
    The model has three outputs: 
        - velocity_seq: (batch, 1, window_size) - velocity predictions for all timesteps (for training)
        - velocity_output: (batch, 1) - velocity prediction at last timestep only (for inference)
        - contact_output: (batch, 1) - contact logits at last timestep
    """
    try:
        import warnings
        
        # Create ONNX path (replace .pt with .onnx)
        onnx_path = checkpoint_path.replace('.pt', '.onnx')
        
        device = next(model.parameters()).device
        model.eval()
        
        # Create example input (RAW features from csv2numpy.py for LEFT leg only)
        # num_features is read from model's base_model attribute
        num_features = model.base_model.num_features
        example_input = torch.randn(1, window_size, num_features).to(device)
        
        # Suppress warnings and use standard export
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                input_names=['input'],
                output_names=['velocity_seq', 'velocity_output'],  # velocity_seq: (batch,1,T), velocity_output: (batch,1)
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'velocity_seq': {0: 'batch_size', 2: 'window_size'},
                    'velocity_output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print(f"  ✓ ONNX model saved (velocity sequence + last timestep): {onnx_path}")
        
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
    
    # Velocity: HuberLoss (robust to outliers)
    huber_delta = float(config.get('Huber_delta', 0.5))
    velocity_criterion = nn.HuberLoss(delta=huber_delta, reduction='none')  # Element-wise for masking
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


    # best_acc = 0
    best_loss = 1000000000
    best_velocity_mae = 1000000000
    best_contact_acc = 0
    
    # Pre-compute temporal weights ONCE (before training loop) to avoid recomputation every batch
    device = next(model.parameters()).device
    if use_dense_supervision and temporal_weight_power > 0:
        T = config['window_size']
        timesteps = torch.arange(1, T + 1, dtype=torch.float32, device=device)
        temporal_weights = ((timesteps / T) ** temporal_weight_power).view(1, T, 1)  # [1, T, 1]
        
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
        running_loss = 0.0  # For periodic printing
        loss_sum = 0.0  # For epoch average
        contact_loss_sum = 0.0
        velocity_loss_sum = 0.0
        derivative_loss_sum = 0.0
        
        model.train()
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            contact_label = samples['label']  # Shape: (batch, 1) - left leg only (last timestep) - for reporting
            contact_label_seq = samples['label_seq']  # Shape: (batch, window_size, 1) - full contact sequence for masking
            velocity_label_seq = samples['velocity']  # Shape: (batch, window_size, 1) - full velocity sequence from dataset
            velocity_label = velocity_label_seq[:, -1, :]  # Extract last timestep: (batch, 1)
        

            optimizer.zero_grad()
            
            velocity_seq, velocity_output, contact_output = model(input_data)  # velocity_seq: (batch, 1, window_size), velocity_output: (batch, 1), contact: (batch, 1)
            
            # Contact loss (binary classification at last timestep)
            contact_loss = contact_criterion(contact_output, contact_label)
            
            if use_dense_supervision:
                # DENSE SUPERVISION: Compute velocity loss on FULL SEQUENCE masked to contact only
                velocity_seq_permuted = velocity_seq.permute(0, 2, 1)  # [B, T, 1]
                
                velocity_loss_elementwise = velocity_criterion(
                    velocity_seq_permuted,
                    velocity_label_seq
                )  # [B, T, 1]
                
                # Use full contact sequence for masking (dense supervision only at contact timesteps)
                contact_mask_seq = (contact_label_seq == 1).float()  # [B, T, 1]
                
                # Apply temporal weighting (pre-computed before training loop)
                velocity_loss = (
                    velocity_loss_elementwise * contact_mask_seq * temporal_weights
                ).sum() / ((contact_mask_seq * temporal_weights).sum() + 1e-8)
                
                # Derivative matching loss: match temporal dynamics (slopes/changes)
                # Only available in dense mode (requires full sequence)
                derivative_loss = torch.tensor(0.0, device=velocity_output.device)
                if derivative_weight > 0:
                    # Compute derivatives: Δy_t = y_t - y_{t-1}
                    pred_derivative = velocity_seq_permuted[:, 1:, :] - velocity_seq_permuted[:, :-1, :]  # [B, T-1, 1]
                    gt_derivative = velocity_label_seq[:, 1:, :] - velocity_label_seq[:, :-1, :]  # [B, T-1, 1]
                    
                    # Compute derivative loss using Huber loss (robust to outliers)
                    derivative_loss_elementwise = F.smooth_l1_loss(
                        pred_derivative,
                        gt_derivative,
                        reduction='none',
                        beta=0.05
                    )  # [B, T-1, 1]
                    
                    # Mask to contact timesteps - both t and t+1 must be in contact
                    # (derivative spans from timestep t to t+1)
                    contact_mask_derivative = contact_mask_seq[:, 1:, :] * contact_mask_seq[:, :-1, :]  # [B, T-1, 1]
                    
                    derivative_loss = (
                        derivative_loss_elementwise * contact_mask_derivative
                    ).sum() / (contact_mask_derivative.sum() + 1e-8)
            else:
                # LAST TIMESTEP ONLY: Compute velocity loss only on final output (simpler, faster)
                velocity_loss_elementwise = velocity_criterion(
                    velocity_output,
                    velocity_label
                )  # [B, 1]
                
                # Mask to contact samples only (last timestep)
                contact_mask = (contact_label == 1).float()  # [B, 1]
                
                velocity_loss = (
                    velocity_loss_elementwise * contact_mask
                ).sum() / (contact_mask.sum() + 1e-8)
                
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
        train_metrics = compute_accuracy(train_dataloader, model, contact_criterion=contact_criterion, velocity_criterion=velocity_criterion)
        val_metrics = compute_accuracy(val_dataloader, model, contact_criterion=contact_criterion, velocity_criterion=velocity_criterion)
        
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
            

        print("Finished epoch %d / %d" % (epoch + 1, config['num_epoch']))
        print("  Train - Contact Acc: %.4f, Velocity MAE: %.4f" % (train_metrics['contact_acc'], train_metrics['velocity_mae'])) 
        print("  Val   - Contact Acc: %.4f, Velocity MAE: %.4f" % (val_metrics['contact_acc'], val_metrics['velocity_mae']))
    
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


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name), Loader=yaml.FullLoader)


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
    train_windows_data = []
    for window_idx in train_indices:
        # Get the actual window data (shape: window_size x 57)
        window = all_dataset[window_idx]['data']  # Use __getitem__ to get proper windowed data
        train_windows_data.append(window)  # Keep on same device as dataset
    
    # Stack all training windows and flatten to get all training samples
    # Shape: (num_train_windows * window_size, num_features)
    train_data_flat = torch.cat(train_windows_data, dim=0)
    
    # Compute 1st and 99th percentiles for each feature to clip outliers
    percentile_1 = torch.quantile(train_data_flat, 0.01, dim=0, keepdim=True)  # (1, num_features)
    percentile_99 = torch.quantile(train_data_flat, 0.99, dim=0, keepdim=True)  # (1, num_features)
    
    # Clip training data to percentile bounds
    train_data_clipped = torch.clamp(train_data_flat, min=percentile_1, max=percentile_99)
    
    # Compute mean and std per feature from clipped data - LEFT LEG ONLY
    # Layout: q(6) + qd(6) + p(3) + v(3) + tau_est(6) + tau_mse(1)
    global_mean = train_data_clipped.mean(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, num_features)
    global_std = train_data_clipped.std(dim=0, keepdim=True).unsqueeze(0)    # Shape: (1, 1, num_features)
    
    # Handle features with zero std (constant values) to avoid division by zero
    global_std = torch.where(global_std == 0, torch.ones_like(global_std), global_std)

    
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
            tcn_dropout=config.get('tcn_dropout', 0.2)
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