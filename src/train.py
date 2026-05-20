import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import warnings

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

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
        contact_criterion: Optional contact loss criterion
        velocity_criterion: Optional velocity loss criterion
    
    Returns:
        dict with metrics: contact_acc, contact_loss, velocity_mae, velocity_loss
    """
    # num_correct = 0
    # num_data = 0
    # contact_loss_sum = 0
    velocity_loss_sum = 0
    velocity_mae_sum = 0
    num_data = 0
    
    # # Track prediction distribution to detect bias
    # num_pred_contact = 0
    # num_pred_no_contact = 0
    # num_gt_contact = 0
    # num_gt_no_contact = 0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # Shape: (batch, 1) - binary labels for left leg (last timestep)
            gt_velocity_seq = sample['velocity']  # Shape: (batch, window_size, 1) - full velocity sequence from dataset
            gt_velocity = gt_velocity_seq[:, -1, :]  # Extract last timestep: (batch, 1)

            # contact_output, velocity_output = model(input_data)  # contact: (batch, 1), velocity: (batch, 1)
            velocity_output = model(input_data)  # velocity: (batch, 1)
            # contact_prediction = (contact_output > 0).float()  # Binary predictions

            # # Compute losses if criteria provided
            # if contact_criterion is not None:
            #     contact_loss = contact_criterion(contact_output, gt_contact)
            #     contact_loss_sum += contact_loss.item()
            
            # Velocity loss masked to contact samples only
            contact_mask = (gt_contact == 1).float()  # (batch, 1)
            if velocity_criterion is not None:
                # Velocity loss on last timestep only, masked to contact samples only
                velocity_loss = velocity_criterion(velocity_output, gt_velocity)  # Mean reduction
                velocity_loss_masked = (velocity_loss * contact_mask.squeeze()).sum() / (contact_mask.sum() + 1e-8)
                velocity_loss_sum += velocity_loss_masked.item()
            
            # MAE for velocity (only on contact samples, last timestep)
            if contact_mask.sum() > 0:
                velocity_errors = torch.abs(velocity_output - gt_velocity)  # (batch, 1)
                velocity_mae = (velocity_errors * contact_mask).sum() / contact_mask.sum()
                velocity_mae_sum += velocity_mae.item()

            # # Contact accuracy
            # num_correct += (contact_prediction == gt_contact).sum().item()
            num_data += input_data.size(0)
            
            # # Track prediction distribution
            # num_pred_contact += (contact_prediction == 1).sum().item()
            # num_pred_no_contact += (contact_prediction == 0).sum().item()
            # num_gt_contact += (gt_contact == 1).sum().item()
            # num_gt_no_contact += (gt_contact == 0).sum().item()

    # print(f"\n  Prediction Distribution (left leg):")
    # print(f"    Model predicts contact (1):    {num_pred_contact}/{num_data} ({100*num_pred_contact/num_data:.1f}%)")
    # print(f"    Model predicts no-contact (0): {num_pred_no_contact}/{num_data} ({100*num_pred_no_contact/num_data:.1f}%)")
    # print(f"    Ground truth contact (1):      {num_gt_contact}/{num_data} ({100*num_gt_contact/num_data:.1f}%)")
    # print(f"    Ground truth no-contact (0):   {num_gt_no_contact}/{num_data} ({100*num_gt_no_contact/num_data:.1f}%)")

    metrics = {
        # 'contact_acc': num_correct / num_data,
        # 'contact_loss': contact_loss_sum / len(dataloader) if contact_criterion else 0,
        'velocity_mae': velocity_mae_sum / len(dataloader),
        'velocity_loss': velocity_loss_sum / len(dataloader) if velocity_criterion else 0
    }
    
    return metrics


def save_onnx_model(model, checkpoint_path, window_size):
    """
    Save ONNX version of the model for C++ deployment.
    The model has one output: 
        - velocity: (batch, 1) - velocity prediction at last timestep only
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
                output_names=['velocity_output'],  # velocity: (batch,1)
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'velocity_output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print(f"  ✓ ONNX model saved (velocity at last timestep): {onnx_path}")
        
    except Exception as e:
        print(f"  ⚠ Warning: Failed to save ONNX model: {e}")


def train(model, train_dataloader, val_dataloader, config):

    writer = SummaryWriter(config['log_writer_path'],comment=config['model_description'])
    writer.add_text("data_folder: ",config['data_folder'])
    writer.add_text("model_save_path: ",config['model_save_path'])
    writer.add_text("log_writer_path: ",config['log_writer_path'])
    writer.add_text("window_size: ",str(config['window_size']))
    writer.add_text("shuffle: ",str(config['shuffle']))
    writer.add_text("batch_size: ",str(config['batch_size']))
    writer.add_text("init_lr: ",str(config['init_lr']))
    writer.add_text("num_epoch: ",str(config['num_epoch']))
    writer.add_text("l1_lambda: ",str(config.get('l1_lambda', 0.0)))
    writer.add_text("l2_lambda: ",str(config.get('l2_lambda', 0.0)))
    writer.add_text("temporal_lambda: ",str(config.get('temporal_lambda', 0.0)))
    writer.add_text("velocity_weight: ",str(config.get('velocity_weight', 1.0)))
    writer.add_text("Huber_delta: ",str(config.get('Huber_delta', 0.5)))


    # Loss functions for velocity regression
    # # Contact: BCEWithLogitsLoss (with optional class weighting)
    # use_weighted_loss = config.get('use_weighted_loss', False)
    # if use_weighted_loss and 'pos_weight' in config:
    #     pos_weight = torch.tensor([config['pos_weight']]).to(next(model.parameters()).device)
    #     contact_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #     print(f"\n✓ Using WEIGHTED BCEWithLogitsLoss with pos_weight={config['pos_weight']:.2f}")
    #     print(f"  This penalizes false negatives (missing contact) {config['pos_weight']:.2f}x more than false positives\n")
    # else:
    #     contact_criterion = nn.BCEWithLogitsLoss()  # Standard unweighted loss
    #     print(f"\n✓ Using standard (unweighted) BCEWithLogitsLoss\n")
    
    # Velocity: Huber loss (robust to outliers, only computed on contact samples)
    huber_delta = float(config.get('Huber_delta', 0.5))
    velocity_criterion = nn.HuberLoss(delta=huber_delta, reduction='none')  # Element-wise for masking
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    
    # Get loss weighting parameters
    velocity_weight = float(config.get('velocity_weight', 1.0))  # Weight for velocity loss

    # best_acc = 0
    best_loss = 1000000000
    best_velocity_mae = 1000000000
    
    for epoch in range(config['num_epoch']):
        running_loss = 0.0  # For periodic printing
        loss_sum = 0.0  # For epoch average
        # contact_loss_sum = 0.0
        velocity_loss_sum = 0.0
        
        model.train()
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            contact_label = samples['label']  # Shape: (batch, 1) - left leg only (last timestep) - used for masking velocity loss
            velocity_label_seq = samples['velocity']  # Shape: (batch, window_size, 1) - full velocity sequence from dataset
            velocity_label = velocity_label_seq[:, -1, :]  # Extract last timestep: (batch, 1)
            
            # # DEBUG: Print ground truth labels once to verify data correctness
            # if not printed_labels and i == 0:
            #     print(f"\n{'='*80}")
            #     print(f"DEBUG: Ground Truth Contact Labels (first 100 samples)")
            #     print(f"{'='*80}")
            #     labels_to_print = contact_label.cpu().numpy()
            #     num_to_print = min(100, len(labels_to_print))
                
            #     left_labels = labels_to_print[:num_to_print, 0]
            #     right_labels = labels_to_print[:num_to_print, 1]
                
            #     print(f"\nLeft leg labels (first {num_to_print}):")
            #     print(left_labels)
            #     print(f"\nLeft leg stats: Mean={left_labels.mean():.3f}, "
            #           f"Contact={np.sum(left_labels==1)}, No-contact={np.sum(left_labels==0)}")
                
            #     print(f"\nRight leg labels (first {num_to_print}):")
            #     print(right_labels)
            #     print(f"\nRight leg stats: Mean={right_labels.mean():.3f}, "
            #           f"Contact={np.sum(right_labels==1)}, No-contact={np.sum(right_labels==0)}")
                
            #     print(f"\nOverall stats for this batch:")
            #     print(f"  Batch size: {len(labels_to_print)}")
            #     print(f"  Left leg contact ratio: {labels_to_print[:, 0].mean():.3f}")
            #     print(f"  Right leg contact ratio: {labels_to_print[:, 1].mean():.3f}")
            #     print(f"{'='*80}\n")
                
            #     printed_labels = True

            optimizer.zero_grad()
            # contact_output, velocity_output = model(input_data)  # contact: (batch, 1), velocity: (batch, 1)
            velocity_output = model(input_data)  # velocity: (batch, 1)

            # # Compute contact loss
            # contact_loss = contact_criterion(contact_output, contact_label)
            
            # Create contact mask (only compute velocity loss during contact)
            contact_mask = (contact_label == 1).float()  # (batch, 1)
            
            # Compute velocity loss (last timestep only, masked to contact only)
            velocity_loss_elementwise = velocity_criterion(velocity_output, velocity_label)  # Mean reduction gives scalar
            # Apply mask: only compute loss when ground truth contact == 1
            velocity_loss = (velocity_loss_elementwise * contact_mask.squeeze()).sum() / (contact_mask.sum() + 1e-8)
            
            # Combined loss (only velocity now, masked to contact samples)
            loss = velocity_weight * velocity_loss
            
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
            # contact_loss_sum += contact_loss.item()
            velocity_loss_sum += velocity_loss.item()

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, loss: %.8f (velocity masked: %.6f, contact samples: %d/%d)" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), 
                     running_loss/config['print_every'],
                     velocity_loss.item(), int(contact_mask.sum().item()), contact_mask.size(0)))
                running_loss = 0.0

        # calculate training and validation metrics
        model.eval()
        train_metrics = compute_accuracy(train_dataloader, model, velocity_criterion=velocity_criterion)
        val_metrics = compute_accuracy(val_dataloader, model, velocity_criterion=velocity_criterion)
        
        train_loss_avg = loss_sum / len(train_dataloader)
        # contact_loss_avg = contact_loss_sum / len(train_dataloader)
        velocity_loss_avg = velocity_loss_sum / len(train_dataloader)

        # log down info in tensorboard
        writer.add_scalar('training/total_loss', train_loss_avg, epoch)
        # writer.add_scalar('training/contact_loss', contact_loss_avg, epoch)
        writer.add_scalar('training/velocity_loss', velocity_loss_avg, epoch)
        # writer.add_scalar('training/contact_accuracy', train_metrics['contact_acc'], epoch)
        writer.add_scalar('training/velocity_mae', train_metrics['velocity_mae'], epoch)
        
        writer.add_scalar('validation/total_loss', val_metrics['velocity_loss'], epoch)
        # writer.add_scalar('validation/contact_loss', val_metrics['contact_loss'], epoch)
        writer.add_scalar('validation/velocity_loss', val_metrics['velocity_loss'], epoch)
        # writer.add_scalar('validation/contact_accuracy', val_metrics['contact_acc'], epoch)
        writer.add_scalar('validation/velocity_mae', val_metrics['velocity_mae'], epoch)

        # # if we achieve best val acc, save the model.
        # if val_metrics['contact_acc'] > best_acc:
        #     best_acc = val_metrics['contact_acc']
        #     
        #     state = {'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': train_loss_avg,
        #             'velocity_mae': train_metrics['velocity_mae'],
        #             'val_loss': val_metrics['velocity_loss'],
        #             'val_velocity_mae': val_metrics['velocity_mae']}
        # 
        #     checkpoint_path = config['model_save_path']+'_best_val_acc.pt'
        #     torch.save(state, checkpoint_path)
        #     # Also save ONNX version for faster C++ deployment
        #     save_onnx_model(model, checkpoint_path, config['window_size'])
        
        # if we achieve best velocity MAE, save the model.
        if val_metrics['velocity_mae'] < best_velocity_mae:
            best_velocity_mae = val_metrics['velocity_mae']
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'velocity_mae': train_metrics['velocity_mae'],
                    'val_loss': val_metrics['velocity_loss'],
                    'val_velocity_mae': val_metrics['velocity_mae']}

            checkpoint_path = config['model_save_path']+'_best_val_velocity.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])

        # if we achieve best val loss, save the model
        val_total_loss = val_metrics['velocity_loss']
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'velocity_mae': train_metrics['velocity_mae'],
                    'val_loss': val_total_loss,
                    'val_velocity_mae': val_metrics['velocity_mae']}

            checkpoint_path = config['model_save_path']+'_best_val_loss.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
            

        print("Finished epoch %d / %d" % (epoch, config['num_epoch']))
        print("  Train - Velocity MAE: %.4f" % train_metrics['velocity_mae']) 
        print("  Val   - Velocity MAE: %.4f" % val_metrics['velocity_mae'])
    
    # save final model     
    state = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_avg,
            'velocity_mae': train_metrics['velocity_mae'],
            'val_loss': val_total_loss,
            'val_velocity_mae': val_metrics['velocity_mae']}

    checkpoint_path = config['model_save_path']+'_final_epo.pt'
    torch.save(state, checkpoint_path)
    # Also save ONNX version for faster C++ deployment
    save_onnx_model(model, checkpoint_path, config['window_size'])

    writer.close()

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name), Loader=yaml.FullLoader)

    print("Using the following params: ")
    print("-------------path-------------")
    print("data_folder: ",config['data_folder'])
    print("model_save_path: ",config['model_save_path'])
    print("log_writer_path: ",config['log_writer_path'])
    # Load num_features from data metadata (source of truth)
    metadata_path = config['data_folder'] + "all_data_metadata.npy"
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        num_features = metadata['num_features']
        print(f"Loaded num_features={num_features} from data metadata")
    else:
        num_features = config.get('num_features', 25)
        print(f"⚠️  Warning: metadata file not found, using config num_features={num_features}")
    
    print("--------network params--------")
    print("num_features: ", num_features)
    print("window_size: ",config['window_size'])
    print("shuffle: ",config['shuffle'])
    print("batch_size: ",config['batch_size'])
    print("init_lr: ",config['init_lr'])
    print("num_epoch: ",config['num_epoch'])
    print("l1_lambda: ",config.get('l1_lambda', 0.0))
    print("l2_lambda: ",config.get('l2_lambda', 0.0))

    
    # Load ALL data (not pre-split) - windowing happens first, then splitting
    all_dataset = contact_dataset(data_path=config['data_folder']+"all_data.npy",\
                                  label_path=config['data_folder']+"all_labels.npy",\
                                  window_size=config['window_size'],device=device)
    
    # Split by RUNS (not windows) to prevent data leakage from overlapping sliding windows
    # Get all unique run IDs
    all_run_ids = np.unique(all_dataset.window_to_run_id)
    num_runs = len(all_run_ids)
    
    # DEBUG: Print run distribution to verify proper splitting
    print(f"\n{'='*60}")
    print(f"DEBUG: Run-based splitting verification")
    print(f"{'='*60}")
    for run_id in all_run_ids[:min(5, len(all_run_ids))]:  # Show first 5 runs
        num_windows_in_run = sum(1 for r in all_dataset.window_to_run_id if r == run_id)
        print(f"  Run {run_id}: {num_windows_in_run} windows")
    if len(all_run_ids) > 5:
        print(f"  ... and {len(all_run_ids) - 5} more runs")
    print(f"{'='*60}\n")
    
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
    
    # Validate non-empty splits
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError(f"Empty train or val set after run-based split. Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    print(f"\nDataset split BY RUNS (prevents data leakage):")
    print(f"  Total runs: {num_runs}")
    print(f"  Train runs: {len(train_run_ids)} -> {len(train_indices)} windows")
    print(f"  Val runs: {len(val_run_ids)} -> {len(val_indices)} windows")
    print(f"  Test runs: {len(test_run_ids)} -> {len(test_indices)} windows")
    print(f"  Total windows: {len(all_dataset)}")
    
    # DEBUG: Verify no overlap between splits
    print(f"\nDEBUG: Verifying split integrity...")
    train_val_overlap = train_run_ids.intersection(val_run_ids)
    train_test_overlap = train_run_ids.intersection(test_run_ids)
    val_test_overlap = val_run_ids.intersection(test_run_ids)
    
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        print(f"  ❌ ERROR: Run overlap detected!")
        print(f"     Train-Val overlap: {train_val_overlap}")
        print(f"     Train-Test overlap: {train_test_overlap}")
        print(f"     Val-Test overlap: {val_test_overlap}")
        raise ValueError("Data leakage detected: runs overlap between splits!")
    else:
        print(f"  ✓ No run overlap - splits are clean")
    
    # DEBUG: Show which runs went to which split (first few)
    print(f"  Train run IDs (first 5): {sorted(list(train_run_ids))[:5]}")
    print(f"  Val run IDs (first 5): {sorted(list(val_run_ids))[:5]}")
    print(f"  Test run IDs (first 5): {sorted(list(test_run_ids))[:5]}")
    
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
    train_data_engineered = torch.cat(train_windows_data, dim=0)
    
    # num_features already loaded from metadata above
    # Feature layout (LEFT LEG ONLY): auto-detected from data in csv2numpy.py
    # All features are RAW from csv2numpy.py - no pre-normalization
    
    # Compute 1st and 99th percentiles for each feature to clip outliers
    percentile_1 = torch.quantile(train_data_engineered, 0.01, dim=0, keepdim=True)  # (1, num_features)
    percentile_99 = torch.quantile(train_data_engineered, 0.99, dim=0, keepdim=True)  # (1, num_features)
    
    # Clip training data to percentile bounds
    train_data_clipped = torch.clamp(train_data_engineered, min=percentile_1, max=percentile_99)
    
    # Compute mean and std per feature from clipped data - LEFT LEG ONLY
    # Layout: q(6) + qd(6) + p(3) + v(3) + tau_est(6) + tau_mse(1)
    global_mean = train_data_clipped.mean(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, num_features)
    global_std = train_data_clipped.std(dim=0, keepdim=True).unsqueeze(0)    # Shape: (1, 1, num_features)
    
    # Handle features with zero std (constant values) to avoid division by zero
    global_std = torch.where(global_std == 0, torch.ones_like(global_std), global_std)
    
    print(f"\nGlobal normalization statistics computed from clipped training data:")
    print(f"  Total features: {num_features} (q + qd + p + v + tau_est + tau_mse) - LEFT LEG ONLY")
    print(f"  All features are RAW (not pre-normalized in csv2numpy.py)")
    print(f"  Clipped to [1st, 99th] percentiles per feature")
    print(f"  Mean shape: {global_mean.shape}")
    print(f"  Std shape: {global_std.shape}")
    print(f"  Mean range: [{global_mean.min().item():.4f}, {global_mean.max().item():.4f}]")
    print(f"  Std range: [{global_std.min().item():.4f}, {global_std.max().item():.4f}]")
    print(f"  Device: {global_mean.device}")
    
    # Create dataloaders
    # Train: shuffle each epoch for better training (but with reproducible seed from config)
    # Val: don't shuffle - we want consistent validation across epochs
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],\
                                  shuffle=config['shuffle'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'],\
                                shuffle=False)

    # init network with built-in normalization using global training statistics
    # num_features loaded from metadata at the start of main()
    base_model = contact_cnn(window_size=config['window_size'], num_features=num_features)
    from contact_cnn import ContactCNNWithNormalization
    model = ContactCNNWithNormalization(base_model, global_mean=global_mean, global_std=global_std)
    model = model.to(device)

    train(model, train_dataloader, val_dataloader, config)

   

if __name__ == '__main__':
    main()
