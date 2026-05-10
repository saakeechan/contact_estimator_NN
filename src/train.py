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

def compute_accuracy(dataloader, model):

    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(2)  # 2 legs: [left, right]
    velocity_mse_sum = 0.0  # Track velocity prediction error for both legs
    
    for sample in tqdm(dataloader):
        input_data = sample['data']
        gt_label = sample['label']  # Shape: (batch, 2) - binary labels for both legs
        gt_velocity = sample['velocity']  # Shape: (batch, 2) - velocity norms for both legs

        contact_output, velocity_output = model(input_data)  # Two outputs
        contact_prediction = (torch.sigmoid(contact_output) > 0.5).float()  # Binary predictions

        # Per-leg accuracy
        correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
        num_data += input_data.size(0)
        # Overall accuracy (averaged across both legs)
        num_correct += (contact_prediction == gt_label).sum().item()

        # Velocity MSE (for monitoring)
        velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).mean().item()

    # Total accuracy considers all predictions (both legs)
    total_predictions = num_data * 2  # 2 legs per sample
    return num_correct/total_predictions, correct_per_leg/num_data, velocity_mse_sum/len(dataloader)

def compute_accuracy_and_loss(dataloader, model, contact_criterion, velocity_criterion, velocity_weight=1.0):

    num_correct = 0
    num_data = 0
    contact_loss_sum = 0
    velocity_loss_sum = 0
    total_loss_sum = 0
    velocity_mse_sum = 0.0
    correct_per_leg = np.zeros(2)  # 2 legs: [left, right]
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']  # Shape: (batch, 2) - binary labels for both legs
            gt_velocity = sample['velocity']  # Shape: (batch, 2) - velocity norms for both legs

            contact_output, velocity_output = model(input_data)  # Two outputs
            contact_prediction = (torch.sigmoid(contact_output) > 0.5).float()  # Binary predictions

            contact_loss = contact_criterion(contact_output, gt_label)

            # Mask velocity loss by ground truth contact labels
            # gt_label shape: (batch, 2), velocity shape: (batch, 2)
            contact_mask = gt_label  # (batch, 2) - both legs
            velocity_loss_elementwise = velocity_criterion(velocity_output, gt_velocity)
            # Multiply by contact mask (only penalize velocities in contact)
            # Use max(1.0, sum) to avoid explosion when few contacts in batch
            velocity_loss = (velocity_loss_elementwise * contact_mask).sum() / torch.clamp(contact_mask.sum(), min=1.0)
            
            total_loss = contact_loss + velocity_weight * velocity_loss

            # Per-leg accuracy
            correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            # Overall accuracy (averaged across both legs)
            num_correct += (contact_prediction == gt_label).sum().item()

            contact_loss_sum += contact_loss.item()
            velocity_loss_sum += velocity_loss.item()
            total_loss_sum += total_loss.item()
            velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).mean().item()

    # Total accuracy considers all predictions (both legs)
    total_predictions = num_data * 2  # 2 legs per sample
    return (num_correct/total_predictions, correct_per_leg/num_data, 
            contact_loss_sum/len(dataloader), velocity_loss_sum/len(dataloader), 
            total_loss_sum/len(dataloader), velocity_mse_sum/len(dataloader))

# def decimal2binary(x):
#     # LEFT LEG ONLY: extract bit 1 (left foot) from decimal labels
#     # Decimal: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
#     # We only care about bit 1 (left foot)
#     mask = torch.tensor([2], device=x.device, dtype=x.dtype)  # Bit 1 mask
#     return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()



def save_onnx_model(model, checkpoint_path, window_size):
    """
    Save ONNX version of the model for C++ deployment.
    The model has two outputs: 
    - contact predictions (both legs: [left, right])
    - velocity predictions (both legs: [left, right])
    """
    try:
        import warnings
        
        # Create ONNX path (replace .pt with .onnx)
        onnx_path = checkpoint_path.replace('.pt', '.onnx')
        
        device = next(model.parameters()).device
        model.eval()
        
        # Create example input (RAW features - 57 from csv2numpy.py for both legs)
        example_input = torch.randn(1, window_size, 57).to(device)
        
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
                output_names=['contact_output', 'velocity_output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'contact_output': {0: 'batch_size'},
                    'velocity_output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print(f"  ✓ ONNX model saved (contact + velocity outputs, 2 legs each): {onnx_path}")
        
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


    # Multi-task learning: contact classification + velocity regression for both legs
    contact_criterion = nn.BCEWithLogitsLoss()  # For contact detection
    huber_delta = float(config.get('Huber_delta', 0.5))
    velocity_criterion = nn.HuberLoss(delta=huber_delta, reduction='none')  # Element-wise Huber loss for masking
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    
    # Get loss weighting parameters
    temporal_lambda = float(config.get('temporal_lambda', 0.0))
    velocity_weight = float(config.get('velocity_weight', 1.0))  # Weight for velocity loss

    best_acc = 0
    best_leg_acc = 0
    best_loss = 1000000000
    
    # DEBUG: Track if we've printed labels yet
    printed_labels = False
    
    for epoch in range(config['num_epoch']):
        running_loss = 0.0
        running_contact_loss = 0.0
        running_velocity_loss = 0.0
        loss_sum = 0.0
        contact_loss_sum = 0.0
        velocity_loss_sum = 0.0
        
        model.train()
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            contact_label = samples['label']  # Shape: (batch, 2) - [left, right]
            velocity_label = samples['velocity']  # Shape: (batch, 2) - [left, right]
            
            # DEBUG: Print ground truth labels once to verify data correctness
            if not printed_labels and i == 0:
                print(f"\n{'='*80}")
                print(f"DEBUG: Ground Truth Contact Labels (first 100 samples)")
                print(f"{'='*80}")
                labels_to_print = contact_label.cpu().numpy()
                num_to_print = min(100, len(labels_to_print))
                
                left_labels = labels_to_print[:num_to_print, 0]
                right_labels = labels_to_print[:num_to_print, 1]
                
                print(f"\nLeft leg labels (first {num_to_print}):")
                print(left_labels)
                print(f"\nLeft leg stats: Mean={left_labels.mean():.3f}, "
                      f"Contact={np.sum(left_labels==1)}, No-contact={np.sum(left_labels==0)}")
                
                print(f"\nRight leg labels (first {num_to_print}):")
                print(right_labels)
                print(f"\nRight leg stats: Mean={right_labels.mean():.3f}, "
                      f"Contact={np.sum(right_labels==1)}, No-contact={np.sum(right_labels==0)}")
                
                print(f"\nOverall stats for this batch:")
                print(f"  Batch size: {len(labels_to_print)}")
                print(f"  Left leg contact ratio: {labels_to_print[:, 0].mean():.3f}")
                print(f"  Right leg contact ratio: {labels_to_print[:, 1].mean():.3f}")
                print(f"{'='*80}\n")
                
                printed_labels = True

            optimizer.zero_grad()
            contact_output, velocity_output = model(input_data)  # Two outputs: (batch, 2) each

            # Compute losses for both tasks
            contact_loss = contact_criterion(contact_output, contact_label)
            
            # Mask velocity loss by ground truth contact labels
            # contact_label shape: (batch, 2), velocity shape: (batch, 2)
            contact_mask = contact_label  # (batch, 2) - both legs
            velocity_loss_elementwise = velocity_criterion(velocity_output, velocity_label)
            # Multiply by contact mask (only penalize velocities in contact)
            # Use max(1.0, sum) to avoid explosion when few contacts in batch
            velocity_loss = (velocity_loss_elementwise * contact_mask).sum() / torch.clamp(contact_mask.sum(), min=1.0)
            
            # Combined loss with weighting
            loss = contact_loss + velocity_weight * velocity_loss
            # Add Elastic Net regularization (L1 + L2) if specified
            l1_lambda = float(config.get('l1_lambda', 0.0))
            l2_lambda = float(config.get('l2_lambda', 0.0))
            
            if l1_lambda > 0 or l2_lambda > 0:
                l1_norm = torch.tensor(0.0, device=contact_output.device)
                l2_norm = torch.tensor(0.0, device=contact_output.device)
                
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
            running_contact_loss += contact_loss.item()
            running_velocity_loss += velocity_loss.item()
            loss_sum += loss.item()
            contact_loss_sum += contact_loss.item()
            velocity_loss_sum += velocity_loss.item()

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, total loss: %.8f, contact loss: %.8f, velocity loss: %.8f" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), 
                     running_loss/config['print_every'],
                     running_contact_loss/config['print_every'],
                     running_velocity_loss/config['print_every']))
                running_loss = 0.0
                running_contact_loss = 0.0
                running_velocity_loss = 0.0

        # calculate training and validation metrics
        model.eval()
        train_acc, train_acc_per_leg, train_velocity_mse = compute_accuracy(train_dataloader, model)
        train_loss_avg = loss_sum/len(train_dataloader)
        train_contact_loss_avg = contact_loss_sum/len(train_dataloader)
        train_velocity_loss_avg = velocity_loss_sum/len(train_dataloader)

        (val_acc, val_acc_per_leg, val_contact_loss_avg, val_velocity_loss_avg, 
         val_loss_avg, val_velocity_mse) = compute_accuracy_and_loss(
            val_dataloader, model, contact_criterion, velocity_criterion, velocity_weight)

        train_acc_per_leg_avg = train_acc_per_leg.mean()  # Average of both legs
        val_acc_per_leg_avg = val_acc_per_leg.mean()  # Average of both legs

        # log down info in tensorboard
        writer.add_scalar('training loss', train_loss_avg, epoch)
        writer.add_scalar('training contact loss', train_contact_loss_avg, epoch)
        writer.add_scalar('training velocity loss', train_velocity_loss_avg, epoch)
        writer.add_scalar('training velocity MSE', train_velocity_mse, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('training acc left leg', train_acc_per_leg[0], epoch)
        writer.add_scalar('training acc right leg', train_acc_per_leg[1], epoch)
        writer.add_scalar('training acc leg avg', train_acc_per_leg_avg, epoch)
        
        writer.add_scalar('validation loss', val_loss_avg, epoch)
        writer.add_scalar('validation contact loss', val_contact_loss_avg, epoch)
        writer.add_scalar('validation velocity loss', val_velocity_loss_avg, epoch)
        writer.add_scalar('validation velocity MSE', val_velocity_mse, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)
        writer.add_scalar('validation acc left leg', val_acc_per_leg[0], epoch)
        writer.add_scalar('validation acc right leg', val_acc_per_leg[1], epoch)
        writer.add_scalar('validation acc leg avg', val_acc_per_leg_avg, epoch)

        # if we achieve best val acc, save the model.
        if val_acc > best_acc:
            best_acc = val_acc
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            checkpoint_path = config['model_save_path']+'_best_val_acc.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
        
        # if we achieve best val leg acc, save the model.
        if val_acc_per_leg_avg > best_leg_acc:
            best_leg_acc = val_acc_per_leg_avg
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            checkpoint_path = config['model_save_path']+'_best_val_leg_acc.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])

        # if we achieve best val loss, save the model
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            checkpoint_path = config['model_save_path']+'_best_val_loss.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
            

        print("Finished epoch %d / %d, training acc: %.4f, validation acc: %.4f" %\
            (epoch, config['num_epoch'], train_acc, val_acc)) 
        print("train left leg acc: %.4f, val left leg acc: %.4f" %\
            (train_acc_per_leg[0], val_acc_per_leg[0]))
        print("train right leg acc: %.4f, val right leg acc: %.4f" %\
            (train_acc_per_leg[1], val_acc_per_leg[1]))
        print("train velocity MSE: %.6f, val velocity MSE: %.6f" %\
            (train_velocity_mse, val_velocity_mse))
    
    # save model     
    state = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_avg,
            'acc': train_acc,
            'val_loss': val_loss_avg,
            'val_acc': val_acc}

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
    print("--------network params--------")
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
    
    # CRITICAL WARNING: Check if all data is treated as one run
    if num_runs == 1:
        print(f"\n{'='*60}")
        print(f"⚠️  CRITICAL WARNING: Only 1 run detected!")
        print(f"{'='*60}")
        print(f"All windows belong to the same run. This means:")
        print(f"  - Overlapping windows are NOT separated by runs")
        print(f"  - Data leakage is STILL PRESENT")
        print(f"  - High test accuracy is ARTIFICIALLY INFLATED")
        print(f"\nSOLUTION: Your CSV files likely don't have time resets.")
        print(f"  1. Check csv2numpy.py run boundary detection")
        print(f"  2. Ensure CSVs have clear run separations")
        print(f"  3. Or manually split data into separate CSV files per run")
        print(f"{'='*60}\n")
    
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
    # Shape: (num_train_windows * window_size, 57)
    train_data_engineered = torch.cat(train_windows_data, dim=0)
    
    # Feature layout (57 features): acc(3) + omega(3) + q(12) + qd(12) + p(6) + v(6) + tau_est(12) + tau_mse(2) + cmd_vel(1)
    # All features are RAW from csv2numpy.py - no pre-normalization
    
    # Compute 1st and 99th percentiles for each feature to clip outliers
    percentile_1 = torch.quantile(train_data_engineered, 0.01, dim=0, keepdim=True)  # (1, 57)
    percentile_99 = torch.quantile(train_data_engineered, 0.99, dim=0, keepdim=True)  # (1, 57)
    
    # Clip training data to percentile bounds
    train_data_clipped = torch.clamp(train_data_engineered, min=percentile_1, max=percentile_99)
    
    # Compute mean and std per feature from clipped data (57 features total)
    # Layout: acc(0-2) + omega(3-5) + q(6-17) + qd(18-29) + p(30-35) + v(36-41) + tau_est(42-53) + tau_mse(54-55) + cmd_vel(56)
    global_mean = train_data_clipped.mean(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, 57)
    global_std = train_data_clipped.std(dim=0, keepdim=True).unsqueeze(0)    # Shape: (1, 1, 57)
    
    # Handle features with zero std (constant values) to avoid division by zero
    global_std = torch.where(global_std == 0, torch.ones_like(global_std), global_std)
    
    print(f"\nGlobal normalization statistics computed from clipped training data:")
    print(f"  Total features: 57 (acc + omega + q + qd + p + v + tau_est + tau_mse + cmd_vel)")
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
    base_model = contact_cnn(window_size=config['window_size'])
    from contact_cnn import ContactCNNWithNormalization
    model = ContactCNNWithNormalization(base_model, global_mean=global_mean, global_std=global_std)
    model = model.to(device)

    train(model, train_dataloader, val_dataloader, config)

   

if __name__ == '__main__':
    main()
