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
    correct_per_leg = np.zeros(1)  # 1 leg: LEFT only
    # velocity_mse_sum = 0.0  # Track velocity prediction error for both legs
    
    # Track prediction distribution to detect bias
    num_pred_contact = 0  # How many times model predicts 1 (contact)
    num_pred_no_contact = 0  # How many times model predicts 0 (no-contact)
    num_gt_contact = 0  # How many ground truth 1's
    num_gt_no_contact = 0  # How many ground truth 0's
    
    for sample in tqdm(dataloader):
        input_data = sample['data']
        gt_label = sample['label']  # Shape: (batch, 1) - binary labels for LEFT leg only
        # gt_velocity = sample['velocity']  # Shape: (batch, 1) - velocity norm for LEFT leg only

        contact_output = model(input_data)  # Only contact output
        contact_prediction = (contact_output > 0).float()  # Binary predictions (more efficient than sigmoid)

        # Per-leg accuracy
        correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
        num_data += input_data.size(0)
        # Overall accuracy (LEFT leg)
        num_correct += (contact_prediction == gt_label).sum().item()
        
        # Track prediction distribution
        num_pred_contact += (contact_prediction == 1).sum().item()
        num_pred_no_contact += (contact_prediction == 0).sum().item()
        num_gt_contact += (gt_label == 1).sum().item()
        num_gt_no_contact += (gt_label == 0).sum().item()

        # # Velocity MSE (for monitoring)
        # velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).mean().item()

    # Total accuracy considers all predictions (LEFT leg only)
    total_predictions = num_data * 1  # 1 leg per sample
    
    # Print prediction distribution to detect bias
    print(f"\n  Prediction Distribution:")
    print(f"    Model predicts contact (1):    {num_pred_contact}/{total_predictions} ({100*num_pred_contact/total_predictions:.1f}%)")
    print(f"    Model predicts no-contact (0): {num_pred_no_contact}/{total_predictions} ({100*num_pred_no_contact/total_predictions:.1f}%)")
    print(f"    Ground truth contact (1):      {num_gt_contact}/{total_predictions} ({100*num_gt_contact/total_predictions:.1f}%)")
    print(f"    Ground truth no-contact (0):   {num_gt_no_contact}/{total_predictions} ({100*num_gt_no_contact/total_predictions:.1f}%)")
    
    return num_correct/total_predictions, correct_per_leg/num_data  # , velocity_mse_sum/len(dataloader)

def compute_accuracy_and_loss(dataloader, model, contact_criterion):  # , velocity_criterion, velocity_weight=1.0):

    num_correct = 0
    num_data = 0
    contact_loss_sum = 0
    # velocity_loss_sum = 0
    total_loss_sum = 0
    # velocity_mse_sum = 0.0
    correct_per_leg = np.zeros(1)  # 1 leg: LEFT only
    
    # Track prediction distribution
    num_pred_contact = 0
    num_pred_no_contact = 0
    num_gt_contact = 0
    num_gt_no_contact = 0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']  # Shape: (batch, 1) - binary labels for LEFT leg only
            # gt_velocity = sample['velocity']  # Shape: (batch, 1) - velocity norm for LEFT leg only

            contact_output = model(input_data)  # Only contact output
            contact_prediction = (contact_output > 0).float()  # Binary predictions (more efficient)

            contact_loss = contact_criterion(contact_output, gt_label)

            # # Mask velocity loss by ground truth contact labels
            # # gt_label shape: (batch, 1), velocity shape: (batch, 1)
            # contact_mask = gt_label  # (batch, 1) - LEFT leg only
            # velocity_loss_elementwise = velocity_criterion(velocity_output, gt_velocity)
            # # Multiply by contact mask (only penalize velocities in contact)
            # # Use max(1.0, sum) to avoid explosion when few contacts in batch
            # velocity_loss = (velocity_loss_elementwise * contact_mask).sum() / torch.clamp(contact_mask.sum(), min=1.0)
            
            total_loss = contact_loss  # + velocity_weight * velocity_loss

            # Per-leg accuracy
            correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            # Overall accuracy (LEFT leg)
            num_correct += (contact_prediction == gt_label).sum().item()
            
            # Track distribution
            num_pred_contact += (contact_prediction == 1).sum().item()
            num_pred_no_contact += (contact_prediction == 0).sum().item()
            num_gt_contact += (gt_label == 1).sum().item()
            num_gt_no_contact += (gt_label == 0).sum().item()

            contact_loss_sum += contact_loss.item()
            # velocity_loss_sum += velocity_loss.item()
            total_loss_sum += total_loss.item()
            # velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).mean().item()

    # Total accuracy considers all predictions (LEFT leg only)
    total_predictions = num_data * 1  # 1 leg per sample
    
    # Print prediction distribution
    print(f"\n  Prediction Distribution:")
    print(f"    Model predicts contact (1):    {num_pred_contact}/{total_predictions} ({100*num_pred_contact/total_predictions:.1f}%)")
    print(f"    Model predicts no-contact (0): {num_pred_no_contact}/{total_predictions} ({100*num_pred_no_contact/total_predictions:.1f}%)")
    print(f"    Ground truth contact (1):      {num_gt_contact}/{total_predictions} ({100*num_gt_contact/total_predictions:.1f}%)")
    print(f"    Ground truth no-contact (0):   {num_gt_no_contact}/{total_predictions} ({100*num_gt_no_contact/total_predictions:.1f}%)")
    
    return (num_correct/total_predictions, correct_per_leg/num_data, 
            contact_loss_sum/len(dataloader))  # , velocity_loss_sum/len(dataloader), total_loss_sum/len(dataloader), velocity_mse_sum/len(dataloader))

# def decimal2binary(x):
#     # LEFT LEG ONLY: extract bit 1 (left foot) from decimal labels
#     # Decimal: 0=[0,0], 1=[0,1], 2=[1,0], 3=[1,1]
#     # We only care about bit 1 (left foot)
#     mask = torch.tensor([2], device=x.device, dtype=x.dtype)  # Bit 1 mask
#     return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()



def save_onnx_model(model, checkpoint_path, window_size):
    """
    Save ONNX version of the model for C++ deployment.
    The model has one output: contact prediction (LEFT leg only)
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
                output_names=['contact_output'],  # Only contact output - LEFT leg
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'contact_output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print(f"  ✓ ONNX model saved (contact output, LEFT leg only): {onnx_path}")
        
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
    # writer.add_text("velocity_weight: ",str(config.get('velocity_weight', 1.0)))
    # writer.add_text("Huber_delta: ",str(config.get('Huber_delta', 0.5)))


    # Contact classification only (velocity heads removed)
    # Check if using weighted loss for class imbalance
    use_weighted_loss = config.get('use_weighted_loss', False)
    if use_weighted_loss and 'pos_weight' in config:
        pos_weight = torch.tensor([config['pos_weight']]).to(next(model.parameters()).device)
        contact_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"\n✓ Using WEIGHTED BCEWithLogitsLoss with pos_weight={config['pos_weight']:.2f}")
        print(f"  This penalizes false negatives (missing contact) {config['pos_weight']:.2f}x more than false positives\n")
    else:
        contact_criterion = nn.BCEWithLogitsLoss()  # Standard unweighted loss
        print(f"\n✓ Using standard (unweighted) BCEWithLogitsLoss\n")
    
    # huber_delta = float(config.get('Huber_delta', 0.5))
    # velocity_criterion = nn.HuberLoss(delta=huber_delta, reduction='none')  # Element-wise Huber loss for masking
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    
    # Get loss weighting parameters
    # temporal_lambda = float(config.get('temporal_lambda', 0.0))
    # velocity_weight = float(config.get('velocity_weight', 1.0))  # Weight for velocity loss

    best_acc = 0
    best_leg_acc = 0
    best_loss = 1000000000
    
    # DEBUG: Track if we've printed labels yet
    printed_labels = False
    
    for epoch in range(config['num_epoch']):
        running_loss = 0.0  # For periodic printing
        loss_sum = 0.0  # For epoch average
        
        model.train()
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            contact_label = samples['label']  # Shape: (batch, 1) - LEFT leg only
            # velocity_label = samples['velocity']  # Shape: (batch, 1) - LEFT leg only
            
            # DEBUG: Print ground truth labels once to verify data correctness
            if not printed_labels and i == 0:
                print(f"\n{'='*80}")
                print(f"DEBUG: Ground Truth Contact Labels (first 100 samples)")
                print(f"{'='*80}")
                labels_to_print = contact_label.cpu().numpy()
                num_to_print = min(100, len(labels_to_print))
                
                left_labels = labels_to_print[:num_to_print, 0]
                # right_labels = labels_to_print[:num_to_print, 1]  # COMMENTED OUT - only LEFT leg
                
                print(f"\nLeft leg labels (first {num_to_print}):")
                print(left_labels)
                print(f"\nLeft leg stats: Mean={left_labels.mean():.3f}, "
                      f"Contact={np.sum(left_labels==1)}, No-contact={np.sum(left_labels==0)}")
                
                # print(f"\nRight leg labels (first {num_to_print}):")  # COMMENTED OUT
                # print(right_labels)  # COMMENTED OUT
                # print(f"\nRight leg stats: Mean={right_labels.mean():.3f}, "  # COMMENTED OUT
                #       f"Contact={np.sum(right_labels==1)}, No-contact={np.sum(right_labels==0)}")  # COMMENTED OUT
                
                print(f"\nOverall stats for this batch:")
                print(f"  Batch size: {len(labels_to_print)}")
                print(f"  Left leg contact ratio: {labels_to_print[:, 0].mean():.3f}")
                # print(f"  Right leg contact ratio: {labels_to_print[:, 1].mean():.3f}")  # COMMENTED OUT
                print(f"{'='*80}\n")
                
                printed_labels = True

            optimizer.zero_grad()
            contact_output = model(input_data)  # Only contact output: (batch, 1)

            # Compute contact loss (this is the only loss now)
            loss = contact_criterion(contact_output, contact_label)
            
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
            loss_sum += loss.item()

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, loss: %.8f" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), 
                     running_loss/config['print_every']))
                running_loss = 0.0

        # calculate training and validation metrics
        model.eval()
        train_acc, train_acc_per_leg = compute_accuracy(train_dataloader, model)
        train_loss_avg = loss_sum/len(train_dataloader)

        (val_acc, val_acc_per_leg, val_loss_avg) = compute_accuracy_and_loss(
            val_dataloader, model, contact_criterion)

        train_acc_per_leg_avg = train_acc_per_leg.mean()  # Average (only one leg now)
        val_acc_per_leg_avg = val_acc_per_leg.mean()  # Average (only one leg now)

        # log down info in tensorboard
        writer.add_scalar('training loss', train_loss_avg, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('training acc left leg', train_acc_per_leg[0], epoch)
        # writer.add_scalar('training acc right leg', train_acc_per_leg[1], epoch)  # COMMENTED OUT
        writer.add_scalar('training acc leg avg', train_acc_per_leg_avg, epoch)
        
        writer.add_scalar('validation loss', val_loss_avg, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)
        writer.add_scalar('validation acc left leg', val_acc_per_leg[0], epoch)
        # writer.add_scalar('validation acc right leg', val_acc_per_leg[1], epoch)  # COMMENTED OUT
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
        # print("train right leg acc: %.4f, val right leg acc: %.4f" %\  # COMMENTED OUT
        #     (train_acc_per_leg[1], val_acc_per_leg[1]))  # COMMENTED OUT
    
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
    
    # # =====================================================================
    # # COUNT CONTACT LABELS BEFORE TRAINING (Class Balance Analysis)
    # # =====================================================================
    # print(f"\n{'='*70}")
    # print(f"CONTACT LABEL DISTRIBUTION (LEFT LEG ONLY)")
    # print(f"{'='*70}")
    
    # # Count labels for each split by collecting all window labels
    # def count_labels_in_indices(dataset, indices):
    #     """Count contact (1) and no-contact (0) labels in given window indices."""
    #     num_contact = 0
    #     num_no_contact = 0
    #     for idx in indices:
    #         label = dataset[idx]['label']  # Shape: (1,) for left leg
    #         if label.item() == 1:
    #             num_contact += 1
    #         else:
    #             num_no_contact += 1
    #     return num_contact, num_no_contact
    
    # print("\nCounting labels in all splits...")
    # train_contact, train_no_contact = count_labels_in_indices(all_dataset, train_indices)
    # val_contact, val_no_contact = count_labels_in_indices(all_dataset, val_indices)
    # test_contact, test_no_contact = count_labels_in_indices(all_dataset, test_indices)
    
    # # Total counts
    # total_contact = train_contact + val_contact + test_contact
    # total_no_contact = train_no_contact + val_no_contact + test_no_contact
    # total_samples = total_contact + total_no_contact
    
    # # Print overall distribution
    # print(f"\n{'='*70}")
    # print(f"OVERALL DATASET (all {total_samples} windows):")
    # print(f"{'='*70}")
    # print(f"  Contact (1):     {total_contact:6d} samples ({100*total_contact/total_samples:5.2f}%)")
    # print(f"  No-contact (0):  {total_no_contact:6d} samples ({100*total_no_contact/total_samples:5.2f}%)")
    # print(f"  Imbalance ratio: 1:{total_no_contact/max(total_contact,1):.2f} (contact:no-contact)")
    
    # # Print train split distribution
    # train_total = train_contact + train_no_contact
    # print(f"\n{'='*70}")
    # print(f"TRAIN SPLIT ({train_total} windows):")
    # print(f"{'='*70}")
    # print(f"  Contact (1):     {train_contact:6d} samples ({100*train_contact/train_total:5.2f}%)")
    # print(f"  No-contact (0):  {train_no_contact:6d} samples ({100*train_no_contact/train_total:5.2f}%)")
    # print(f"  Imbalance ratio: 1:{train_no_contact/max(train_contact,1):.2f} (contact:no-contact)")
    
    # # Print val split distribution
    # val_total = val_contact + val_no_contact
    # print(f"\n{'='*70}")
    # print(f"VALIDATION SPLIT ({val_total} windows):")
    # print(f"{'='*70}")
    # print(f"  Contact (1):     {val_contact:6d} samples ({100*val_contact/val_total:5.2f}%)")
    # print(f"  No-contact (0):  {val_no_contact:6d} samples ({100*val_no_contact/val_total:5.2f}%)")
    # print(f"  Imbalance ratio: 1:{val_no_contact/max(val_contact,1):.2f} (contact:no-contact)")
    
    # # Print test split distribution
    # test_total = test_contact + test_no_contact
    # print(f"\n{'='*70}")
    # print(f"TEST SPLIT ({test_total} windows):")
    # print(f"{'='*70}")
    # print(f"  Contact (1):     {test_contact:6d} samples ({100*test_contact/test_total:5.2f}%)")
    # print(f"  No-contact (0):  {test_no_contact:6d} samples ({100*test_no_contact/test_total:5.2f}%)")
    # print(f"  Imbalance ratio: 1:{test_no_contact/max(test_contact,1):.2f} (contact:no-contact)")
    
    # # Warn if severe class imbalance and suggest pos_weight
    # if train_contact > 0:
    #     imbalance_ratio = train_no_contact / train_contact
    #     suggested_pos_weight = train_no_contact / train_contact  # Weight for positive class (contact)
        
    #     print(f"\nCLASS IMBALANCE ANALYSIS:")
    #     print(f"  No-contact / Contact ratio: {imbalance_ratio:.2f}")
    #     print(f"  Suggested pos_weight for BCEWithLogitsLoss: {suggested_pos_weight:.2f}")
    #     print(f"  (This makes the loss penalize missing contacts {suggested_pos_weight:.2f}x more)")
        
    #     if imbalance_ratio > 3 or imbalance_ratio < 0.33:
    #         print(f"\n⚠️  WARNING: Class imbalance detected!")
    #         print(f"   To fix bias, add to your config file:")
    #         print(f"     use_weighted_loss: true")
    #         print(f"     pos_weight: {suggested_pos_weight:.2f}")
    #     elif abs(imbalance_ratio - 1.0) < 0.2:
    #         print(f"\n✓ Classes are well balanced, no weighting needed.")
    
    # print(f"{'='*70}\n")
    # # =====================================================================
    
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
