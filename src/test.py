import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm

import torch.optim as optim

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix

from contact_cnn import *
from utils.data_handler import *

def compute_confusion_mat(bin_contact_pred_arr, bin_contact_gt_arr):
    """Compute confusion matrices for left/right contact classification."""
    confusion_mat = {}
    
    for leg_idx, leg_name in enumerate(['left_leg', 'right_leg']):
        leg_cm = confusion_matrix(
            bin_contact_gt_arr[:, leg_idx],
            bin_contact_pred_arr[:, leg_idx],
            labels=[0, 1]
        )
        confusion_mat[leg_name] = leg_cm
        confusion_mat[f'{leg_name}_ratio'] = leg_cm / (np.sum(leg_cm) + 1e-8)

    combined_cm = confusion_matrix(
        bin_contact_gt_arr.flatten(),
        bin_contact_pred_arr.flatten(),
        labels=[0, 1]
    )
    confusion_mat['combined'] = combined_cm
    confusion_mat['combined_ratio'] = combined_cm / (np.sum(combined_cm) + 1e-8)

    fn_rate = combined_cm[1, 0] / (combined_cm[1, 0] + combined_cm[1, 1] + 1e-8)
    fp_rate = combined_cm[0, 1] / (combined_cm[0, 0] + combined_cm[0, 1] + 1e-8)

    return confusion_mat, fn_rate, fp_rate


def compute_precision(bin_pred_arr, bin_gt_arr):
    """Compute precision across left/right binary contact classification."""
    precision = precision_score(bin_gt_arr.flatten(), bin_pred_arr.flatten(), zero_division=0)
    return precision

def compute_jaccard(bin_pred_arr, bin_gt_arr):
    """Compute Jaccard score across left/right binary contact classification."""
    jaccard = jaccard_score(bin_gt_arr.flatten(), bin_pred_arr.flatten(), zero_division=0)
    return jaccard

def compute_accuracy(dataloader, model, device=torch.device('cpu')):
    """
    Compute metrics for left/right velocity and contact classification.
    Returns:
        dict with overall and per-leg metrics.
    """
    velocity_abs_error_sum = torch.zeros(2, device=device)
    velocity_sq_error_sum = torch.zeros(2, device=device)
    num_contact_samples = torch.zeros(2, device=device)
    
    all_contact_preds = []
    all_contact_gt = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # [B, 2] - left/right contact at last timestep
            gt_velocity_seq = sample['velocity']  # [B, T, 2] - full left/right velocity sequence
            gt_velocity = gt_velocity_seq[:, -1, :]  # [B, 2] - last timestep

            velocity_seq, velocity_output, contact_output = model(input_data)  # velocity_seq: [B, 2, T], velocity/contact: [B, 2]
            
            # Collect contact predictions for classification metrics
            contact_pred_binary = (contact_output > 0).long()  # Contact outputs are logits.
            all_contact_preds.append(contact_pred_binary.cpu())
            all_contact_gt.append(gt_contact.cpu())
            
            # Velocity metrics (only on contact samples, last timestep only)
            contact_mask = (gt_contact == 1).float()  # [B, 2]
            if contact_mask.sum() > 0:
                # Compute errors at last timestep for contact samples
                velocity_errors = torch.abs(velocity_output - gt_velocity)  # [B, 2]
                velocity_sq_errors = (velocity_output - gt_velocity) ** 2  # [B, 2]
                velocity_abs_error_sum += (velocity_errors * contact_mask).sum(dim=0)
                velocity_sq_error_sum += (velocity_sq_errors * contact_mask).sum(dim=0)
                num_contact_samples += contact_mask.sum(dim=0)
    
    # Compute contact classification metrics
    all_contact_preds = torch.cat(all_contact_preds, dim=0).numpy()
    all_contact_gt = torch.cat(all_contact_gt, dim=0).numpy()

    flat_contact_preds = all_contact_preds.flatten()
    flat_contact_gt = all_contact_gt.flatten()

    per_leg_velocity_mae = (
        velocity_abs_error_sum / (num_contact_samples + 1e-8)
    ).cpu().numpy()
    per_leg_velocity_mse = (
        velocity_sq_error_sum / (num_contact_samples + 1e-8)
    ).cpu().numpy()

    contact_precision = precision_score(flat_contact_gt, flat_contact_preds, zero_division=0)
    contact_recall = recall_score(flat_contact_gt, flat_contact_preds, zero_division=0)
    contact_f1 = 2 * (contact_precision * contact_recall) / (contact_precision + contact_recall) if (contact_precision + contact_recall) > 0 else 0

    metrics = {
        'velocity_mae': float(velocity_abs_error_sum.sum().item() / (num_contact_samples.sum().item() + 1e-8)),
        'velocity_mse': float(velocity_sq_error_sum.sum().item() / (num_contact_samples.sum().item() + 1e-8)),
        'contact_accuracy': float(np.mean(flat_contact_preds == flat_contact_gt)),
        'contact_precision': float(contact_precision),
        'contact_recall': float(contact_recall),
        'contact_f1': float(contact_f1),
        'per_leg': {
            'left': {
                'velocity_mae': float(per_leg_velocity_mae[0]),
                'velocity_mse': float(per_leg_velocity_mse[0]),
                'contact_accuracy': float(np.mean(all_contact_preds[:, 0] == all_contact_gt[:, 0])),
                'contact_precision': float(precision_score(all_contact_gt[:, 0], all_contact_preds[:, 0], zero_division=0)),
                'contact_recall': float(recall_score(all_contact_gt[:, 0], all_contact_preds[:, 0], zero_division=0)),
            },
            'right': {
                'velocity_mae': float(per_leg_velocity_mae[1]),
                'velocity_mse': float(per_leg_velocity_mse[1]),
                'contact_accuracy': float(np.mean(all_contact_preds[:, 1] == all_contact_gt[:, 1])),
                'contact_precision': float(precision_score(all_contact_gt[:, 1], all_contact_preds[:, 1], zero_division=0)),
                'contact_recall': float(recall_score(all_contact_gt[:, 1], all_contact_preds[:, 1], zero_division=0)),
            }
        }
    }

    for leg_name in ['left', 'right']:
        leg_precision = metrics['per_leg'][leg_name]['contact_precision']
        leg_recall = metrics['per_leg'][leg_name]['contact_recall']
        metrics['per_leg'][leg_name]['contact_f1'] = (
            2 * (leg_precision * leg_recall) / (leg_precision + leg_recall)
            if (leg_precision + leg_recall) > 0 else 0
        )

    return metrics

def decimal2binary(x):
    mask = 2**torch.arange(2-1,-1,-1).to(x.device, x.dtype)  # 2 legs for biped
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
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

    # Load ALL data (not pre-split) - use same splitting logic as train.py
    all_dataset = contact_dataset(data_path=config['data_folder']+"all_data.npy",\
                                  label_path=config['data_folder']+"all_labels.npy",\
                                  window_size=config['window_size'],device=device)
    if all_dataset.label.ndim != 2 or all_dataset.label.shape[1] != 2:
        raise ValueError(
            "Expected all_labels.npy to have shape (N, 2) ordered [left, right]. "
            "Regenerate the numpy dataset with the updated csv2numpy script."
        )
    if all_dataset.foot_velocity.ndim != 2 or all_dataset.foot_velocity.shape[1] != 2:
        raise ValueError(
            "Expected all_foot_velocities.npy to have shape (N, 2) ordered [left, right]. "
            "Regenerate the numpy dataset with the updated csv2numpy script."
        )
    
    # Split by RUNS (not windows) to prevent data leakage - must match train.py exactly
    # Get all unique run IDs
    all_run_ids = np.unique(all_dataset.window_to_run_id)
    num_runs = len(all_run_ids)
    
    # Validate sufficient runs
    if num_runs < 3:
        print(f"Warning: Only {num_runs} runs available. Test split may be empty or small.")
    
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    
    train_num_runs = int(train_ratio * num_runs)
    val_num_runs = int(val_ratio * num_runs)
    test_num_runs = num_runs - train_num_runs - val_num_runs  # Test gets the remainder
    
    # Ensure at least 1 run per split (same logic as train.py)
    if train_num_runs == 0:
        train_num_runs = 1
        val_num_runs = max(1, (num_runs - train_num_runs) // 2)
        test_num_runs = num_runs - train_num_runs - val_num_runs
    
    print(f"\nDataset split: {train_num_runs} train runs, {val_num_runs} val runs, {test_num_runs} test runs (total: {num_runs})")
    
    # Use same random seed and shuffle setting as training to get EXACT same split
    if config.get('shuffle', True):
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(all_run_ids)
    
    # Split run IDs (must match train.py)
    test_run_ids = set(all_run_ids[train_num_runs + val_num_runs:])
    
    # Get all windows belonging to test runs
    test_indices = all_dataset.get_windows_by_run_ids(test_run_ids)
    
    if len(test_indices) == 0:
        raise ValueError(f"No test windows found. Dataset may be too small or split ratios may need adjustment.")
    
    print(f"\nTesting on {len(test_run_ids)} runs -> {len(test_indices)} windows")
    print(f"Total runs in dataset: {num_runs}")
    
    # Create subset dataset for test data (no batch shuffling for deterministic results)
    from torch.utils.data import Subset
    test_dataset = Subset(all_dataset, test_indices)
    
    # Use test_batch_size if specified, otherwise fall back to batch_size
    test_batch_size = config.get('test_batch_size', config.get('batch_size', 30))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,\
                                 shuffle=False)


    # init network with built-in normalization (same as training)
    # num_features loaded from metadata at the start of main()
    # Select model architecture based on config (must match training)
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
        base_model = contact_cnn(window_size=config['window_size'], num_features=num_features)
    else:
        raise ValueError(f"Unknown model_architecture: {model_arch}. Options: 'attention_tcn', 'tcn', 'vanilla_cnn'")
    
    from contact_cnn import ContactCNNWithNormalization
    model = ContactCNNWithNormalization(base_model)  # Will load stats from checkpoint

    # Find latest PyTorch checkpoint from logs
    logs_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    latest_pt = None
    if os.path.exists(logs_root):
        # list run folders sorted by modified time
        run_dirs = [os.path.join(logs_root, d) for d in os.listdir(logs_root) if os.path.isdir(os.path.join(logs_root, d))]
        if run_dirs:
            run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for rd in run_dirs:
                candidate = os.path.join(rd, 'model_best_val_velocity.pt')
                if os.path.exists(candidate):
                    latest_pt = candidate
                    break

    if latest_pt is None:
        raise FileNotFoundError(f"No model_best_val_velocity.pt found in logs directory: {logs_root}")

    # Load PyTorch checkpoint
    print(f"Using PyTorch checkpoint from latest run: {latest_pt}")
    checkpoint = torch.load(latest_pt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)
    
    # DEBUG: Verify normalization stats were loaded correctly
    print(f"\n{'='*60}")
    print(f"DEBUG: Normalization stats in test model")
    print(f"{'='*60}")
    print(f"Global mean shape: {model.global_mean.shape}")
    print(f"Global std shape: {model.global_std.shape}")
    print(f"Global mean range: [{model.global_mean.min().item():.4f}, {model.global_mean.max().item():.4f}]")
    print(f"Global std range: [{model.global_std.min().item():.4f}, {model.global_std.max().item():.4f}]")
    
    # Check if using default fallback values (zeros/ones)
    if torch.allclose(model.global_mean, torch.zeros_like(model.global_mean)):
        print(f"⚠️  WARNING: global_mean is all zeros (using fallback - normalization NOT loaded!)")
    if torch.allclose(model.global_std, torch.ones_like(model.global_std)):
        print(f"⚠️  WARNING: global_std is all ones (using fallback - normalization NOT loaded!)")
    print(f"{'='*60}\n")

    metrics = compute_accuracy(test_dataloader, model, device=device)

    print("\n" + "="*60)
    print("BIPED TEST RESULTS")
    print("="*60)
    
    print("\nContact Classification Metrics (combined left + right):")
    print("  Accuracy:  %.4f" % metrics['contact_accuracy'])
    print("  Precision: %.4f" % metrics['contact_precision'])
    print("  Recall:    %.4f" % metrics['contact_recall'])
    print("  F1 Score:  %.4f" % metrics['contact_f1'])
    
    print("\nVelocity Regression Metrics (combined contact samples, last timestep only):")
    print("  Velocity MAE: %.6f" % metrics['velocity_mae'])
    print("  Velocity MSE: %.6f" % metrics['velocity_mse'])
    print("  Velocity RMSE: %.6f" % np.sqrt(metrics['velocity_mse']))

    for leg_name in ['left', 'right']:
        leg_metrics = metrics['per_leg'][leg_name]
        print(f"\n{leg_name.capitalize()} Leg Metrics:")
        print("  Contact Accuracy:  %.4f" % leg_metrics['contact_accuracy'])
        print("  Contact Precision: %.4f" % leg_metrics['contact_precision'])
        print("  Contact Recall:    %.4f" % leg_metrics['contact_recall'])
        print("  Contact F1 Score:  %.4f" % leg_metrics['contact_f1'])
        print("  Velocity MAE:      %.6f" % leg_metrics['velocity_mae'])
        print("  Velocity MSE:      %.6f" % leg_metrics['velocity_mse'])
        print("  Velocity RMSE:     %.6f" % np.sqrt(leg_metrics['velocity_mse']))
    print("="*60)

if __name__ == '__main__':
    main()
