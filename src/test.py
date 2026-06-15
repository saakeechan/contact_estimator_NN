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
    """Compute confusion matrix for left leg only."""
    confusion_mat = {}
    
    # Only left leg (single column)
    confusion_mat['left_leg'] = confusion_matrix(bin_contact_gt_arr.flatten(), bin_contact_pred_arr.flatten(), labels=[0,1])
    confusion_mat['ratio'] = confusion_mat['left_leg'] / np.sum(confusion_mat['left_leg'])
    
    # false negative and false positive rate
    # false negative = FN/P; false positive = FP/N
    fn_rate = confusion_mat['left_leg'][1,0] / (confusion_mat['left_leg'][1,0] + confusion_mat['left_leg'][1,1])
    fp_rate = confusion_mat['left_leg'][0,1] / (confusion_mat['left_leg'][0,0] + confusion_mat['left_leg'][0,1])

    return confusion_mat, fn_rate, fp_rate


def compute_precision(bin_pred_arr, bin_gt_arr):
    """Compute precision for left leg binary classification."""
    precision = precision_score(bin_gt_arr.flatten(), bin_pred_arr.flatten())
    return precision

def compute_jaccard(bin_pred_arr, bin_gt_arr):
    """Compute Jaccard score for left leg binary classification."""
    jaccard = jaccard_score(bin_gt_arr.flatten(), bin_pred_arr.flatten())
    return jaccard

def compute_accuracy(dataloader, model, device=torch.device('cpu')):
    """
    Compute metrics for left leg velocity and contact classification.
    Returns:
        velocity_mae: mean absolute error for velocity (on contact samples, last timestep only)
        velocity_mse: mean squared error for velocity (on contact samples, last timestep only)
        contact_accuracy: binary classification accuracy
        contact_precision: precision score
        contact_recall: recall score
        contact_f1: F1 score
    """
    num_data = 0
    velocity_mae_sum = 0.0
    velocity_mse_sum = 0.0
    num_contact_samples = 0
    
    all_contact_preds = []
    all_contact_gt = []
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # Shape: (batch, 1) - binary labels for left leg (last timestep)
            gt_velocity_seq = sample['velocity']  # Shape: (batch, window_size, 1) - full velocity sequence from dataset
            gt_velocity = gt_velocity_seq[:, -1, :]  # Extract last timestep: (batch, 1)

            velocity_seq, velocity_output, contact_output = model(input_data)  # velocity_seq: (batch, 1, window_size), velocity_output: (batch, 1), contact: (batch, 1)

            num_data += input_data.size(0)
            
            # Collect contact predictions for classification metrics
            contact_pred_binary = (contact_output > 0.5).long()  # Threshold at 0.5
            all_contact_preds.append(contact_pred_binary.cpu())
            all_contact_gt.append(gt_contact.cpu())
            
            # Velocity metrics (only on contact samples, last timestep only)
            contact_mask = (gt_contact == 1)  # (batch, 1)
            if contact_mask.sum() > 0:
                # Compute errors at last timestep for contact samples
                velocity_errors = torch.abs(velocity_output - gt_velocity)  # (batch, 1)
                velocity_errors_masked = velocity_errors * contact_mask  # Zero out non-contact samples
                
                velocity_mae_sum += velocity_errors_masked.sum().item()
                velocity_mse_sum += ((velocity_output - gt_velocity) ** 2 * contact_mask).sum().item()
                
                # Count contact samples
                num_contact_samples += contact_mask.sum().item()

    velocity_mae = velocity_mae_sum / num_contact_samples if num_contact_samples > 0 else 0
    velocity_mse = velocity_mse_sum / num_contact_samples if num_contact_samples > 0 else 0
    
    # Compute contact classification metrics
    all_contact_preds = torch.cat(all_contact_preds, dim=0).numpy().flatten()
    all_contact_gt = torch.cat(all_contact_gt, dim=0).numpy().flatten()
    
    contact_accuracy = np.mean(all_contact_preds == all_contact_gt)
    contact_precision = precision_score(all_contact_gt, all_contact_preds, zero_division=0)
    contact_recall = recall_score(all_contact_gt, all_contact_preds, zero_division=0)
    contact_f1 = 2 * (contact_precision * contact_recall) / (contact_precision + contact_recall) if (contact_precision + contact_recall) > 0 else 0
    
    return velocity_mae, velocity_mse, contact_accuracy, contact_precision, contact_recall, contact_f1

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

    velocity_mae, velocity_mse, contact_accuracy, contact_precision, contact_recall, contact_f1 = compute_accuracy(
        test_dataloader, model, device=device)

    print("\n" + "="*60)
    print("LEFT LEG TEST RESULTS")
    print("="*60)
    
    print("\nContact Classification Metrics:")
    print("  Accuracy:  %.4f" % contact_accuracy)
    print("  Precision: %.4f" % contact_precision)
    print("  Recall:    %.4f" % contact_recall)
    print("  F1 Score:  %.4f" % contact_f1)
    
    print("\nVelocity Regression Metrics (on contact samples, last timestep only):")
    print("  Velocity MAE: %.6f" % velocity_mae)
    print("  Velocity MSE: %.6f" % velocity_mse)
    print("  Velocity RMSE: %.6f" % np.sqrt(velocity_mse))
    print("="*60)
    
    # Raw values for easy copy-paste
    print("\nRaw Values:")
    print(f"Contact: acc={contact_accuracy:.4f}, prec={contact_precision:.4f}, recall={contact_recall:.4f}, f1={contact_f1:.4f}")
    print(f"Velocity: mae={velocity_mae:.6f}, mse={velocity_mse:.6f}, rmse={np.sqrt(velocity_mse):.6f}")

if __name__ == '__main__':
    main()