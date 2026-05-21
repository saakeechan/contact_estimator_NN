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

def compute_accuracy(dataloader, model):
    """
    Compute metrics for left leg velocity at last timestep.
    Returns:
        velocity_mae: mean absolute error for velocity (on contact samples, last timestep only)
        velocity_mse: mean squared error for velocity (on contact samples, last timestep only)
    """
    num_data = 0
    velocity_mae_sum = 0.0
    velocity_mse_sum = 0.0
    num_contact_samples = 0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # Shape: (batch, 1) - binary labels for left leg (last timestep)
            gt_velocity_seq = sample['velocity']  # Shape: (batch, window_size, 1) - full velocity sequence from dataset
            gt_velocity = gt_velocity_seq[:, -1, :]  # Extract last timestep: (batch, 1)

            velocity_seq, velocity_output = model(input_data)  # velocity_seq: (batch, 1, window_size), velocity_output: (batch, 1)

            num_data += input_data.size(0)
            
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
    
    return velocity_mae, velocity_mse

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
    
    # Ensure at least 1 run per split (same logic as train.py)
    if train_num_runs == 0:
        train_num_runs = 1
        val_num_runs = max(1, (num_runs - train_num_runs) // 2)
    
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
    base_model = contact_cnn(window_size=config['window_size'], num_features=num_features)
    from contact_cnn import ContactCNNWithNormalization
    model = ContactCNNWithNormalization(base_model)  # Will load stats from checkpoint

    checkpoint = torch.load(config['model_load_path'])
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

    velocity_mae, velocity_mse = compute_accuracy(test_dataloader, model)

    print("\n" + "="*60)
    print("LEFT LEG TEST RESULTS")
    print("="*60)
    
    print("\nVelocity Regression Metrics (on contact samples, last timestep only):")
    print("  Velocity MAE: %.6f" % velocity_mae)
    print("  Velocity MSE: %.6f" % velocity_mse)
    print("  Velocity RMSE: %.6f" % np.sqrt(velocity_mse))
    print("="*60)
    
    # Raw values for easy copy-paste
    print("\nRaw Values:")
    print(velocity_mae)
    print(velocity_mse)
    print(np.sqrt(velocity_mse))

if __name__ == '__main__':
    main()