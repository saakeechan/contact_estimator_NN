import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix

from contact_cnn import *
from utils.data_handler import *

PLOT_WINDOW_SIZE = 100

def compute_confusion_mat(bin_contact_pred_arr, bin_contact_gt_arr):
    """Compute left-foot and combined confusion matrices."""
    confusion_mat = {
        leg: confusion_matrix(bin_contact_gt_arr[:, index], bin_contact_pred_arr[:, index], labels=[0, 1])
        for index, leg in enumerate(('left_leg',))
    }
    combined = confusion_matrix(bin_contact_gt_arr.ravel(), bin_contact_pred_arr.ravel(), labels=[0, 1])
    confusion_mat['combined'] = combined
    fn_rate = combined[1, 0] / (combined[1, 0] + combined[1, 1] + 1e-8)
    fp_rate = combined[0, 1] / (combined[0, 0] + combined[0, 1] + 1e-8)

    return confusion_mat, fn_rate, fp_rate


def compute_precision(bin_pred_arr, bin_gt_arr):
    """Compute left-foot precision."""
    precision = precision_score(bin_gt_arr.ravel(), bin_pred_arr.ravel(), zero_division=0)
    return precision

def compute_jaccard(bin_pred_arr, bin_gt_arr):
    """Compute left-foot Jaccard score."""
    jaccard = jaccard_score(bin_gt_arr.ravel(), bin_pred_arr.ravel(), zero_division=0)
    return jaccard

def compute_accuracy(dataloader, model, device=torch.device('cpu')):
    """
    Compute left-foot contact and body-velocity metrics.
    """
    velocity_abs_error_sum = torch.zeros((1, 3), device=device)
    velocity_sq_error_sum = torch.zeros((1, 3), device=device)
    num_contact_samples = torch.zeros(1, device=device)
    
    true_positive = torch.zeros(1, device=device)
    false_positive = torch.zeros(1, device=device)
    false_negative = torch.zeros(1, device=device)
    true_negative = torch.zeros(1, device=device)
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_contact = sample['label']  # [B, 1] left foot
            gt_velocity_seq = sample['velocity']  # [B, T, 1, 3]
            gt_velocity = gt_velocity_seq[:, -1, :, :]  # [B, 1, 3]

            velocity_seq, velocity_output, covariance_seq, covariance_output, contact_output = model(
                input_data, return_sequence=False
            )
            
            # Collect contact predictions for classification metrics
            contact_pred_binary = contact_output > 0  # Logit threshold at 0.0
            contact_gt_binary = gt_contact > 0.5
            true_positive += (contact_pred_binary & contact_gt_binary).sum(dim=0)
            false_positive += (contact_pred_binary & ~contact_gt_binary).sum(dim=0)
            false_negative += (~contact_pred_binary & contact_gt_binary).sum(dim=0)
            true_negative += (~contact_pred_binary & ~contact_gt_binary).sum(dim=0)
            
            # Velocity metrics (only on contact samples, last timestep only)
            contact_mask = (gt_contact == 1).float().unsqueeze(-1)  # [B, 1, 1]
            if contact_mask.sum() > 0:
                velocity_abs_error_sum += (torch.abs(velocity_output - gt_velocity) * contact_mask).sum(dim=0)
                velocity_sq_error_sum += (((velocity_output - gt_velocity) ** 2) * contact_mask).sum(dim=0)
                num_contact_samples += contact_mask.squeeze(-1).sum(dim=0)
    
    total_true_positive = true_positive.sum().item()
    total_false_positive = false_positive.sum().item()
    total_false_negative = false_negative.sum().item()
    total_true_negative = true_negative.sum().item()
    contact_accuracy = (total_true_positive + total_true_negative) / (total_true_positive + total_false_positive + total_false_negative + total_true_negative + 1e-8)
    contact_precision = total_true_positive / (total_true_positive + total_false_positive + 1e-8)
    contact_recall = total_true_positive / (total_true_positive + total_false_negative + 1e-8)
    contact_f1 = 2 * contact_precision * contact_recall / (contact_precision + contact_recall + 1e-8)

    per_leg_mae = (velocity_abs_error_sum / (num_contact_samples[:, None] + 1e-8)).cpu().numpy()
    per_leg_mse = (velocity_sq_error_sum / (num_contact_samples[:, None] + 1e-8)).cpu().numpy()
    metrics = {
        'velocity_mae': float(velocity_abs_error_sum.sum().item() / (num_contact_samples.sum().item() * 3 + 1e-8)),
        'velocity_mse': float(velocity_sq_error_sum.sum().item() / (num_contact_samples.sum().item() * 3 + 1e-8)),
        'contact_accuracy': float(contact_accuracy),
        'contact_precision': float(contact_precision),
        'contact_recall': float(contact_recall),
        'contact_f1': float(contact_f1),
        'per_leg': {},
    }
    for index, leg in enumerate(('left',)):
        precision = true_positive[index].item() / (true_positive[index].item() + false_positive[index].item() + 1e-8)
        recall = true_positive[index].item() / (true_positive[index].item() + false_negative[index].item() + 1e-8)
        accuracy = (true_positive[index].item() + true_negative[index].item()) / (
            true_positive[index].item() + false_positive[index].item() + false_negative[index].item() + true_negative[index].item() + 1e-8
        )
        metrics['per_leg'][leg] = {
            'velocity_mae': float(per_leg_mae[index].mean()),
            'velocity_mse': float(per_leg_mse[index].mean()),
            'velocity_mae_components': per_leg_mae[index],
            'velocity_mse_components': per_leg_mse[index],
            'contact_accuracy': float(accuracy),
            'contact_precision': float(precision),
            'contact_recall': float(recall),
            'contact_f1': float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
    return metrics


def save_velocity_plots(dataloader, model, output_dir):
    """Save a three-component body-velocity predicted-vs-ground-truth plot."""
    predicted, ground_truth, contact = [], [], []
    with torch.no_grad():
        for sample in dataloader:
            _, velocity_output, _, _, _ = model(sample['data'], return_sequence=False)
            predicted.append(velocity_output.cpu())
            ground_truth.append(sample['velocity'][:, -1, :, :].cpu())
            contact.append(sample['label'].cpu())

    predicted = torch.cat(predicted).numpy()
    ground_truth = torch.cat(ground_truth).numpy()
    contact = torch.cat(contact).numpy()
    sample_index = np.arange(len(contact))
    os.makedirs(output_dir, exist_ok=True)

    for leg_index, leg_name in enumerate(('body',)):
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        contact_changes = np.diff(np.concatenate(([0], contact[:, leg_index] > 0.5, [0])))
        contact_starts = np.where(contact_changes == 1)[0]
        contact_ends = np.where(contact_changes == -1)[0]

        for component, ax in enumerate(axes):
            for start, end in zip(contact_starts, contact_ends):
                ax.axvspan(start, end - 1, color='red', alpha=0.2)
            ax.plot(sample_index, ground_truth[:, leg_index, component], color='black', label='Ground truth')
            ax.plot(sample_index, predicted[:, leg_index, component], color='tab:blue', linestyle='--', label='Predicted')
            ax.set_ylabel(f'v{"xyz"[component]} (m/s)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        axes[0].set_title('Body-frame body velocity: prediction vs ground truth')
        axes[-1].set_xlabel('Time step in sampled window')
        fig.tight_layout()
        output_path = os.path.join(output_dir, 'body_velocity_comparison.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved body-velocity plot: {output_path}')

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

    plot_starts = [
        start for start in range(len(test_indices) - PLOT_WINDOW_SIZE + 1)
        if all_dataset.get_run_id(test_indices[start]) == all_dataset.get_run_id(test_indices[start + PLOT_WINDOW_SIZE - 1])
    ]
    if not plot_starts:
        raise ValueError(f"No test run has the {PLOT_WINDOW_SIZE} windows required for velocity plots.")
    plot_start = int(np.random.default_rng().choice(plot_starts))
    plot_indices = test_indices[plot_start:plot_start + PLOT_WINDOW_SIZE]
    plot_dataloader = DataLoader(Subset(all_dataset, plot_indices), batch_size=test_batch_size, shuffle=False)
    print(f"Plotting {PLOT_WINDOW_SIZE} random consecutive test windows from run {all_dataset.get_run_id(plot_indices[0])}.")


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
            tcn_dropout=config.get('tcn_dropout', 0.2),
            natpn_flow_layers=config.get('natpn_flow_layers', 8),
            natpn_certainty_budget=config.get('natpn_certainty_budget', 'normal'),
            natpn_evidence_source=config.get('natpn_evidence_source', 'task'),
            input_natpn_checkpoint=config.get('input_natpn_checkpoint'),
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
    save_velocity_plots(plot_dataloader, model, os.path.dirname(latest_pt))

    print("\n" + "="*60)
    print("BODY-VELOCITY TEST RESULTS")
    print("="*60)
    
    print("\nContact Classification Metrics (combined):")
    print("  Accuracy:  %.4f" % metrics['contact_accuracy'])
    print("  Precision: %.4f" % metrics['contact_precision'])
    print("  Recall:    %.4f" % metrics['contact_recall'])
    print("  F1 Score:  %.4f" % metrics['contact_f1'])
    
    print("\nBody-Velocity Regression Metrics (on contact samples, last timestep only):")
    print("  Velocity MAE: %.6f" % metrics['velocity_mae'])
    print("  Velocity MSE: %.6f" % metrics['velocity_mse'])
    print("  Velocity RMSE: %.6f" % np.sqrt(metrics['velocity_mse']))
    for leg, leg_metrics in metrics['per_leg'].items():
        print(f"\n{leg.capitalize()}:")
        print("  Contact Accuracy:  %.4f" % leg_metrics['contact_accuracy'])
        print("  Contact Precision: %.4f" % leg_metrics['contact_precision'])
        print("  Contact Recall:    %.4f" % leg_metrics['contact_recall'])
        print("  Contact F1:        %.4f" % leg_metrics['contact_f1'])
        print("  Velocity MAE:      %.6f [vx, vy, vz: %.6f, %.6f, %.6f]" % (
            leg_metrics['velocity_mae'], *leg_metrics['velocity_mae_components']))
        print("  Velocity MSE:      %.6f [vx, vy, vz: %.6f, %.6f, %.6f]" % (
            leg_metrics['velocity_mse'], *leg_metrics['velocity_mse_components']))
    print("="*60)
    

if __name__ == '__main__':
    main()
