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
    
    confusion_mat = {}
    
    # LEFT leg only
    confusion_mat['left_leg'] = confusion_matrix(bin_contact_gt_arr[:,0],bin_contact_pred_arr[:,0], labels=[0,1])
    # confusion_mat['right_leg'] = confusion_matrix(bin_contact_gt_arr[:,1],bin_contact_pred_arr[:,1], labels=[0,1])  # COMMENTED OUT
    confusion_mat['total'] = confusion_mat['left_leg']  # Only left leg
    confusion_mat['total_ratio'] = confusion_mat['total'] / np.sum(confusion_mat['total'])
    
    # false negative and false positive rate
    # false negative = FN/P; false positive = FP/N
    fn_rate = {}
    fp_rate = {}

    fn_rate['left_leg'] = confusion_mat['left_leg'][0,1] / (confusion_mat['left_leg'][0,0]+confusion_mat['left_leg'][0,1])
    # fn_rate['right_leg'] = confusion_mat['right_leg'][0,1] / (confusion_mat['right_leg'][0,0]+confusion_mat['right_leg'][0,1])  # COMMENTED OUT
    fn_rate['total'] = confusion_mat['total'][0,1] / (confusion_mat['total'][0,0]+confusion_mat['total'][0,1])

    fp_rate['left_leg'] = confusion_mat['left_leg'][1,0] / (confusion_mat['left_leg'][1,0] + confusion_mat['left_leg'][1,1])
    # fp_rate['right_leg'] = confusion_mat['right_leg'][1,0] / (confusion_mat['right_leg'][1,0] + confusion_mat['right_leg'][1,1])  # COMMENTED OUT
    fp_rate['total'] = confusion_mat['total'][1,0] / (confusion_mat['total'][1,0] + confusion_mat['total'][1,1])

    return confusion_mat, fn_rate, fp_rate


def compute_precision(bin_pred_arr, bin_gt_arr):
    """Compute precision for LEFT leg binary classification."""
    precision_of_all_legs = precision_score(bin_gt_arr.flatten(),bin_pred_arr.flatten())
    precision_of_legs = []
    for i in range(1):  # 1 leg (LEFT only)
        precision_of_legs.append(precision_score(bin_gt_arr[:,i],bin_pred_arr[:,i]))

    return precision_of_legs, precision_of_all_legs

def compute_jaccard(bin_pred_arr, bin_gt_arr):
    """Compute Jaccard score for LEFT leg binary classification."""
    jaccard_of_all_legs = jaccard_score(bin_gt_arr.flatten(),bin_pred_arr.flatten())
    jaccard_of_legs = []
    for i in range(1):  # 1 leg (LEFT only)
        jaccard_of_legs.append(jaccard_score(bin_gt_arr[:,i],bin_pred_arr[:,i]))

    return jaccard_of_legs, jaccard_of_all_legs

def compute_accuracy(dataloader, model):
    """
    Compute accuracy for LEFT leg binary classification.
    Returns:
        accuracy: overall contact accuracy
        per_leg_accuracy: (1,) per-leg contact accuracy [LEFT only]
        bin_pred_arr: (N, 1) binary predictions [LEFT only]
        bin_gt_arr: (N, 1) binary ground truth [LEFT only]
    """
    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(1)  # 1 leg for LEFT only
    bin_pred_arr = np.zeros((0,1))  # 1 leg for LEFT only
    bin_gt_arr = np.zeros((0,1))  # 1 leg for LEFT only
    # velocity_mse_sum = 0.0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']  # Shape: (batch, 1) - binary labels [LEFT only]
            # gt_velocity = sample['velocity']  # Shape: (batch, 1) - velocity [LEFT only]

            contact_output = model(input_data)  # Only contact output: (batch, 1)
            contact_prediction = (torch.sigmoid(contact_output) > 0.5).float()  # Binary predictions

            bin_pred_arr = np.vstack((bin_pred_arr, contact_prediction.cpu().numpy()))
            bin_gt_arr = np.vstack((bin_gt_arr, gt_label.cpu().numpy()))

            # Per-leg contact accuracy
            correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            # Overall contact accuracy (LEFT leg only)
            num_correct += (contact_prediction == gt_label).sum().item()
            
            # # Velocity MSE
            # velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).sum().item()

    # Total accuracy considers all predictions (LEFT leg only)
    total_predictions = num_data * 1  # 1 leg per sample
    # velocity_mse = velocity_mse_sum / total_predictions
    
    return num_correct/total_predictions, correct_per_leg/num_data, bin_pred_arr, bin_gt_arr  # , velocity_mse

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

    test_acc, acc_per_leg, bin_pred_arr, bin_gt_arr = compute_accuracy(test_dataloader, model)  # removed velocity_mse
    precision_of_legs, precision_of_all_legs = compute_precision(bin_pred_arr, bin_gt_arr)
    jaccard_of_legs, jaccard_of_all_legs = compute_jaccard(bin_pred_arr, bin_gt_arr)
    confusion_mat, fn_rate, fp_rate = compute_confusion_mat(bin_pred_arr, bin_gt_arr)

    print("Test accuracy (LEFT leg): %.4f" % test_acc)
    print("Accuracy of left leg: %.4f" % acc_per_leg[0])
    # print("Accuracy of right leg: %.4f" % acc_per_leg[1])  # COMMENTED OUT - only LEFT leg
    print("Average leg accuracy: %.4f" % acc_per_leg.mean())
    # print("Velocity MSE (both legs): %.6f" % velocity_mse)  # COMMENTED OUT
    print("---------------")
    print("Precision of left leg: %.4f" % precision_of_legs[0])
    # print("Precision of right leg: %.4f" % precision_of_legs[1])  # COMMENTED OUT - only LEFT leg
    print("Precision of all legs: %.4f" % precision_of_all_legs)
    print("---------------")
    print("Jaccard of left leg: %.4f" % jaccard_of_legs[0])
    # print("Jaccard of right leg: %.4f" % jaccard_of_legs[1])  # COMMENTED OUT - only LEFT leg
    print("Jaccard of all legs: %.4f" % jaccard_of_all_legs)
    print("---------------")
    print("confusion matrix of left leg is: ")
    print(confusion_mat['left_leg'])
    # print("confusion matrix of right leg is: ")  # COMMENTED OUT - only LEFT leg
    # print(confusion_mat['right_leg'])  # COMMENTED OUT - only LEFT leg
    print("confusion matrix sum is: ")
    print(confusion_mat['total'])
    print("confusion matrix ratio: ")
    print(confusion_mat['total_ratio'])
    print("---------------")
    print("false negative rate of left leg is: %.4f" % fn_rate['left_leg'])
    # print("false negative rate of right leg is: %.4f" % fn_rate['right_leg'])  # COMMENTED OUT - only LEFT leg
    print("AVG false negative rate is: %.4f" % fn_rate['total'])
    print("---------------")
    print("false positive rate of left leg is: %.4f" % fp_rate['left_leg'])
    # print("false positive rate of right leg is: %.4f" % fp_rate['right_leg'])  # COMMENTED OUT - only LEFT leg
    print("AVG false positive rate is: %.4f" % fp_rate['total'])
    print("---------------")

    print(test_acc)
    print(acc_per_leg[0])
    # print(acc_per_leg[1])  # COMMENTED OUT - only LEFT leg
    print(acc_per_leg.mean())
    print("---------------")
    print(precision_of_legs[0])
    # print(precision_of_legs[1])  # COMMENTED OUT - only LEFT leg
    print("---------------")
    print(precision_of_all_legs)
    print("---------------")
    print(jaccard_of_legs[0])
    # print(jaccard_of_legs[1])  # COMMENTED OUT - only LEFT leg
    print(jaccard_of_all_legs)
    print("---------------")
    print(fn_rate['left_leg'])
    # print(fn_rate['right_leg'])  # COMMENTED OUT - only LEFT leg
    print(fn_rate['total'])
    print("---------------")
    print(fp_rate['left_leg'])
    # print(fp_rate['right_leg'])  # COMMENTED OUT - only LEFT leg
    print(fp_rate['total'])

if __name__ == '__main__':
    main()