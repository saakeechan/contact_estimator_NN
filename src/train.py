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
    correct_per_leg = np.zeros(1)  # 1 leg (left) for this branch
    velocity_mse_sum = 0.0  # Track velocity prediction error
    
    for sample in tqdm(dataloader):
        input_data = sample['data']
        gt_label = sample['label']  # Shape: (batch, 1) - binary labels for left leg
        gt_velocity = sample['velocity']  # Shape: (batch, 1) - left foot velocity norm

        contact_output, velocity_output = model(input_data)  # Two outputs now
        contact_prediction = (torch.sigmoid(contact_output) > 0.5).float()  # Binary predictions

        # Per-leg accuracy (just left leg)
        correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
        num_data += input_data.size(0)
        # Overall accuracy (same as leg accuracy since only 1 leg)
        num_correct += (contact_prediction == gt_label).sum().item()
        
        # Velocity MSE (for monitoring)
        velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).mean().item()

    return num_correct/num_data, correct_per_leg/num_data, velocity_mse_sum/len(dataloader)

def compute_accuracy_and_loss(dataloader, model, contact_criterion, velocity_criterion, velocity_weight=1.0):

    num_correct = 0
    num_data = 0
    contact_loss_sum = 0
    velocity_loss_sum = 0
    total_loss_sum = 0
    velocity_mse_sum = 0.0
    correct_per_leg = np.zeros(1)  # 1 leg (left) for this branch
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']  # Shape: (batch, 1) - binary labels for left leg
            gt_velocity = sample['velocity']  # Shape: (batch, 1) - left foot velocity norm

            contact_output, velocity_output = model(input_data)  # Two outputs
            contact_prediction = (torch.sigmoid(contact_output) > 0.5).float()  # Binary predictions

            contact_loss = contact_criterion(contact_output, gt_label)
            
            # Mask velocity loss by ground truth contact labels
            # gt_label shape: (batch, 1), velocity shape: (batch, 1)
            contact_mask = gt_label  # (batch, 1)
            velocity_loss_elementwise = velocity_criterion(velocity_output, gt_velocity)
            # Multiply by contact mask (only penalize velocities in contact)
            velocity_loss = (velocity_loss_elementwise * contact_mask).sum() / (contact_mask.sum() + 1e-8)
            
            total_loss = contact_loss + velocity_weight * velocity_loss

            # Per-leg accuracy (just left leg)
            correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            # Overall accuracy (same as leg accuracy since only 1 leg)
            num_correct += (contact_prediction == gt_label).sum().item()

            contact_loss_sum += contact_loss.item()
            velocity_loss_sum += velocity_loss.item()
            total_loss_sum += total_loss.item()
            velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).mean().item()

    return (num_correct/num_data, correct_per_leg/num_data, 
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
    The model now has two outputs: 
    - contact predictions (left leg only)
    - foot velocity norm (left foot only)
    """
    try:
        import warnings
        
        # Create ONNX path (replace .pt with .onnx)
        onnx_path = checkpoint_path.replace('.pt', '.onnx')
        
        device = next(model.parameters()).device
        model.eval()
        
        # Create example input (pre-engineered data - 31 features from csv2numpy.py)
        example_input = torch.randn(1, window_size, 32).to(device)
        
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
        
        print(f"  ✓ ONNX model saved with two outputs (contact, velocity): {onnx_path}")
        
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


    # Multi-task learning: contact classification + velocity regression
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
            contact_label = samples['label']
            velocity_label = samples['velocity']

            optimizer.zero_grad()
            contact_output, velocity_output = model(input_data)  # Two outputs

            # Compute losses for both tasks
            contact_loss = contact_criterion(contact_output, contact_label)
            
            # Mask velocity loss by ground truth contact labels
            # contact_label shape: (batch, 1), velocity shape: (batch, 1)
            contact_mask = contact_label  # (batch, 1) - left leg only
            velocity_loss_elementwise = velocity_criterion(velocity_output, velocity_label)
            # Multiply by contact mask (only penalize velocities in contact)
            velocity_loss = (velocity_loss_elementwise * contact_mask).sum() / (contact_mask.sum() + 1e-8)
            
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
            
            # Add temporal consistency loss for contact continuity
            if temporal_lambda > 0 and contact_output.size(0) > 1:
                # Convert logits to probabilities [0,1] for meaningful distance metric
                contact_predictions = torch.sigmoid(contact_output)
                
                # Compute differences between consecutive predictions
                temporal_diff = contact_predictions[1:] - contact_predictions[:-1]
                
                # L1 loss: penalize absolute differences
                temporal_loss = torch.abs(temporal_diff).mean()
                
                loss = loss + temporal_lambda * temporal_loss
            
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

        train_acc_per_leg_avg = train_acc_per_leg[0]  # Only 1 leg (left)
        val_acc_per_leg_avg = val_acc_per_leg[0]  # Only 1 leg (left)

        # log down info in tensorboard
        writer.add_scalar('training loss', train_loss_avg, epoch)
        writer.add_scalar('training contact loss', train_contact_loss_avg, epoch)
        writer.add_scalar('training velocity loss', train_velocity_loss_avg, epoch)
        writer.add_scalar('training velocity MSE', train_velocity_mse, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('training acc left leg', train_acc_per_leg[0], epoch)
        writer.add_scalar('training acc leg avg', train_acc_per_leg_avg, epoch)
        
        writer.add_scalar('validation loss', val_loss_avg, epoch)
        writer.add_scalar('validation contact loss', val_contact_loss_avg, epoch)
        writer.add_scalar('validation velocity loss', val_velocity_loss_avg, epoch)
        writer.add_scalar('validation velocity MSE', val_velocity_mse, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)
        writer.add_scalar('validation acc left leg', val_acc_per_leg[0], epoch)
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
    
    # Split the windows into train/val/test
    dataset_size = len(all_dataset)
    indices = list(range(dataset_size))
    
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    # Shuffle indices if specified
    if config['shuffle']:
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Total windows: {dataset_size}")
    print(f"  Train windows: {len(train_indices)}")
    print(f"  Val windows: {len(val_indices)}")
    print(f"  Test windows: {len(test_indices)}")
    
    # Create Subset datasets for deterministic sampling
    from torch.utils.data import Subset
    train_dataset = Subset(all_dataset, train_indices)
    val_dataset = Subset(all_dataset, val_indices)
    
    # Compute global normalization statistics from TRAINING data only
    # Extract only training samples to compute unbiased statistics
    # Features are already engineered in csv2numpy.py (qd_norm, tau_est_norm, tau_mse, cmd_vel)
    train_data_samples = all_dataset.data[train_indices]  # Get only training data points (32 features)
    
    # Feature layout (32 features): acc(3) + omega(3) + q(6) + qd_norm(6) + p(3) + v(3) + tau_est_norm(6) + tau_mse(1) + cmd_vel(1)
    # No additional feature engineering needed - already done in csv2numpy.py
    train_data_engineered = train_data_samples  # (num_train_samples, 32)



    # Compute 1st and 99th percentiles for each feature to clip outliers
    percentile_1 = torch.quantile(train_data_engineered, 0.01, dim=0, keepdim=True)  # (1, 31)
    percentile_99 = torch.quantile(train_data_engineered, 0.99, dim=0, keepdim=True)  # (1, 31)
    
    # Clip training data to percentile bounds
    train_data_clipped = torch.clamp(train_data_engineered, min=percentile_1, max=percentile_99)
    
    # Compute mean and std per feature from clipped data (31 features total)
    # Layout: acc(0-2) + omega(3-5) + q(6-11) + qd_norm(12-17) + p(18-20) + v(21-23) + tau_est_norm(24-29) + tau_mse(30)
    global_mean = train_data_clipped.mean(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, 31)
    global_std = train_data_clipped.std(dim=0, keepdim=True).unsqueeze(0)    # Shape: (1, 1, 31)
    
    # Handle features with zero std (constant values) to avoid division by zero
    global_std = torch.where(global_std == 0, torch.ones_like(global_std), global_std)
    
    print(f"\nGlobal normalization statistics computed from clipped training data:")
    print(f"  Total features: 32 (acc + omega + q + qd_norm + p + v + tau_est_norm + tau_mse + cmd_vel)")
    print(f"  Features engineered in csv2numpy.py (qd and tau_est normalized by cmd_vel)")
    print(f"  Clipped to [1st, 99th] percentiles per feature")
    print(f"  Mean shape: {global_mean.shape}")
    print(f"  Std shape: {global_std.shape}")
    print(f"  Mean range: [{global_mean.min().item():.4f}, {global_mean.max().item():.4f}]")
    print(f"  Std range: [{global_std.min().item():.4f}, {global_std.max().item():.4f}]")
    print(f"  tau_mse mean: {global_mean[0, 0, 30].item():.4f}, std: {global_std[0, 0, 30].item():.4f}")
    print(f"  cmd_vel mean: {global_mean[0, 0, 31].item():.4f}, std: {global_std[0, 0, 31].item():.4f}")
    
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
