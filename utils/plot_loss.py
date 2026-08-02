import os
import argparse
import glob
import json
import time

# Set matplotlib cache directory to /tmp to avoid permission issues in Docker
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dir):
    """Load scalars from TensorBoard event files."""
    # Find all event files in the directory
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    # Use the most recent event file
    event_file = max(event_files, key=os.path.getmtime)
    print(f"Loading from: {event_file}")
    
    # Load the events
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Print available tags
    print(f"Available scalar tags: {ea.Tags()['scalars']}")
    
    return ea


def extract_scalars(ea, tag):
    """Extract values from a scalar tag."""
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        print(f"Warning: Tag '{tag}' not found")
        return [], []


def generate_training_summary(run_dir, config, train_metrics, val_metrics, best_metrics, 
                              train_time_seconds, checkpoint_paths):
    """
    Generate comprehensive training summary with plots and metadata.
    
    Args:
        run_dir: Directory where all outputs will be saved
        config: Training configuration dictionary
        train_metrics: Final training metrics dict (loss, mae, etc.)
        val_metrics: Final validation metrics dict
        best_metrics: Best metrics across all epochs dict
        train_time_seconds: Total training time in seconds
        checkpoint_paths: Dict of saved model paths
    """
    print(f"\n{'='*60}")
    print("Generating training summary and plots...")
    print(f"{'='*60}\n")
    
    # Format training time
    hours = int(train_time_seconds // 3600)
    minutes = int((train_time_seconds % 3600) // 60)
    seconds = int(train_time_seconds % 60)
    time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Create summary dictionary
    summary = {
        "run_directory": run_dir,
        "timestamp": os.path.basename(run_dir).replace("run_", ""),
        "training_time_seconds": train_time_seconds,
        "training_time_formatted": time_formatted,
        "config": {
            "window_size": config.get('window_size'),
            "batch_size": config.get('batch_size'),
            "learning_rate": config.get('init_lr'),
            "num_epochs": config.get('num_epoch'),
            "model_architecture": config.get('model_architecture', 'vanilla_cnn'),
            "velocity_weight": config.get('velocity_weight', 1.0),
            "use_dense_supervision": config.get('use_dense_supervision', False),
            "velocity_loss": "gaussian_nll",
        },
        "final_metrics": train_metrics,
        "best_metrics": best_metrics,
        "saved_models": checkpoint_paths
    }
    
    # Save JSON summary
    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary (JSON) saved to: {summary_path}")
    
    # Save text summary
    summary_txt_path = os.path.join(run_dir, "training_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING RUN SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Run Directory: {run_dir}\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Training Time: {time_formatted}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-"*60 + "\n")
        for key, value in summary['config'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("FINAL METRICS (Last Epoch)\n")
        f.write("-"*60 + "\n")
        for key, value in train_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("BEST METRICS (Across All Epochs)\n")
        f.write("-"*60 + "\n")
        for key, value in best_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("SAVED MODEL CHECKPOINTS\n")
        f.write("-"*60 + "\n")
        for key, value in checkpoint_paths.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Training summary (text) saved to: {summary_txt_path}")
    
    # Generate plots from TensorBoard logs
    try:
        tb_log_dir = os.path.join(run_dir, "tensorboard")
        event_files = glob.glob(os.path.join(tb_log_dir, 'events.out.tfevents.*'))
        
        if not event_files:
            print("⚠ No TensorBoard event files found. Skipping plot generation.")
            return
        
        event_file = max(event_files, key=os.path.getmtime)
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Extract metrics
        train_loss_steps, train_loss = extract_scalars(ea, 'training/total_loss')
        val_loss_steps, val_loss = extract_scalars(ea, 'validation/total_loss')
        train_mae_steps, train_mae = extract_scalars(ea, 'training/velocity_mae')
        val_mae_steps, val_mae = extract_scalars(ea, 'validation/velocity_mae')
        
        # Plot loss
        if train_loss or val_loss:
            plt.figure(figsize=(10, 6))
            if train_loss:
                plt.plot(train_loss_steps, train_loss, 'g', label='Training loss', linewidth=2)
            if val_loss:
                plt.plot(val_loss_steps, val_loss, 'b', label='Validation loss', linewidth=2)
            plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            loss_plot_path = os.path.join(run_dir, 'training_validation_loss.png')
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Loss plot saved to: {loss_plot_path}")
            plt.close()
        
        # Plot MAE
        if train_mae or val_mae:
            plt.figure(figsize=(10, 6))
            if train_mae:
                plt.plot(train_mae_steps, train_mae, 'g', label='Training MAE', linewidth=2)
            if val_mae:
                plt.plot(val_mae_steps, val_mae, 'b', label='Validation MAE', linewidth=2)
            plt.title('Training vs Validation MAE (Velocity)', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Mean Absolute Error', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            mae_plot_path = os.path.join(run_dir, 'training_validation_mae.png')
            plt.savefig(mae_plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ MAE plot saved to: {mae_plot_path}")
            plt.close()
        
        
    except Exception as e:
        print(f"⚠ Warning: Could not generate plots: {e}")
        print(f"  You can manually run: python3 utils/plot_loss.py --log-dir {tb_log_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, 
                        default='logs/left_leg_contact',
                        help='Path to TensorBoard log directory')
    parser.add_argument('--output-dir', type=str,
                        default='results',
                        help='Directory to save plot images')
    parser.add_argument('--train-loss-tag', type=str, 
                        default='training/total_loss',
                        help='Tag name for training loss')
    parser.add_argument('--val-loss-tag', type=str, 
                        default='validation/total_loss',
                        help='Tag name for validation loss')
    parser.add_argument('--train-acc-tag', type=str, 
                        default='training/velocity_mae',
                        help='Tag name for training accuracy/MAE')
    parser.add_argument('--val-acc-tag', type=str, 
                        default='validation/velocity_mae',
                        help='Tag name for validation accuracy/MAE')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load TensorBoard logs
    ea = load_tensorboard_logs(args.log_dir)
    
    # Extract metrics
    train_loss_steps, train_loss = extract_scalars(ea, args.train_loss_tag)
    val_loss_steps, val_loss = extract_scalars(ea, args.val_loss_tag)
    train_acc_steps, train_acc = extract_scalars(ea, args.train_acc_tag)
    val_acc_steps, val_acc = extract_scalars(ea, args.val_acc_tag)
    
    # Plot loss
    if train_loss or val_loss:
        plt.figure(figsize=(10, 6))
        if train_loss:
            plt.plot(train_loss_steps, train_loss, 'g', label='Training loss')
        if val_loss:
            plt.plot(val_loss_steps, val_loss, 'b', label='Validation loss')
        plt.title('Training loss vs. Validation loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(args.output_dir, 'training_validation_loss.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to: {loss_plot_path}")
        plt.close()
    else:
        print("No loss data found. Check tag names.")
    
    # Plot accuracy/MAE
    if train_acc or val_acc:
        plt.figure(figsize=(10, 6))
        if train_acc:
            plt.plot(train_acc_steps, train_acc, 'g', label='Training MAE')
        if val_acc:
            plt.plot(val_acc_steps, val_acc, 'b', label='Validation MAE')
        plt.title('Training MAE vs. Validation MAE')
        plt.xlabel('Steps')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        mae_plot_path = os.path.join(args.output_dir, 'training_validation_mae.png')
        plt.savefig(mae_plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved MAE plot to: {mae_plot_path}")
        plt.close()
    else:
        print("No MAE data found. Check tag names.")


if __name__ == '__main__':
    main()
