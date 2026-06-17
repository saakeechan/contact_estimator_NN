#!/usr/bin/env python3
"""
Analyze ground truth foot velocities from CSV files.

For each CSV file:
1. Load the data
2. Split by runs (when timestamp resets)
3. For each run:
   - Compute left foot velocity in world frame by numerical differentiation
   - Calculate velocity magnitude (norm)
   - Plot velocity over time
   - Save plot as PNG
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ========================================
# CONFIGURATION FLAGS
# ========================================
# Set to False to skip foot velocity plotting
PLOT_FOOT_VELOCITIES = True

# Set to True to analyze command velocities vs frequency
ANALYZE_COMMAND_VELOCITIES = False

# Command velocity range filter [min, max] in m/s
# Only runs within this range will be processed when using "all"
CMD_VEL_RANGE = [0.0, 100]


# ========================================
# MANUALLY SPECIFY CSV FILES TO ANALYZE
# ========================================
# Use "all" to process all CSV files in the directory
# Or specify individual filenames
CSV_FILES_TO_ANALYZE = [
    "robotstate_0_env1.csv",
    # Add more CSV filenames here as needed
]


def analyze_foot_velocities(csv_dir="Data/CSVFiles", output_dir=None):
    """
    Analyze foot velocities from CSV files and generate plots.
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory to save velocity plots (auto-generated if None)
    """
    # Auto-generate output directory name with cmd_vel range
    if output_dir is None:
        output_dir = f"footVelocitiesPlots_cmdvel{CMD_VEL_RANGE[0]:.1f}-{CMD_VEL_RANGE[1]:.1f}"
    
    # Create output directory if plotting is enabled
    if PLOT_FOOT_VELOCITIES:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if "all" is in the list
    if "all" in CSV_FILES_TO_ANALYZE:
        # Get all CSV files in the directory
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
        # Randomly sample 10 files (or fewer if less than 10 available)
        if len(csv_files) > 5:
            np.random.seed(42)  # For reproducibility
            csv_files = list(np.random.choice(csv_files, size=10, replace=False))
            csv_files.sort()  # Sort for consistent ordering
            print(f"Randomly sampled 10 files from {len(glob.glob(os.path.join(csv_dir, '*.csv')))} total")
    else:
        # Build full paths from the manual list
        csv_files = [os.path.join(csv_dir, fname) for fname in CSV_FILES_TO_ANALYZE]
        # Filter out files that don't exist
        csv_files = [f for f in csv_files if os.path.exists(f)]
    
    if not csv_files:
        print(f"No valid CSV files found")
        print(f"Directory checked: {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Foot velocity plotting: {'ON' if PLOT_FOOT_VELOCITIES else 'OFF'}")
    print(f"Command velocity analysis: {'ON' if ANALYZE_COMMAND_VELOCITIES else 'OFF'}")
    
    # Global run counter across all files
    global_run_number = 0
    
    # For command velocity analysis - list of cmd_vel_x values from each run
    cmd_vel_data = []
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        
        # Load CSV data
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_cols = ['timestamp', 'lfoot_pos_x', 'lfoot_pos_y', 'lfoot_pos_z']
        if not all(col in df.columns for col in required_cols):
            print(f"  Skipping - missing required columns")
            continue
        
        # Check if contact column exists
        has_contact = 'lfoot-contact' in df.columns
        
        # Split by runs - detect when timestamp resets
        run_boundaries = [0]
        timestamps = df['timestamp'].values
        
        for i in range(1, len(timestamps)):
            dt = abs(timestamps[i] - timestamps[i-1])
            # Time reset indicates new run
            if dt > 0.025:  # 25ms threshold
                run_boundaries.append(i)
        
        run_boundaries.append(len(df))
        
        num_runs = len(run_boundaries) - 1
        print(f"  Found {num_runs} runs")
        
        # Process each run
        for run_idx in range(num_runs):
            start_idx = run_boundaries[run_idx]
            end_idx = run_boundaries[run_idx + 1]
            
            # Extract run data
            df_run = df.iloc[start_idx:end_idx]
            
            # Skip short runs
            if len(df_run) < 2:
                print(f"    Run {run_idx}: Too short, skipping")
                continue
            
            # Extract left foot positions
            lfoot_x = df_run['lfoot_pos_x'].values
            lfoot_y = df_run['lfoot_pos_y'].values
            lfoot_z = df_run['lfoot_pos_z'].values
            time = df_run['timestamp'].values
            
            # Extract contact state if available
            contact_state = df_run['lfoot-contact'].values if has_contact else None
            
            # Extract cmd_vel_x if available
            cmd_vel_x = df_run['cmd_vel_x'].values[0] if 'cmd_vel_x' in df_run.columns else None
            
            # Filter out runs with cmd_vel_x outside the configured range
            if cmd_vel_x is not None:
                if not (CMD_VEL_RANGE[0] <= cmd_vel_x <= CMD_VEL_RANGE[1]):
                    print(f"    Run {run_idx}: Skipping (cmd_vel_x={cmd_vel_x:.3f} not in [{CMD_VEL_RANGE[0]}, {CMD_VEL_RANGE[1]}])")
                    continue
            
            # Store command velocity for frequency analysis
            if ANALYZE_COMMAND_VELOCITIES and cmd_vel_x is not None:
                cmd_vel_data.append(cmd_vel_x)
            
            # Only compute velocities and plot if foot velocity plotting is enabled
            if PLOT_FOOT_VELOCITIES:
                # Compute velocities by numerical differentiation
                # Use forward differences for all except last point
                vel_x = np.diff(lfoot_x) / np.diff(time)
                vel_y = np.diff(lfoot_y) / np.diff(time)
                vel_z = np.diff(lfoot_z) / np.diff(time)
                
                # Pad to match original length (repeat last velocity)
                vel_x = np.append(vel_x, vel_x[-1])
                vel_y = np.append(vel_y, vel_y[-1])
                vel_z = np.append(vel_z, vel_z[-1])
                
                # Compute velocity magnitude (norm)
                vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                
                # Create plot
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                # Add contact regions as red background on both plots
                if contact_state is not None:
                    for ax in axes:
                        # Find contact regions
                        in_contact = contact_state > 0.5
                        # Create shaded regions where foot is in contact
                        contact_changes = np.diff(np.concatenate(([0], in_contact, [0])))
                        contact_starts = np.where(contact_changes == 1)[0]
                        contact_ends = np.where(contact_changes == -1)[0]
                        
                        for start, end in zip(contact_starts, contact_ends):
                            ax.axvspan(time[start], time[min(end, len(time)-1)], 
                                      alpha=0.3, color='red', label='Contact' if start == contact_starts[0] else '')
                
                # Plot individual components
                axes[0].plot(time, vel_x, label='vel_x', alpha=0.7)
                axes[0].plot(time, vel_y, label='vel_y', alpha=0.7)
                axes[0].plot(time, vel_z, label='vel_z', alpha=0.7)
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Velocity (m/s)')
                title_text = f'Left Foot Velocity Components - Run {global_run_number}'
                if cmd_vel_x is not None:
                    title_text += f' | cmd_vel_x: {cmd_vel_x:.3f} m/s'
                axes[0].set_title(title_text)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot magnitude
                axes[1].plot(time, vel_magnitude, 'b-', linewidth=2)
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('Velocity Magnitude (m/s)')
                title_text = f'Left Foot Velocity Magnitude - Run {global_run_number}'
                if cmd_vel_x is not None:
                    title_text += f' | cmd_vel_x: {cmd_vel_x:.3f} m/s'
                axes[1].set_title(title_text)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                cmd_vel_filename = f"_cmdvel{cmd_vel_x:.3f}" if cmd_vel_x is not None else ""
                output_path = os.path.join(output_dir, f"run_{global_run_number}{cmd_vel_filename}_velocity.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                cmd_vel_str = f", cmd_vel_x={cmd_vel_x:.3f} m/s" if cmd_vel_x is not None else ""
                print(f"    Run {run_idx} -> Global run {global_run_number}: "
                      f"{len(df_run)} samples{cmd_vel_str}")
            else:
                # Just print basic info without plotting
                cmd_vel_str = f", cmd_vel_x={cmd_vel_x:.3f} m/s" if cmd_vel_x is not None else ""
                print(f"    Run {run_idx} -> Global run {global_run_number}: "
                      f"{len(df_run)} samples{cmd_vel_str}")
            
            global_run_number += 1
    
    print(f"\nDone! Processed {global_run_number} runs total.")
    if PLOT_FOOT_VELOCITIES:
        print(f"Foot velocity plots saved to: {output_dir}/")
    
    # Create command velocity analysis plot
    if ANALYZE_COMMAND_VELOCITIES and len(cmd_vel_data) > 0:
        print(f"\nCommand velocity analysis: {len(cmd_vel_data)} runs with valid data")
        
        # Count frequency of occurrence for each unique command velocity
        cmd_vel_array = np.array(cmd_vel_data)
        
        # Create histogram with 0.3 m/s bins
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate bins: every 0.3 m/s
        bin_width = 0.3
        min_vel = np.floor(np.min(cmd_vel_array) / bin_width) * bin_width
        max_vel = np.ceil(np.max(cmd_vel_array) / bin_width) * bin_width
        bins = np.arange(min_vel, max_vel + bin_width, bin_width)
        
        ax.hist(cmd_vel_array, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Command Velocity X (m/s)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Command Velocity Distribution', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add some statistics text
        stats_text = f'Total runs: {len(cmd_vel_data)}\n'
        stats_text += f'Bin width: {bin_width} m/s\n'
        stats_text += f'Range: [{np.min(cmd_vel_array):.3f}, {np.max(cmd_vel_array):.3f}] m/s'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot in utils folder with cmd_vel range in filename
        cmd_vel_output_dir = "utils/cmd_vel_analysis"
        os.makedirs(cmd_vel_output_dir, exist_ok=True)
        histogram_filename = f"cmd_vel_histogram_{CMD_VEL_RANGE[0]:.1f}-{CMD_VEL_RANGE[1]:.1f}.png"
        cmd_vel_plot_path = os.path.join(cmd_vel_output_dir, histogram_filename)
        plt.savefig(cmd_vel_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Command velocity histogram saved to: {cmd_vel_plot_path}")
        
        # Print bin counts
        counts, bin_edges = np.histogram(cmd_vel_array, bins=bins)
        print("\nCommand Velocity Histogram:")
        for i, count in enumerate(counts):
            if count > 0:
                print(f"  [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}) m/s: {count} runs")
    elif ANALYZE_COMMAND_VELOCITIES:
        print("\nWarning: Command velocity analysis enabled but no valid data found")
        print("Make sure CSV files have 'cmd_vel_x' column")


if __name__ == "__main__":
    analyze_foot_velocities()