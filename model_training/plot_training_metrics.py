"""
Plot training metrics from a saved training_metrics.pkl file.
This script loads the metrics saved by run_training_sgd1.py and generates diagnostic plots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse

def plot_training_metrics(metrics, args, output_filename=None):
    """Plot training diagnostics from train_stats.
    
    Args:
        metrics: Dictionary with training metrics
        args: Training arguments/config
        output_filename: Base filename for output (defaults to 'training_diagnostics')
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Use custom filename if provided, otherwise default
    if output_filename is None:
        output_filename = 'training_diagnostics'
    
    train_losses = np.array(metrics['train_losses'])
    val_losses = np.array(metrics['val_losses'])
    val_PERs = np.array(metrics['val_PERs'])
    
    # Calculate batch numbers for validation steps
    batches_per_val_step = args['batches_per_val_step']
    val_batches = np.arange(0, len(val_PERs)) * batches_per_val_step
    
    # train_losses has one entry per training batch
    # Downsample for plotting if there are too many points
    train_batches = np.arange(len(train_losses))
    if len(train_losses) > 1000:
        # Downsample to ~1000 points for faster plotting
        step = len(train_losses) // 1000
        train_batches = train_batches[::step]
        train_losses = train_losses[::step]
    
    # Create figure with 2x2 subplots (4 plots total)
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Train Loss vs Val Loss (overfitting detection)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(train_batches, train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(val_batches, val_losses, 'r-', label='Val Loss', alpha=0.7, linewidth=2, marker='o', markersize=6)
    ax1.set_xlabel('Training Batch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11, color='b')
    ax1_twin.set_ylabel('Val Loss', fontsize=11, color='r')
    ax1.set_title('Train Loss vs Val Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_yscale('log')
    
    # 2. Validation PER over time (main metric) - NORMAL axis (decreasing = improving)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(val_batches, val_PERs, 'g-', linewidth=2, marker='o', markersize=6)
    best_per = val_PERs.min()
    best_idx = np.argmin(val_PERs)
    ax2.axhline(y=best_per, color='r', linestyle='--', alpha=0.5, 
                label=f'Best PER: {best_per:.4f}')
    ax2.plot(val_batches[best_idx], val_PERs[best_idx], 'ro', markersize=10)
    ax2.set_xlabel('Training Batch', fontsize=11)
    ax2.set_ylabel('Validation PER', fontsize=11)
    ax2.set_title('Validation PER Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Train/Val Loss Ratio (overfitting metric)
    # Interpolate val losses to match train batch times
    val_losses_interp = np.interp(train_batches, val_batches, val_losses)
    train_val_ratio = train_losses / (val_losses_interp + 1e-8)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(train_batches, train_val_ratio, 'orange', linewidth=2, alpha=0.7)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ratio = 1.0')
    ax3.set_xlabel('Training Batch', fontsize=11)
    ax3.set_ylabel('Train Loss / Val Loss', fontsize=11)
    ax3.set_title('Train/Val Loss Ratio', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    if len(train_val_ratio) > 0:
        ax3.set_ylim([0, max(2, train_val_ratio.max() * 1.1)])
    
    # 4. Learning Rate Schedule
    ax4 = plt.subplot(2, 2, 4)
    lr_max = args['lr_max']
    lr_min = args['lr_min']
    lr_decay_steps = args['lr_decay_steps']
    lr_warmup_steps = args.get('lr_warmup_steps', 0)
    scheduler_type = args.get('lr_scheduler_type', 'cosine')
    
    def compute_lr(step, sched_type):
        """Compute learning rate based on scheduler type."""
        # Warmup phase (if warmup_steps > 0)
        if lr_warmup_steps > 0 and step < lr_warmup_steps:
            return lr_max * (step / lr_warmup_steps)
        
        # Decay phase (after warmup)
        if step < lr_decay_steps:
            progress = (step - lr_warmup_steps) / max(1, lr_decay_steps - lr_warmup_steps)
            
            if sched_type == 'cosine':
                # Cosine decay: smooth curve from 1.0 to min_lr_ratio
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                min_lr_ratio = lr_min / lr_max
                return lr_min + (lr_max - lr_min) * cosine_decay
            
            elif sched_type == 'linear':
                # Linear decay: straight line from lr_max to lr_min
                return lr_max - (lr_max - lr_min) * progress
            
            elif sched_type == 'exponential':
                # Exponential decay: exponential curve
                decay_rate = lr_min / lr_max
                return lr_max * (decay_rate ** progress)
            
            elif sched_type == 'step':
                # Step decay: reduce by factor at milestones (simplified - assumes single step)
                # For true step decay, you'd need milestone info from args
                step_size = lr_decay_steps / 3  # Assume 3 steps
                gamma = 0.5  # Reduce by half each step
                steps = int(progress * 3)
                return lr_max * (gamma ** steps)
            
            else:
                # Unknown scheduler - assume constant after warmup
                return lr_max
        else:
            # After decay_steps, maintain minimum
            return lr_min
    
    max_batch = max(train_batches.max() if len(train_batches) > 0 else 10000, 
                    val_batches.max() if len(val_batches) > 0 else 10000)
    all_batches = np.arange(0, max_batch + 1, 100)
    lrs = [compute_lr(b, scheduler_type) for b in all_batches]
    ax4.plot(all_batches, lrs, 'b-', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Training Batch', fontsize=11)
    ax4.set_ylabel('Learning Rate', fontsize=11)
    
    # Update title based on scheduler type
    scheduler_names = {
        'cosine': 'Cosine Decay',
        'linear': 'Linear Decay',
        'exponential': 'Exponential Decay',
        'step': 'Step Decay',
        'constant': 'Constant (No Decay)'
    }
    title = f"Learning Rate Schedule ({scheduler_names.get(scheduler_type, scheduler_type.title())})"
    ax4.set_title(title, fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure (will overwrite if exists)
    output_file = Path(output_dir) / f'{output_filename}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved diagnostic plots to: {output_file}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from saved pickle file')
    parser.add_argument('--metrics_file', type=str, 
                       help='Path to training_metrics.pkl file')
    parser.add_argument('--modelname', type=str, default='training_diagnostics',
                       help='Base name for output files (default: training_diagnostics)')
    
    args_cmd = parser.parse_args()
    
    # Find metrics file
    if args_cmd.metrics_file:
        metrics_file = Path(args_cmd.metrics_file)
    else:
        print("Error: Must provide --metrics_file")
        exit(1)
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        exit(1)
    
    # Load metrics
    with open(metrics_file, 'rb') as f:
        data = pickle.load(f)
    
    metrics = data['metrics']
    args = data['args']
    
    print(f"Loaded metrics:")
    print(f"  Training batches: {len(metrics['train_losses'])}")
    print(f"  Validation steps: {len(metrics['val_PERs'])}")
    
    # Check if best validation metrics exist
    checkpoint_dir = Path(metrics_file.parent) / 'checkpoint'
    best_val_metrics_file = checkpoint_dir / 'val_metrics.pkl'
    if best_val_metrics_file.exists():
        print(f"\n  Note: Best validation metrics available at: {best_val_metrics_file}")
        print(f"        (Contains detailed predictions from best validation step)")
    
    print(f"\nGenerating plots with modelname: {args_cmd.modelname}")
    print(f"Output will be saved to: {args_cmd.modelname}.png")
    
    plot_training_metrics(metrics, args, output_filename=args_cmd.modelname)
    print("\nPlots generated successfully!")
#example of how you use it
# !cd /kaggle/working/nejm-brain-to-text/model_training/ && \
# python plot_training_metrics.py --metrics_file trained_models/minimal_baseline/training_metrics.pkl --modelname baseline_adamW
