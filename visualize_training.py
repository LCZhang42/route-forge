"""
Visualize training performance from TensorBoard logs and checkpoints.

This script reads TensorBoard event files and creates visualizations of:
- Training and validation loss over time
- Learning rate schedule
- Checkpoint comparison
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import torch
import seaborn as sns

sns.set_style("whitegrid")


def load_tensorboard_logs(log_dir):
    """Load training metrics from TensorBoard logs."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        raise ValueError(f"Log directory not found: {log_dir}")
    
    # Find event files
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(log_path))
    ea.Reload()
    
    # Extract metrics
    metrics = {}
    
    # Get available tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available metrics: {scalar_tags}")
    
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = {'steps': steps, 'values': values}
    
    return metrics


def load_checkpoint_info(checkpoint_dir):
    """Load information from saved checkpoints."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoints = []
    
    # Find all epoch checkpoints
    for ckpt_file in sorted(checkpoint_path.glob("epoch_*.pt")):
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            checkpoints.append({
                'file': ckpt_file.name,
                'epoch': ckpt['epoch'],
                'val_loss': ckpt['val_loss'],
                'step': ckpt['current_step'],
            })
        except Exception as e:
            print(f"Warning: Could not load {ckpt_file}: {e}")
    
    # Load best checkpoint
    best_path = checkpoint_path / 'best.pt'
    if best_path.exists():
        try:
            best_ckpt = torch.load(best_path, map_location='cpu')
            print(f"\nBest checkpoint: Epoch {best_ckpt['epoch']}, Val Loss: {best_ckpt['val_loss']:.4f}")
        except Exception as e:
            print(f"Warning: Could not load best checkpoint: {e}")
    
    return checkpoints


def plot_training_curves(metrics, output_dir):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training loss over steps
    if 'Train/Loss' in metrics:
        train_data = metrics['Train/Loss']
        axes[0].plot(train_data['steps'], train_data['values'], 
                    label='Training Loss', alpha=0.7, linewidth=1)
        
        # Add smoothed line
        window = min(100, len(train_data['values']) // 10)
        if window > 1:
            smoothed = np.convolve(train_data['values'], 
                                  np.ones(window)/window, mode='valid')
            axes[0].plot(train_data['steps'][window-1:], smoothed, 
                        label='Smoothed', linewidth=2, color='darkblue')
    
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Validation loss over epochs
    if 'Val/Loss' in metrics:
        val_data = metrics['Val/Loss']
        axes[1].plot(val_data['steps'], val_data['values'], 
                    marker='o', label='Validation Loss', 
                    linewidth=2, markersize=6, color='orange')
        
        # Mark minimum
        min_idx = np.argmin(val_data['values'])
        axes[1].axhline(y=val_data['values'][min_idx], 
                       color='red', linestyle='--', alpha=0.5,
                       label=f'Best: {val_data["values"][min_idx]:.4f}')
        axes[1].scatter([val_data['steps'][min_idx]], 
                       [val_data['values'][min_idx]], 
                       color='red', s=100, zorder=5)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss Over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


def plot_learning_rate(metrics, output_dir):
    """Plot learning rate schedule."""
    if 'Train/LearningRate' not in metrics:
        print("Learning rate data not found in logs")
        return
    
    lr_data = metrics['Train/LearningRate']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lr_data['steps'], lr_data['values'], linewidth=2, color='green')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'learning_rate.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning rate plot to {output_path}")
    plt.close()


def plot_checkpoint_comparison(checkpoints, output_dir):
    """Plot comparison of checkpoint performance."""
    if not checkpoints:
        print("No checkpoint data available")
        return
    
    epochs = [c['epoch'] for c in checkpoints]
    val_losses = [c['val_loss'] for c in checkpoints]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(epochs, val_losses, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Checkpoint Performance (Every 10 Epochs)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for epoch, loss in zip(epochs, val_losses):
        ax.text(epoch, loss, f'{loss:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'checkpoint_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved checkpoint comparison to {output_path}")
    plt.close()


def print_summary(metrics, checkpoints):
    """Print training summary statistics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if 'Train/Loss' in metrics:
        train_losses = metrics['Train/Loss']['values']
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final:   {train_losses[-1]:.4f}")
        print(f"  Min:     {min(train_losses):.4f}")
        print(f"  Total steps: {len(train_losses)}")
    
    if 'Val/Loss' in metrics:
        val_losses = metrics['Val/Loss']['values']
        val_epochs = metrics['Val/Loss']['steps']
        min_idx = np.argmin(val_losses)
        print(f"\nValidation Loss:")
        print(f"  Initial: {val_losses[0]:.4f}")
        print(f"  Final:   {val_losses[-1]:.4f}")
        print(f"  Best:    {val_losses[min_idx]:.4f} (Epoch {val_epochs[min_idx]})")
        print(f"  Total epochs: {len(val_losses)}")
    
    if checkpoints:
        print(f"\nCheckpoints saved: {len(checkpoints)}")
        print("Epochs:", [c['epoch'] for c in checkpoints])
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training performance from TensorBoard logs'
    )
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='runs/climb_path_cpu',
        help='Path to TensorBoard log directory'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/climb_path_cpu',
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='training_plots',
        help='Directory to save visualization plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading training metrics...")
    try:
        metrics = load_tensorboard_logs(args.log_dir)
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")
        return
    
    print("\nLoading checkpoint information...")
    checkpoints = load_checkpoint_info(args.checkpoint_dir)
    
    print("\nGenerating visualizations...")
    plot_training_curves(metrics, args.output_dir)
    plot_learning_rate(metrics, args.output_dir)
    
    if checkpoints:
        plot_checkpoint_comparison(checkpoints, args.output_dir)
    
    print_summary(metrics, checkpoints)
    
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
