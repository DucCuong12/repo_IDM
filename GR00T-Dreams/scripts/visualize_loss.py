#!/usr/bin/env python3
"""
Script to visualize training loss from CSV log file.
Usage:
    python scripts/visualize_loss.py --log_file <path_to_csv> [--output <output_image>]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_loss(log_file: str, output_file: str = None, show: bool = True):
    """Plot training loss from CSV log file."""
    
    # Read CSV file
    df = pd.read_csv(log_file)
    
    # Filter out rows where loss is 'N/A' or NaN
    df = df[df['loss'] != 'N/A']
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df = df.dropna(subset=['loss', 'step'])
    
    if len(df) == 0:
        print("No valid loss data found in log file!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IDM Training Progress', fontsize=14, fontweight='bold')
    
    # Plot 1: Loss vs Step
    ax1 = axes[0, 0]
    ax1.plot(df['step'], df['loss'], 'b-', alpha=0.7, linewidth=0.8, label='Loss')
    # Add smoothed loss (rolling average)
    window_size = min(50, len(df) // 10 + 1)
    if window_size > 1:
        df['loss_smooth'] = df['loss'].rolling(window=window_size, center=True).mean()
        ax1.plot(df['step'], df['loss_smooth'], 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log Loss vs Step
    ax2 = axes[0, 1]
    ax2.semilogy(df['step'], df['loss'], 'b-', alpha=0.7, linewidth=0.8)
    if 'loss_smooth' in df.columns:
        ax2.semilogy(df['step'], df['loss_smooth'], 'r-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate vs Step (if available)
    ax3 = axes[1, 0]
    if 'learning_rate' in df.columns:
        df_lr = df[df['learning_rate'] != 'N/A'].copy()
        df_lr['learning_rate'] = pd.to_numeric(df_lr['learning_rate'], errors='coerce')
        df_lr = df_lr.dropna(subset=['learning_rate'])
        if len(df_lr) > 0:
            ax3.plot(df_lr['step'], df_lr['learning_rate'], 'g-', linewidth=1.5)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            ax3.text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = f"""
Training Statistics
{'='*40}

Total Steps: {int(df['step'].max())}
Data Points: {len(df)}

Loss Statistics:
  - Initial: {df['loss'].iloc[0]:.6f}
  - Final: {df['loss'].iloc[-1]:.6f}
  - Minimum: {df['loss'].min():.6f} (step {int(df.loc[df['loss'].idxmin(), 'step'])})
  - Maximum: {df['loss'].max():.6f}
  - Mean: {df['loss'].mean():.6f}
  - Std Dev: {df['loss'].std():.6f}

Improvement:
  - Absolute: {df['loss'].iloc[0] - df['loss'].iloc[-1]:.6f}
  - Relative: {((df['loss'].iloc[0] - df['loss'].iloc[-1]) / df['loss'].iloc[0] * 100):.2f}%
"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    if show:
        plt.show()
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Visualize IDM training loss')
    parser.add_argument('--log_file', type=str, required=True,
                        help='Path to the training_loss.csv file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file path (optional)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (only save)')
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return
    
    # Default output path if not specified
    if args.output is None:
        args.output = str(log_file.parent / 'training_loss_plot.png')
    
    plot_training_loss(
        log_file=str(log_file),
        output_file=args.output,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()
