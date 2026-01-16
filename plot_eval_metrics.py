import json
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

def plot_eval_metrics(json_dir):
    """Read JSON eval results and plot metrics across epochs."""

    # Read all JSON files
    json_files = sorted(Path(json_dir).glob("e2e_epoch*.json"))

    # Extract epoch numbers and metrics
    epochs = []
    metrics_data = {}

    for json_file in json_files:
        # Extract epoch number from filename
        match = re.search(r'epoch(\d+)', json_file.name)
        if not match:
            continue
        epoch = int(match.group(1))
        epochs.append(epoch)

        # Read metrics
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Store each metric
        for key, value in data.items():
            if key not in metrics_data:
                metrics_data[key] = []
            metrics_data[key].append(value)

    # Sort by epoch
    sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in sorted_indices]
    for key in metrics_data:
        metrics_data[key] = [metrics_data[key][i] for i in sorted_indices]

    # Create subplots
    num_metrics = len(metrics_data)
    cols = 3
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    # Plot each metric
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        ax.plot(epochs, values, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path(json_dir).parent / 'eval_metrics_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    plt.show()

if __name__ == "__main__":
    json_dir = "logs/e2e/glim_stage2_e2e_mtv_20260115_195030/json"
    plot_eval_metrics(json_dir)
