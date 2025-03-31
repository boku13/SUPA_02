import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Define file paths
csv_file_path = 'results/enhanced_pipeline/evaluation_results.csv'
output_dir_path = Path('results/enhanced_pipeline/')

# Ensure the output directory exists
output_dir_path.mkdir(parents=True, exist_ok=True)

# --- Read the CSV file ---
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data from {csv_file_path}")
    # Drop rows with NaN values that would break plotting
    df.dropna(subset=['SDR', 'SIR', 'SAR', 'PESQ'], inplace=True)
    if df.empty:
        print("Error: No valid data rows remain after dropping NaNs. Cannot generate plots.")
        exit()
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Select the metrics columns
metrics = ['SDR', 'SIR', 'SAR', 'PESQ']
if not all(metric in df.columns for metric in metrics):
    print(f"Error: CSV file must contain the columns: {', '.join(metrics)}")
    missing_cols = [m for m in metrics if m not in df.columns]
    print(f"Missing columns: {', '.join(missing_cols)}")
    exit()

# --- 1. Generate Histograms ---
output_hist_path = output_dir_path / 'enhancement_histograms.png'
fig_hist, axes_hist = plt.subplots(2, 2, figsize=(12, 10))
fig_hist.suptitle('Distribution of Enhancement Metrics', fontsize=16)
axes_hist = axes_hist.flatten()

for i, metric in enumerate(metrics):
    axes_hist[i].hist(df[metric], bins=20, edgecolor='black')
    axes_hist[i].set_title(f'{metric} Distribution')
    axes_hist[i].set_xlabel(metric)
    axes_hist[i].set_ylabel('Frequency')
    axes_hist[i].grid(axis='y', alpha=0.75)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
try:
    plt.savefig(output_hist_path)
    print(f"Histograms saved to '{output_hist_path}'")
except Exception as e:
    print(f"Error saving histograms: {e}")
plt.close(fig_hist)

# --- 2. Generate Scatter Plots ---
output_scatter_path = output_dir_path / 'enhancement_scatter_plots.png'
scatter_pairs = [('SDR', 'SIR'), ('SDR', 'SAR'), ('SDR', 'PESQ'), ('SIR', 'PESQ')]
fig_scatter, axes_scatter = plt.subplots(2, 2, figsize=(12, 10))
fig_scatter.suptitle('Scatter Plots of Enhancement Metrics', fontsize=16)
axes_scatter = axes_scatter.flatten()

for i, (x_metric, y_metric) in enumerate(scatter_pairs):
    axes_scatter[i].scatter(df[x_metric], df[y_metric], alpha=0.5)
    axes_scatter[i].set_title(f'{x_metric} vs. {y_metric}')
    axes_scatter[i].set_xlabel(x_metric)
    axes_scatter[i].set_ylabel(y_metric)
    axes_scatter[i].grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
try:
    plt.savefig(output_scatter_path)
    print(f"Scatter plots saved to '{output_scatter_path}'")
except Exception as e:
    print(f"Error saving scatter plots: {e}")
plt.close(fig_scatter)

# --- 3. Generate Pair Plot ---
output_pair_path = output_dir_path / 'enhancement_pair_plot.png'
try:
    pair_plot = sns.pairplot(df[metrics])
    pair_plot.fig.suptitle('Pair Plot of Enhancement Metrics', y=1.02) # Adjust title position
    pair_plot.savefig(output_pair_path)
    print(f"Pair plot saved to '{output_pair_path}'")
except Exception as e:
    print(f"Error saving pair plot: {e}")

print("\nVisualization complete.")

# Optionally, display the plots if needed (uncomment)
# plt.show() 