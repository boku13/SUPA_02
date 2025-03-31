import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np

# Define file paths
csv_file_path = 'results/speaker_separation/evaluation_results.csv'
output_dir_path = Path('results/speaker_separation/')

# Ensure the output directory exists
output_dir_path.mkdir(parents=True, exist_ok=True)

# --- Read the CSV file ---
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data from {csv_file_path}")
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Identify numeric metrics columns automatically
metrics = df.select_dtypes(include=np.number).columns.tolist()
# Remove mixture_id if it was parsed as numeric, though unlikely based on format
if 'mixture_id' in metrics:
    metrics.remove('mixture_id') 

print(f"Identified metrics for plotting: {metrics}")

# Drop rows with NaN in essential metrics to avoid plotting errors
# (Allow NaNs in finetuned_speaker_acc if it's not present everywhere)
essential_metrics = [m for m in metrics if m != 'finetuned_speaker_acc']
df_clean = df.dropna(subset=essential_metrics).copy()

if df_clean.empty:
    print("Error: No valid data rows remain after dropping NaNs in essential metrics. Cannot generate plots.")
    exit()

# --- 1. Generate Histograms ---
num_metrics = len(metrics)
n_cols = 3
n_rows = (num_metrics + n_cols - 1) // n_cols # Calculate rows needed

output_hist_path = output_dir_path / 'separation_histograms.png'
fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
fig_hist.suptitle('Distribution of Separation Metrics', fontsize=16)
axes_hist = axes_hist.flatten()

for i, metric in enumerate(metrics):
    if metric in df_clean.columns:
        # Only plot if column exists and has data after cleaning
        if df_clean[metric].notna().any(): 
            axes_hist[i].hist(df_clean[metric].dropna(), bins=20, edgecolor='black') # Drop NaNs for this specific metric
            axes_hist[i].set_title(f'{metric} Distribution')
            axes_hist[i].set_xlabel(metric)
            axes_hist[i].set_ylabel('Frequency')
            axes_hist[i].grid(axis='y', alpha=0.75)
        else:
            axes_hist[i].set_title(f'{metric} (No Data)')
            axes_hist[i].text(0.5, 0.5, 'No Data', ha='center', va='center')

# Hide unused subplots
for j in range(i + 1, len(axes_hist)):
    fig_hist.delaxes(axes_hist[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
try:
    plt.savefig(output_hist_path)
    print(f"Histograms saved to '{output_hist_path}'")
except Exception as e:
    print(f"Error saving histograms: {e}")
plt.close(fig_hist)

# --- 2. Generate Scatter Plots ---
output_scatter_path = output_dir_path / 'separation_scatter_plots.png'
scatter_pairs = [
    ('SDR', 'SIR'), ('SDR', 'SAR'), ('SDR', 'PESQ'), 
    ('PESQ', 'pretrained_speaker_acc'), ('SDR', 'pretrained_speaker_acc')
]
# Add finetuned plots if the column exists
if 'finetuned_speaker_acc' in df_clean.columns and df_clean['finetuned_speaker_acc'].notna().any():
    scatter_pairs.extend([('PESQ', 'finetuned_speaker_acc'), ('SDR', 'finetuned_speaker_acc')])

num_scatter = len(scatter_pairs)
n_scatter_cols = 3
n_scatter_rows = (num_scatter + n_scatter_cols - 1) // n_scatter_cols

fig_scatter, axes_scatter = plt.subplots(n_scatter_rows, n_scatter_cols, figsize=(5 * n_scatter_cols, 4 * n_scatter_rows))
fig_scatter.suptitle('Scatter Plots of Separation Metrics', fontsize=16)
axes_scatter = axes_scatter.flatten()

scatter_plot_count = 0
for i, (x_metric, y_metric) in enumerate(scatter_pairs):
    if x_metric in df_clean.columns and y_metric in df_clean.columns:
        # Plot only if both columns exist
        plot_df = df_clean[[x_metric, y_metric]].dropna()
        if not plot_df.empty:
            ax = axes_scatter[scatter_plot_count]
            ax.scatter(plot_df[x_metric], plot_df[y_metric], alpha=0.5)
            ax.set_title(f'{x_metric} vs. {y_metric}')
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.grid(True, alpha=0.5)
            scatter_plot_count += 1
        else:
             print(f"Skipping scatter plot for ({x_metric}, {y_metric}) due to missing data after NaN drop.")

# Hide unused scatter subplots
for j in range(scatter_plot_count, len(axes_scatter)):
    fig_scatter.delaxes(axes_scatter[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
try:
    plt.savefig(output_scatter_path)
    print(f"Scatter plots saved to '{output_scatter_path}'")
except Exception as e:
    print(f"Error saving scatter plots: {e}")
plt.close(fig_scatter)

# --- 3. Generate Pair Plot ---
output_pair_path = output_dir_path / 'separation_pair_plot.png'
try:
    # Use only columns with actual data for pairplot
    plot_metrics = [m for m in metrics if m in df_clean.columns and df_clean[m].notna().any()]
    if plot_metrics:
        pair_plot = sns.pairplot(df_clean[plot_metrics], corner=True) # Use corner=True for efficiency
        pair_plot.fig.suptitle('Pair Plot of Separation Metrics', y=1.02) # Adjust title position
        pair_plot.savefig(output_pair_path)
        print(f"Pair plot saved to '{output_pair_path}'")
    else:
        print("Skipping pair plot as no metrics have data after cleaning.")
except Exception as e:
    print(f"Error saving pair plot: {e}")

print("\nVisualization complete.")

# Optionally, display the plots if needed (uncomment)
# plt.show() 