import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
csv_file_path = 'results/enhanced_pipeline/evaluation_results.csv'
output_dir = 'results/enhanced_pipeline/'
output_image_path = os.path.join(output_dir, 'metrics_histograms.png')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
try:
    df = pd.read_csv(csv_file_path)
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

# Create histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Evaluation Metrics', fontsize=16)

axes = axes.flatten() # Flatten the 2x2 array for easy iteration

for i, metric in enumerate(metrics):
    ax = axes[i]
    ax.hist(df[metric], bins=20, edgecolor='black')
    ax.set_title(f'{metric} Distribution')
    ax.set_xlabel(metric)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.75)

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
try:
    plt.savefig(output_image_path)
    print(f"Histograms saved to '{output_image_path}'")
except Exception as e:
    print(f"Error saving plot: {e}")

# Optionally, display the plot
# plt.show() # Uncomment this line if you want to display the plot directly when running the script 