import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

def generate_comparison_report(baseline_results, enhanced_results, output_dir):
    """
    Generate comparison report between baseline and enhanced model
    
    Args:
        baseline_results: Path to baseline results CSV
        enhanced_results: Path to enhanced model results CSV
        output_dir: Directory to save the report
        
    Returns:
        Summary DataFrame with comparison metrics
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    baseline_df = pd.read_csv(baseline_results)
    enhanced_df = pd.read_csv(enhanced_results)
    
    logger.info(f"Loaded baseline results with {len(baseline_df)} entries")
    logger.info(f"Loaded enhanced results with {len(enhanced_df)} entries")
    
    # Check if we have the same mixture IDs in both dataframes
    baseline_mixtures = set(baseline_df['mixture_id'])
    enhanced_mixtures = set(enhanced_df['mixture_id'])
    common_mixtures = baseline_mixtures.intersection(enhanced_mixtures)
    
    logger.info(f"Found {len(common_mixtures)} common mixtures for comparison")
    
    # Filter to common mixtures
    baseline_df = baseline_df[baseline_df['mixture_id'].isin(common_mixtures)]
    enhanced_df = enhanced_df[enhanced_df['mixture_id'].isin(common_mixtures)]
    
    # Sort by mixture ID for consistency
    baseline_df = baseline_df.sort_values('mixture_id')
    enhanced_df = enhanced_df.sort_values('mixture_id')
    
    # Calculate metrics to compare
    metrics = ['SDR', 'SIR', 'SAR', 'PESQ']
    
    # Create summary dataframe
    summary = []
    
    for metric in metrics:
        baseline_mean = np.nanmean(baseline_df[metric])
        enhanced_mean = np.nanmean(enhanced_df[metric])
        difference = enhanced_mean - baseline_mean
        relative_improvement = (difference / baseline_mean) * 100 if baseline_mean != 0 else float('inf')
        
        summary.append({
            'Metric': metric,
            'Baseline': baseline_mean,
            'Enhanced': enhanced_mean,
            'Absolute Improvement': difference,
            'Relative Improvement (%)': relative_improvement
        })
    
    # Add speaker identification accuracy if available
    if 'pretrained_speaker_acc' in enhanced_df.columns:
        pretrained_acc = np.nanmean(enhanced_df['pretrained_speaker_acc'])
        summary.append({
            'Metric': 'Pretrained Speaker ID Accuracy',
            'Baseline': 'N/A',
            'Enhanced': pretrained_acc,
            'Absolute Improvement': 'N/A',
            'Relative Improvement (%)': 'N/A'
        })
    
    if 'finetuned_speaker_acc' in enhanced_df.columns:
        finetuned_acc = np.nanmean(enhanced_df['finetuned_speaker_acc'])
        summary.append({
            'Metric': 'Finetuned Speaker ID Accuracy',
            'Baseline': 'N/A',
            'Enhanced': finetuned_acc,
            'Absolute Improvement': 'N/A',
            'Relative Improvement (%)': 'N/A'
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Save summary to CSV
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    
    # Print summary
    logger.info("Comparison Summary:")
    for _, row in summary_df.iterrows():
        metric = row['Metric']
        baseline = row['Baseline']
        enhanced = row['Enhanced']
        abs_improvement = row['Absolute Improvement']
        rel_improvement = row['Relative Improvement (%)']
        
        # Format the output based on data type
        if isinstance(baseline, str) or isinstance(enhanced, str):
            logger.info(f"  {metric}: Baseline={baseline}, Enhanced={enhanced}")
        else:
            logger.info(f"  {metric}: Baseline={baseline:.4f}, Enhanced={enhanced:.4f}, " +
                       f"Abs. Improvement={abs_improvement:.4f}, Rel. Improvement={rel_improvement:.2f}%")
    
    # Generate visualizations
    # 1. Metric comparison bar chart
    plot_metrics = []
    baseline_values = []
    enhanced_values = []
    
    for metric in metrics:
        plot_metrics.append(metric)
        baseline_values.append(np.nanmean(baseline_df[metric]))
        enhanced_values.append(np.nanmean(enhanced_df[metric]))
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(plot_metrics))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, baseline_values, width, label='Baseline (SepFormer)')
    plt.bar(x + width/2, enhanced_values, width, label='Enhanced Combined Model')
    
    # Add labels and title
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Speech Enhancement Metrics')
    plt.xticks(x, plot_metrics)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(baseline_values):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(enhanced_values):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png")
    plt.close()
    
    # 2. Per-mixture comparison scatter plots
    for metric in metrics:
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(baseline_df[metric], enhanced_df[metric], alpha=0.6)
        
        # Add diagonal line (y=x)
        min_val = min(baseline_df[metric].min(), enhanced_df[metric].min())
        max_val = max(baseline_df[metric].max(), enhanced_df[metric].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel(f'Baseline {metric}')
        plt.ylabel(f'Enhanced {metric}')
        plt.title(f'Per-Mixture Comparison of {metric}')
        
        # Add color based on improvement
        improvement = enhanced_df[metric] - baseline_df[metric]
        positive = improvement > 0
        
        # Count number improved
        num_improved = positive.sum()
        percent_improved = (num_improved / len(positive)) * 100
        
        plt.text(0.05, 0.95, f'Improved: {num_improved}/{len(positive)} ({percent_improved:.1f}%)',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_scatter.png")
        plt.close()
    
    # 3. Generate a histograms of improvements
    for metric in metrics:
        improvement = enhanced_df[metric] - baseline_df[metric]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(improvement, kde=True)
        
        plt.xlabel(f'{metric} Improvement')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {metric} Improvement')
        
        # Add a vertical line at x=0
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Calculate and display statistics
        mean_improvement = np.nanmean(improvement)
        median_improvement = np.nanmedian(improvement)
        
        plt.text(0.05, 0.95, f'Mean: {mean_improvement:.2f}\nMedian: {median_improvement:.2f}',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_improvement_histogram.png")
        plt.close()
    
    
    return summary_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparison report between baseline and enhanced model")
    parser.add_argument("--baseline_results", type=str, required=True,
                        help="Path to baseline results CSV")
    parser.add_argument("--enhanced_results", type=str, required=True,
                        help="Path to enhanced model results CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the report")
    
    args = parser.parse_args()
    
    # Generate comparison report
    summary_df = generate_comparison_report(
        args.baseline_results,
        args.enhanced_results,
        args.output_dir
    )
    
    logger.info("Report generation complete!")

if __name__ == "__main__":
    main() 