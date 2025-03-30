"""
Helper functions for speaker verification notebook.
These functions are designed to work well in a Jupyter notebook environment.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
from IPython.display import clear_output, display, HTML
from IPython import get_ipython

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if we're in a notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except:
        return False      # Probably standard Python interpreter

# Use a different progress display method depending on environment
if is_notebook():
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        from tqdm import tqdm
else:
    from tqdm import tqdm

def display_progress(current, total, message="", clear=True):
    """Display progress in a notebook-friendly way"""
    if is_notebook() and clear:
        clear_output(wait=True)
    
    percentage = 100 * current / total
    progress_bar = "▓" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
    
    print(f"{message}")
    print(f"Progress: {current}/{total} ({percentage:.2f}%) [{progress_bar}]")
    
    if not is_notebook():
        print()  # Add a newline for better terminal display

def run_pretrained_evaluation(model_name, trial_file, audio_root, voxceleb_dataset, 
                             batch_size=8, max_samples=100, 
                             output_dir="models/speaker_verification"):
    """
    Run evaluation of a pretrained model with nice progress tracking
    
    Args:
        model_name: Name of the pretrained model
        trial_file: Path to the trial file
        audio_root: Root directory for audio files
        voxceleb_dataset: VoxCelebTrialDataset instance
        batch_size: Batch size
        max_samples: Max samples to evaluate (set to -1 for all)
        output_dir: Directory to save outputs
        
    Returns:
        dict: Evaluation results
    """
    from src.speaker_verification.wait import load_pretrained_model, calculate_eer, calculate_tar_at_far, extract_embeddings, compute_similarity
    
    start_time = time.time()
    logger.info(f"Evaluating pretrained model: {model_name}")
    print(f"Starting evaluation of {model_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare log file
    log_file = os.path.join(output_dir, f"{model_name}_evaluation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load model
    print(f"Loading model {model_name}...")
    model, feature_extractor = load_pretrained_model(model_name)
    model.eval()
    
    # Create dataset subset if needed
    if max_samples > 0 and max_samples < len(voxceleb_dataset):
        print(f"Using {max_samples} samples for evaluation (out of {len(voxceleb_dataset)} total)")
        indices = list(range(len(voxceleb_dataset)))
        np.random.shuffle(indices)
        indices = indices[:max_samples]
        original_trials = voxceleb_dataset.trials
        voxceleb_dataset.trials = [original_trials[i] for i in indices]
    
    # Configure dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        voxceleb_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=lambda batch: {
            'label': torch.stack([sample['label'] for sample in batch]),
            'audio1': [sample['audio1'] for sample in batch],
            'audio2': [sample['audio2'] for sample in batch]
        }
    )
    
    # Process batches
    all_labels = []
    all_scores = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    total_batches = len(dataloader)
    print(f"Processing {len(voxceleb_dataset)} trial pairs in {total_batches} batches")
    
    # Progress tracking stats
    batch_times = []
    processed_pairs = 0
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        labels = batch['label']
        audio1_samples = batch['audio1']
        audio2_samples = batch['audio2']
        
        # Process samples
        batch_scores = []
        for i in range(len(audio1_samples)):
            with torch.no_grad():
                # Process first audio
                audio1 = audio1_samples[i].unsqueeze(0).to(device)
                emb1 = extract_embeddings(model, audio1)
                
                # Process second audio
                audio2 = audio2_samples[i].unsqueeze(0).to(device)
                emb2 = extract_embeddings(model, audio2)
                
                # Compute similarity
                score = compute_similarity(emb1, emb2).item()
                batch_scores.append(score)
        
        # Collect results
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(batch_scores)
        
        # Track progress
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        processed_pairs += len(labels)
        
        # Calculate ETA
        avg_time_per_batch = np.mean(batch_times)
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = avg_time_per_batch * remaining_batches
        
        # Display progress
        progress_message = (
            f"Batch {batch_idx+1}/{total_batches} - "
            f"{processed_pairs}/{len(voxceleb_dataset)} pairs processed\n"
            f"Batch time: {batch_time:.2f}s - Avg: {avg_time_per_batch:.2f}s per batch\n"
            f"ETA: {eta_seconds:.1f}s ({eta_seconds/60:.1f}m)"
        )
        
        display_progress(batch_idx + 1, total_batches, progress_message)
        
        # Log to file
        logger.info(f"Processed batch {batch_idx+1}/{total_batches} with {len(labels)} samples in {batch_time:.2f}s")
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    print("Calculating evaluation metrics...")
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far, far_threshold = calculate_tar_at_far(all_labels, all_scores, far_target=0.01)
    
    from sklearn.metrics import accuracy_score
    pred_labels = (all_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)
    
    results = {
        'model_name': model_name,
        'eer': eer * 100,  # Convert to percentage
        'tar_at_1far': tar_at_far * 100,  # Convert to percentage
        'accuracy': accuracy * 100,  # Convert to percentage
        'eer_threshold': eer_threshold,
        'optimal_far_threshold': far_threshold,
        'num_samples': len(all_labels),
        'evaluation_time': time.time() - start_time
    }
    
    # Log and display results
    logger.info(f"Evaluation results for {model_name}:")
    logger.info(f"  EER: {eer*100:.2f}%")
    logger.info(f"  TAR@1%FAR: {tar_at_far*100:.2f}%")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    logger.info(f"  Evaluation time: {results['evaluation_time']:.2f}s")
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS FOR {model_name}")
    print("="*50)
    print(f"EER: {eer*100:.2f}%")
    print(f"TAR@1%FAR: {tar_at_far*100:.2f}%")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Total evaluation time: {results['evaluation_time']:.2f}s ({results['evaluation_time']/60:.2f}m)")
    print("="*50)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot score distributions
    plt.subplot(1, 2, 1)
    plt.hist(all_scores[all_labels == 1], bins=30, alpha=0.5, label='Same Speaker')
    plt.hist(all_scores[all_labels == 0], bins=30, alpha=0.5, label='Different Speakers')
    plt.axvline(x=eer_threshold, color='r', linestyle='--', label=f'EER Threshold: {eer_threshold:.3f}')
    plt.axvline(x=far_threshold, color='g', linestyle='--', label=f'FAR@1% Threshold: {far_threshold:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution - {model_name}')
    plt.legend()
    
    # Plot ROC-like curve
    plt.subplot(1, 2, 2)
    # Sort scores in descending order for FAR/FRR calculation
    sorted_indices = np.argsort(all_scores)[::-1]
    sorted_scores = all_scores[sorted_indices]
    sorted_labels = all_labels[sorted_indices]
    
    # Calculate FAR and FRR
    far = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)
    frr = 1 - np.cumsum(sorted_labels) / np.sum(sorted_labels)
    
    plt.plot(far, 1-frr, '-', label='ROC Curve')
    plt.plot([0, 1], [0, 1], '--', label='Random Guess')
    plt.plot(eer, 1-eer, 'ro', label=f'EER: {eer*100:.2f}%')
    
    plt.xlabel('False Accept Rate')
    plt.ylabel('True Accept Rate')
    plt.title('Verification Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{model_name}_evaluation.png")
    plt.savefig(plot_path)
    logger.info(f"Saved plot to {plot_path}")
    print(f"Saved visualization to {plot_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    csv_path = os.path.join(output_dir, f"{model_name}_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    # Clean up
    logger.removeHandler(file_handler)
    
    return results

def create_evaluation_summary(results_dict, output_dir="models/speaker_verification"):
    """Create a summary of all evaluations"""
    if not results_dict:
        return "No evaluations to summarize"
    
    # Create summary DataFrame
    summary = pd.DataFrame(results_dict.values())
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary.to_csv(csv_path, index=False)
    
    # Create comparison chart
    plt.figure(figsize=(12, 6))
    
    # EER comparison (lower is better)
    plt.subplot(1, 3, 1)
    plt.bar(summary['model_name'], summary['eer'])
    plt.title('Equal Error Rate (EER) %\n(Lower is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # TAR@1%FAR comparison (higher is better)
    plt.subplot(1, 3, 2)
    plt.bar(summary['model_name'], summary['tar_at_1far'])
    plt.title('True Accept Rate @ 1% FAR\n(Higher is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Accuracy comparison (higher is better)
    plt.subplot(1, 3, 3)
    plt.bar(summary['model_name'], summary['accuracy'])
    plt.title('Accuracy %\n(Higher is Better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.show()
    
    return summary

def html_results_table(results_dict):
    """Create a pretty HTML table for display in the notebook"""
    if not results_dict:
        return HTML("<p>No results available</p>")
    
    # Create summary DataFrame
    summary = pd.DataFrame(results_dict.values())
    
    # Add evaluation time in minutes
    summary['eval_time_min'] = summary['evaluation_time'] / 60
    
    # Format the table
    styled = summary[[
        'model_name', 'eer', 'tar_at_1far', 'accuracy', 
        'num_samples', 'eval_time_min'
    ]].style.format({
        'eer': '{:.2f}%',
        'tar_at_1far': '{:.2f}%',
        'accuracy': '{:.2f}%',
        'eval_time_min': '{:.2f} min'
    }).set_caption('Speaker Verification Model Evaluation Results')
    
    # Add background color based on performance (green is better)
    styled = styled.background_gradient(subset=['tar_at_1far', 'accuracy'], cmap='YlGn')
    # For EER, lower is better, so invert the colors
    styled = styled.background_gradient(subset=['eer'], cmap='YlGn_r')
    
    return styled 