import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import mir_eval
import librosa
from pesq import pesq
from pystoi import stoi

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

def align_signals(reference, estimated):
    """
    Align reference and estimated signals
    
    Args:
        reference: Reference signal
        estimated: Estimated signal
        
    Returns:
        Aligned reference and estimated signals
    """
    # Make the signals the same length
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]
    
    return reference, estimated

def evaluate_separation(reference_sources, estimated_sources, sample_rate=16000):
    """
    Evaluate the quality of source separation
    
    Args:
        reference_sources: Reference source signals
        estimated_sources: Estimated source signals
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure we have numpy arrays
    if isinstance(reference_sources, torch.Tensor):
        reference_sources = reference_sources.numpy()
    if isinstance(estimated_sources, torch.Tensor):
        estimated_sources = estimated_sources.numpy()
    
    # BSS Eval metrics
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        reference_sources, 
        estimated_sources,
        compute_permutation=True
    )
    
    # Reorder estimated sources according to permutation
    estimated_sources_ordered = estimated_sources[perm]
    
    # Collect metrics
    metrics = {
        'SDR': np.mean(sdr),  # Signal to Distortion Ratio
        'SIR': np.mean(sir),  # Signal to Interference Ratio
        'SAR': np.mean(sar),  # Signal to Artifacts Ratio
        'SDR_per_source': sdr.tolist(),
        'SIR_per_source': sir.tolist(),
        'SAR_per_source': sar.tolist(),
    }
    
    # PESQ (if only 2 sources)
    if len(reference_sources) == 2 and len(estimated_sources) == 2:
        try:
            # Calculate PESQ for each source
            pesq_scores = []
            for i in range(len(reference_sources)):
                ref = reference_sources[i]
                est = estimated_sources_ordered[i]
                
                # Align signals
                ref, est = align_signals(ref, est)
                
                # Normalize to [-1, 1]
                ref = ref / np.max(np.abs(ref))
                est = est / np.max(np.abs(est))
                
                # Resample if needed (PESQ requires 8kHz or 16kHz)
                if sample_rate != 8000 and sample_rate != 16000:
                    ref = librosa.resample(ref, orig_sr=sample_rate, target_sr=16000)
                    est = librosa.resample(est, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                # Calculate PESQ
                try:
                    score = pesq(sample_rate, ref, est, 'wb' if sample_rate == 16000 else 'nb')
                    pesq_scores.append(score)
                except Exception as e:
                    logger.warning(f"Error calculating PESQ: {e}")
                    pesq_scores.append(float('nan'))
            
            # Add to metrics
            metrics['PESQ'] = np.nanmean(pesq_scores)
            metrics['PESQ_per_source'] = pesq_scores
        except Exception as e:
            logger.warning(f"Error calculating PESQ: {e}")
            metrics['PESQ'] = float('nan')
            metrics['PESQ_per_source'] = [float('nan')] * len(reference_sources)
    
    return metrics

def evaluate_test_set(ground_truth_file, separated_dir, output_file):
    """
    Evaluate separation performance on a test set
    
    Args:
        ground_truth_file: CSV file with ground truth source paths
        separated_dir: Directory containing separated sources
        output_file: File to save evaluation results
        
    Returns:
        DataFrame with evaluation metrics
    """
    # Read ground truth metadata
    ground_truth = pd.read_csv(ground_truth_file)
    
    # Read separation results
    if os.path.exists(os.path.join(separated_dir, "separated_sources.csv")):
        separation_results = pd.read_csv(os.path.join(separated_dir, "separated_sources.csv"))
        
        # Merge dataframes
        test_set = pd.merge(ground_truth, separation_results, on='mixture_id')
    else:
        logger.warning("No separated_sources.csv found, using ground truth only")
        test_set = ground_truth
        test_set['source1_path_y'] = None
        test_set['source2_path_y'] = None
    
    # Rename columns for clarity
    test_set = test_set.rename(columns={
        'source1_path_x': 'reference_source1',
        'source2_path_x': 'reference_source2',
        'source1_path_y': 'estimated_source1',
        'source2_path_y': 'estimated_source2',
    })
    
    # Evaluate each mixture
    results = []
    for idx, row in tqdm(test_set.iterrows(), total=len(test_set), desc="Evaluating"):
        mixture_id = row['mixture_id']
        
        try:
            # Load reference sources
            ref_source1, sr1 = load_audio(row['reference_source1'])
            ref_source2, sr2 = load_audio(row['reference_source2'])
            
            # Convert to numpy
            ref_source1 = ref_source1.numpy()
            ref_source2 = ref_source2.numpy()
            
            # Reference sources
            reference_sources = np.array([ref_source1, ref_source2])
            
            # If we have separated sources
            if row['estimated_source1'] and row['estimated_source2']:
                # Load estimated sources
                est_source1, sr3 = load_audio(row['estimated_source1'])
                est_source2, sr4 = load_audio(row['estimated_source2'])
                
                # Convert to numpy
                est_source1 = est_source1.numpy()
                est_source2 = est_source2.numpy()
                
                # Estimated sources
                estimated_sources = np.array([est_source1, est_source2])
                
                # Evaluate
                metrics = evaluate_separation(reference_sources, estimated_sources, sr1)
            else:
                # No separated sources
                metrics = {
                    'SDR': float('nan'),
                    'SIR': float('nan'),
                    'SAR': float('nan'),
                    'PESQ': float('nan'),
                }
            
            # Add to results
            results.append({
                'mixture_id': mixture_id,
                'speaker1_id': row['speaker1_id'],
                'speaker2_id': row['speaker2_id'],
                'SDR': metrics['SDR'],
                'SIR': metrics['SIR'],
                'SAR': metrics['SAR'],
                'PESQ': metrics.get('PESQ', float('nan')),
            })
        except Exception as e:
            logger.error(f"Error evaluating mixture {mixture_id}: {e}")
            results.append({
                'mixture_id': mixture_id,
                'speaker1_id': row.get('speaker1_id', 'unknown'),
                'speaker2_id': row.get('speaker2_id', 'unknown'),
                'SDR': float('nan'),
                'SIR': float('nan'),
                'SAR': float('nan'),
                'PESQ': float('nan'),
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = {
        'SDR': np.nanmean(results_df['SDR']),
        'SIR': np.nanmean(results_df['SIR']),
        'SAR': np.nanmean(results_df['SAR']),
        'PESQ': np.nanmean(results_df['PESQ']),
    }
    
    logger.info(f"Average metrics:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results to CSV
    results_df.to_csv(output_file, index=False)
    
    # Save average metrics
    with open(output_file.replace('.csv', '_summary.txt'), 'w') as f:
        f.write("Average metrics:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return results_df, avg_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate speech separation performance")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="CSV file with ground truth source paths")
    parser.add_argument("--separated_dir", type=str, required=True,
                        help="Directory containing separated sources")
    parser.add_argument("--output_file", type=str, required=True,
                        help="File to save evaluation results")
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate_test_set(args.ground_truth, args.separated_dir, args.output_file) 