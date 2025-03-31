import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import mir_eval
from sklearn.manifold import TSNE

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio
from speech_enhancement.evaluation import evaluate_separation
from speaker_verification.pretrained_eval import load_pretrained_model, extract_embeddings
from speaker_verification.finetune import SpeakerVerificationModel, apply_lora
from train_enhanced_pipeline import load_enhanced_model, EnhancedCombinedModel, MultiSpeakerMixtureDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

# Try to import PESQ, but continue if not available
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
    logger.info("PESQ module is available for evaluation")
except ImportError:
    PESQ_AVAILABLE = False
    logger.warning("PESQ module not available, skipping PESQ calculation")

def calculate_pesq(reference, estimated, sample_rate=16000):
    """
    Calculate PESQ score between reference and estimated audio
    
    Args:
        reference: Reference audio signal
        estimated: Estimated audio signal
        sample_rate: Audio sample rate
        
    Returns:
        PESQ score or NaN if calculation fails
    """
    if not PESQ_AVAILABLE:
        return float('nan')
    
    try:
        # Ensure same length
        min_len = min(len(reference), len(estimated))
        reference = reference[:min_len]
        estimated = estimated[:min_len]
        
        # Normalize to [-1, 1]
        reference = reference / (np.max(np.abs(reference)) + 1e-8)
        estimated = estimated / (np.max(np.abs(estimated)) + 1e-8)
        
        # Calculate PESQ
        mode = 'wb' if sample_rate == 16000 else 'nb'
        score = pesq(sample_rate, reference, estimated, mode)
        return score
    except Exception as e:
        logger.warning(f"Error calculating PESQ: {e}")
        return float('nan')

def calculate_speaker_identification_accuracy(model, test_embeddings, reference_embeddings):
    """
    Calculate speaker identification accuracy
    
    Args:
        model: Speaker verification model
        test_embeddings: Embeddings of the test audio
        reference_embeddings: Embeddings of the reference speakers
        
    Returns:
        Accuracy score
    """
    # Calculate similarity matrix
    num_test = len(test_embeddings)
    num_ref = len(reference_embeddings)
    
    similarities = torch.zeros((num_test, num_ref))
    
    for i in range(num_test):
        for j in range(num_ref):
            similarities[i, j] = F.cosine_similarity(
                test_embeddings[i].unsqueeze(0),
                reference_embeddings[j].unsqueeze(0),
                dim=1
            )
    
    # Get predicted speaker for each test embedding
    predicted_speakers = torch.argmax(similarities, dim=1)
    
    # Ground truth (assuming 1-to-1 mapping between test and reference)
    ground_truth = torch.arange(num_test)
    
    # Calculate accuracy
    correct = (predicted_speakers == ground_truth).sum().item()
    accuracy = correct / num_test
    
    return accuracy

def calculate_rank1_accuracy(reference_embeddings, test_embeddings, speaker_ids):
    """
    Calculate Rank-1 identification accuracy
    
    Args:
        reference_embeddings: Dictionary mapping speaker_id to embedding vector
        test_embeddings: List of embeddings from separated sources
        speaker_ids: Ground truth speaker IDs for the sources
        
    Returns:
        Rank-1 accuracy, correct predictions count, total predictions count
    """
    correct = 0
    total = 0
    
    # Convert reference embeddings to tensor
    ref_ids = list(reference_embeddings.keys())
    ref_embeds = torch.stack([reference_embeddings[spk_id] for spk_id in ref_ids]).to(device)
    
    # For each test embedding, find the closest reference
    for emb, true_id in zip(test_embeddings, speaker_ids):
        # Calculate cosine similarity with all reference embeddings
        similarities = F.cosine_similarity(emb.unsqueeze(0), ref_embeds)
        
        # Get the most similar reference embedding
        top_idx = torch.argmax(similarities).item()
        pred_id = ref_ids[top_idx]
        
        # Check if prediction is correct
        if pred_id == true_id:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def visualize_embeddings(reference_embeddings, test_embeddings, speaker_ids, output_path):
    """
    Visualize embeddings using t-SNE
    
    Args:
        reference_embeddings: Dictionary mapping speaker_id to embedding vector
        test_embeddings: List of embeddings from separated sources
        speaker_ids: Ground truth speaker IDs for the sources
        output_path: Path to save visualization
    """
    # Prepare data for t-SNE
    ref_ids = list(reference_embeddings.keys())
    ref_embeds = [reference_embeddings[spk_id].cpu().numpy() for spk_id in ref_ids]
    test_embeds = [emb.cpu().numpy() for emb in test_embeddings]
    
    # Combine reference and test embeddings
    all_embeds = np.vstack(ref_embeds + test_embeds)
    all_ids = ref_ids + speaker_ids
    
    # Calculate t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeds)-1))
    embeddings_2d = tsne.fit_transform(all_embeds)
    
    # Split embeddings for plotting
    ref_points = embeddings_2d[:len(ref_embeds)]
    test_points = embeddings_2d[len(ref_embeds):]
    
    # Get unique speaker IDs for coloring
    unique_ids = list(set(all_ids))
    id_to_color = {spk_id: i for i, spk_id in enumerate(unique_ids)}
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot reference embeddings
    for i, (point, spk_id) in enumerate(zip(ref_points, ref_ids)):
        plt.scatter(point[0], point[1], c=[id_to_color[spk_id]], marker='o', s=100, 
                   label=f'Reference: {spk_id}' if i == 0 else "")
    
    # Plot test embeddings
    for i, (point, spk_id) in enumerate(zip(test_points, speaker_ids)):
        plt.scatter(point[0], point[1], c=[id_to_color[spk_id]], marker='x', s=50, 
                   label=f'Test: {spk_id}' if i == 0 else "")
    
    plt.title('t-SNE Visualization of Speaker Embeddings')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def evaluate_enhanced_model(model, test_loader, output_dir, speaker_model=None):
    """
    Evaluate the enhanced speech separation and speaker identification model
    
    Args:
        model: Enhanced model
        test_loader: DataLoader for test data
        output_dir: Directory to save evaluation results
        speaker_model: Optional additional speaker model for comparison
        
    Returns:
        DataFrame with evaluation results and average metrics
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Initialize results storage
    results = []
    
    # Storage for speaker embeddings
    reference_embeddings = {}
    test_embeddings = []
    test_speaker_ids = []
    
    # Process each batch
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating model")
        for batch in progress_bar:
            # Get batch data
            mixture = batch['mixture'].to(device)
            source1 = batch['source1'].to(device)
            source2 = batch['source2'].to(device)
            
            # Get speaker IDs
            speaker1_ids = batch['speaker1_id']
            speaker2_ids = batch['speaker2_id']
            
            # Get mixture IDs
            mixture_ids = batch['mixture_id']
            
            # Forward pass
            enhanced_sources, source_embeddings = model(mixture)
            
            # Process each item in batch
            for i in range(mixture.shape[0]):
                mixture_id = mixture_ids[i] if isinstance(mixture_ids, list) else mixture_ids[i].item()
                
                # Get ground truth sources for this item
                ref_src1 = source1[i].cpu().numpy()
                ref_src2 = source2[i].cpu().numpy()
                
                # Get enhanced sources for this item
                enh_src1 = enhanced_sources[0][i].cpu().numpy()
                enh_src2 = enhanced_sources[1][i].cpu().numpy()
                
                # Enhanced sources are now expected to be 8kHz (due to changes in forward pass)
                # Reference sources are loaded at 16kHz
                try:
                    original_sr = 16000
                    target_sr = 8000
                    
                    # Resample reference sources to target_sr (8kHz)
                    ref_src1_tensor = torch.from_numpy(ref_src1).to('cpu') # Move to CPU if needed
                    ref_src2_tensor = torch.from_numpy(ref_src2).to('cpu')
                    
                    ref_src1_resampled = torchaudio.functional.resample(ref_src1_tensor, original_sr, target_sr)
                    ref_src2_resampled = torchaudio.functional.resample(ref_src2_tensor, original_sr, target_sr)
                    
                    # Convert enhanced sources to tensors for length alignment
                    enh_src1_tensor = torch.from_numpy(enh_src1).to('cpu')
                    enh_src2_tensor = torch.from_numpy(enh_src2).to('cpu')
                    
                    # Align lengths by trimming to the minimum length
                    min_len = min(ref_src1_resampled.shape[0], enh_src1_tensor.shape[0])
                    
                    ref_src1_aligned = ref_src1_resampled[:min_len].numpy()
                    ref_src2_aligned = ref_src2_resampled[:min_len].numpy()
                    enh_src1_aligned = enh_src1_tensor[:min_len].numpy()
                    enh_src2_aligned = enh_src2_tensor[:min_len].numpy()
                    
                    logger.info(f"Mixture {mixture_id}: Aligned length={min_len} at {target_sr}Hz")
                    
                    # Confirm shapes match after adjustment
                    assert len(ref_src1_aligned) == len(enh_src1_aligned), f"Lengths still don't match: {len(ref_src1_aligned)} vs {len(enh_src1_aligned)}"
                    
                    # Stack for evaluation (using aligned 8kHz versions)
                    ref_sources = np.stack([ref_src1_aligned, ref_src2_aligned])
                    enh_sources = np.stack([enh_src1_aligned, enh_src2_aligned])
                    
                    # Evaluate separation quality (pass target_sr for PESQ calculation)
                    metrics = evaluate_separation(ref_sources, enh_sources, sample_rate=target_sr)
                    
                    # Store reference embeddings for speaker ID evaluation (embeddings are extracted from 8kHz sources)
                    spk1_id = speaker1_ids[i]
                    spk2_id = speaker2_ids[i]
                    
                    # Add to reference embeddings if not already present
                    if spk1_id not in reference_embeddings:
                        reference_embeddings[spk1_id] = source_embeddings[0][i].detach().clone()
                    
                    if spk2_id not in reference_embeddings:
                        reference_embeddings[spk2_id] = source_embeddings[1][i].detach().clone()
                    
                    # Add test embeddings and IDs for later evaluation
                    test_embeddings.append(source_embeddings[0][i].detach().clone())
                    test_embeddings.append(source_embeddings[1][i].detach().clone())
                    test_speaker_ids.append(spk1_id)
                    test_speaker_ids.append(spk2_id)
                    
                    # Save enhanced sources
                    enhanced_dir = output_dir / "enhanced" / str(mixture_id)
                    enhanced_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save source 1
                    save_audio(
                        torch.tensor(enh_src1_aligned).unsqueeze(0),
                        str(enhanced_dir / "source1.wav")
                    )
                    
                    # Save source 2
                    save_audio(
                        torch.tensor(enh_src2_aligned).unsqueeze(0),
                        str(enhanced_dir / "source2.wav")
                    )
                    
                    # Add metrics to results
                    results.append({
                        'mixture_id': mixture_id,
                        'speaker1_id': spk1_id,
                        'speaker2_id': spk2_id,
                        'SDR': metrics['SDR'],
                        'SIR': metrics['SIR'],
                        'SAR': metrics['SAR'],
                        'PESQ': metrics.get('PESQ', float('nan')),
                    })
                    
                except Exception as e:
                    logger.error(f"Error evaluating mixture {mixture_id}: {e}")
                    results.append({
                        'mixture_id': mixture_id,
                        'speaker1_id': speaker1_ids[i],
                        'speaker2_id': speaker2_ids[i],
                        'SDR': float('nan'),
                        'SIR': float('nan'),
                        'SAR': float('nan'),
                        'PESQ': float('nan'),
                    })
    
    # Calculate Rank-1 identification accuracy
    rank1_acc, correct, total = calculate_rank1_accuracy(
        reference_embeddings, test_embeddings, test_speaker_ids
    )
    
    logger.info(f"Rank-1 identification accuracy: {rank1_acc:.4f} ({correct}/{total})")
    
    # Visualize embeddings
    try:
        visualize_embeddings(
            reference_embeddings, 
            test_embeddings, 
            test_speaker_ids,
            output_dir / "embeddings_visualization.png"
        )
    except Exception as e:
        logger.error(f"Error visualizing embeddings: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = {
        'SDR': np.nanmean(results_df['SDR']),
        'SIR': np.nanmean(results_df['SIR']),
        'SAR': np.nanmean(results_df['SAR']),
        'PESQ': np.nanmean(results_df['PESQ']),
        'Rank1_Accuracy': rank1_acc,
    }
    
    # Log average metrics
    logger.info(f"Average metrics:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    
    # Save average metrics
    with open(output_dir / "evaluation_summary.txt", 'w') as f:
        f.write("Average metrics:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return results_df, avg_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate enhanced combined speech separation model")
    parser.add_argument("--test_metadata", type=str, required=True,
                        help="Path to test metadata CSV file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--speaker_model_path", type=str, default=None,
                        help="Path to speaker model for rank-1 accuracy evaluation")
    parser.add_argument("--speaker_model_name", type=str, default="wavlm_base_plus",
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Name of the speaker model architecture")
    parser.add_argument("--output_dir", type=str, default="results/enhanced_pipeline",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Create test dataset
    test_dataset = MultiSpeakerMixtureDataset(args.test_metadata)
    
    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load model
    model = load_enhanced_model(
        speaker_model_path=args.speaker_model_path,
        speaker_model_name=args.speaker_model_name
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info(f"Successfully loaded model weights from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        # Try to load with strict=False
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
            logger.info(f"Loaded model weights with strict=False from {args.model_path}")
        except Exception as e:
            logger.error(f"Error loading model weights with strict=False: {e}")
            logger.warning("Proceeding with uninitialized model - results may be poor")
    
    # Evaluate model
    results_df, avg_metrics = evaluate_enhanced_model(
        model,
        test_loader,
        args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"SDR: {avg_metrics['SDR']:.4f} dB")
    print(f"SIR: {avg_metrics['SIR']:.4f} dB")
    print(f"SAR: {avg_metrics['SAR']:.4f} dB")
    print(f"PESQ: {avg_metrics['PESQ']:.4f}")
    print(f"Rank-1 Accuracy: {avg_metrics['Rank1_Accuracy']:.4f}")
    print("="*50)
    print(f"Detailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 