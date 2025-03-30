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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio
from speech_enhancement.evaluation import evaluate_separation
from speaker_verification.pretrained_eval import load_pretrained_model, extract_embeddings
from speaker_verification.finetune import SpeakerVerificationModel, apply_lora
from train_enhanced_pipeline import load_enhanced_model, EnhancedCombinedModel

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

def evaluate_enhanced_model(model, test_metadata_file, output_dir,
                          pretrained_speaker_model=None, finetuned_speaker_model=None):
    """
    Evaluate the enhanced combined model on test set
    
    Args:
        model: Enhanced combined model
        test_metadata_file: Path to test metadata file
        output_dir: Directory to save evaluation results
        pretrained_speaker_model: Pretrained speaker model for identification (optional)
        finetuned_speaker_model: Fine-tuned speaker model for identification (optional)
        
    Returns:
        DataFrame with evaluation results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test metadata
    test_metadata = pd.read_csv(test_metadata_file)
    logger.info(f"Loaded test metadata with {len(test_metadata)} entries")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize results list
    results = []
    
    # Get unique speakers in the test set for reference embeddings
    unique_speakers = set()
    speaker_references = {}
    
    for _, row in test_metadata.iterrows():
        speaker1_id = row['speaker1_id']
        speaker2_id = row['speaker2_id']
        
        unique_speakers.add(speaker1_id)
        unique_speakers.add(speaker2_id)
        
        # Store first encountered file for each speaker as reference
        if speaker1_id not in speaker_references:
            speaker_references[speaker1_id] = row['source1_path']
        if speaker2_id not in speaker_references:
            speaker_references[speaker2_id] = row['source2_path']
    
    logger.info(f"Found {len(unique_speakers)} unique speakers in test set")
    
    # Extract reference embeddings for all speakers
    reference_embeddings = {}
    
    for speaker_id, audio_path in speaker_references.items():
        try:
            waveform, _ = load_audio(audio_path)
            waveform = waveform.unsqueeze(0).to(device)
            
            # Extract embedding using the speaker model from our combined model
            with torch.no_grad():
                embedding = model.speaker_model(waveform)
                reference_embeddings[speaker_id] = embedding.squeeze(0)
        except Exception as e:
            logger.error(f"Error extracting reference embedding for {speaker_id}: {e}")
    
    # Process each test mixture
    with torch.no_grad():
        for idx, row in tqdm(test_metadata.iterrows(), total=len(test_metadata), desc="Evaluating test mixtures"):
            try:
                # Load mixture
                mixture_path = row['mixture_path']
                mixture, _ = load_audio(mixture_path)
                
                # Load reference sources
                source1_path = row['source1_path']
                source2_path = row['source2_path']
                source1, _ = load_audio(source1_path)
                source2, _ = load_audio(source2_path)
                
                # Get speaker IDs
                speaker1_id = row['speaker1_id']
                speaker2_id = row['speaker2_id']
                mixture_id = row['mixture_id']
                
                # Prepare reference embeddings for this mixture
                mixture_ref_embeddings = torch.stack([
                    reference_embeddings[speaker1_id],
                    reference_embeddings[speaker2_id]
                ]).to(device)
                
                # Process mixture through model
                mixture_tensor = mixture.unsqueeze(0).to(device)
                enhanced_sources, source_embeddings = model(mixture_tensor, mixture_ref_embeddings)
                
                # Convert to numpy for evaluation
                source1_np = source1.cpu().numpy()
                source2_np = source2.cpu().numpy()
                enhanced1_np = enhanced_sources[0].squeeze(0).cpu().numpy()
                enhanced2_np = enhanced_sources[1].squeeze(0).cpu().numpy()
                
                # Make sure they have the same length for evaluation
                min_length = min(len(source1_np), len(source2_np), len(enhanced1_np), len(enhanced2_np))
                source1_np = source1_np[:min_length]
                source2_np = source2_np[:min_length]
                enhanced1_np = enhanced1_np[:min_length]
                enhanced2_np = enhanced2_np[:min_length]
                
                # Stack for BSS eval
                reference_sources = np.stack([source1_np, source2_np])
                estimated_sources = np.stack([enhanced1_np, enhanced2_np])
                
                # Calculate separation metrics
                separation_metrics = evaluate_separation(reference_sources, estimated_sources)
                
                # Calculate PESQ for each source
                pesq1 = calculate_pesq(source1_np, enhanced1_np)
                pesq2 = calculate_pesq(source2_np, enhanced2_np)
                pesq_avg = np.nanmean([pesq1, pesq2])
                
                # Evaluate speaker identification with pretrained model
                pretrained_accuracy = None
                if pretrained_speaker_model is not None:
                    # Extract embeddings from enhanced sources
                    enhanced1_tensor = torch.from_numpy(enhanced1_np).unsqueeze(0).to(device)
                    enhanced2_tensor = torch.from_numpy(enhanced2_np).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding1 = extract_embeddings(pretrained_speaker_model, enhanced1_tensor)
                        embedding2 = extract_embeddings(pretrained_speaker_model, enhanced2_tensor)
                    
                    test_embeddings = [embedding1, embedding2]
                    ref_embeddings = [
                        extract_embeddings(pretrained_speaker_model, source1.unsqueeze(0).to(device)),
                        extract_embeddings(pretrained_speaker_model, source2.unsqueeze(0).to(device))
                    ]
                    
                    pretrained_accuracy = calculate_speaker_identification_accuracy(
                        pretrained_speaker_model, test_embeddings, ref_embeddings
                    )
                
                # Evaluate speaker identification with finetuned model
                finetuned_accuracy = None
                if finetuned_speaker_model is not None:
                    # Extract embeddings from enhanced sources
                    enhanced1_tensor = torch.from_numpy(enhanced1_np).unsqueeze(0).to(device)
                    enhanced2_tensor = torch.from_numpy(enhanced2_np).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        embedding1 = finetuned_speaker_model(enhanced1_tensor)
                        embedding2 = finetuned_speaker_model(enhanced2_tensor)
                    
                    test_embeddings = [embedding1, embedding2]
                    ref_embeddings = [
                        finetuned_speaker_model(source1.unsqueeze(0).to(device)),
                        finetuned_speaker_model(source2.unsqueeze(0).to(device))
                    ]
                    
                    finetuned_accuracy = calculate_speaker_identification_accuracy(
                        finetuned_speaker_model, test_embeddings, ref_embeddings
                    )
                
                # Save enhanced sources
                enhanced_dir = output_dir / "enhanced" / f"mix_{mixture_id:05d}"
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                
                save_audio(
                    torch.from_numpy(enhanced1_np).unsqueeze(0),
                    str(enhanced_dir / "source1.wav")
                )
                
                save_audio(
                    torch.from_numpy(enhanced2_np).unsqueeze(0),
                    str(enhanced_dir / "source2.wav")
                )
                
                # Add to results
                results.append({
                    'mixture_id': mixture_id,
                    'speaker1_id': speaker1_id, 
                    'speaker2_id': speaker2_id,
                    'SDR': separation_metrics['SDR'],
                    'SIR': separation_metrics['SIR'],
                    'SAR': separation_metrics['SAR'],
                    'PESQ': pesq_avg,
                    'pretrained_speaker_acc': pretrained_accuracy,
                    'finetuned_speaker_acc': finetuned_accuracy
                })
                
            except Exception as e:
                logger.error(f"Error processing mixture {row['mixture_id']}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = {
        'SDR': np.nanmean(results_df['SDR']),
        'SIR': np.nanmean(results_df['SIR']),
        'SAR': np.nanmean(results_df['SAR']),
        'PESQ': np.nanmean(results_df['PESQ']),
    }
    
    if pretrained_speaker_model is not None:
        avg_metrics['pretrained_speaker_acc'] = np.nanmean(results_df['pretrained_speaker_acc'])
    
    if finetuned_speaker_model is not None:
        avg_metrics['finetuned_speaker_acc'] = np.nanmean(results_df['finetuned_speaker_acc'])
    
    # Log average metrics
    logger.info("Average metrics:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results to CSV
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    
    # Save average metrics
    with open(output_dir / "evaluation_summary.txt", 'w') as f:
        f.write("Average metrics:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Visualize metrics
    metrics_to_plot = ['SDR', 'SIR', 'SAR', 'PESQ']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(results_df[metric], bins=20)
        plt.title(f"{metric} Distribution")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_dir / f"{metric}_distribution.png")
        plt.close()
    
    return results_df, avg_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate enhanced combined speech separation model")
    parser.add_argument("--test_metadata", type=str, required=True,
                        help="Path to test metadata CSV file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--speaker_model_path", type=str, default=None,
                        help="Path to pretrained/finetuned speaker model used in training")
    parser.add_argument("--pretrained_speaker_model_path", type=str, default=None,
                        help="Path to pretrained speaker model for identification evaluation")
    parser.add_argument("--finetuned_speaker_model_path", type=str, default=None,
                        help="Path to finetuned speaker model for identification evaluation")
    parser.add_argument("--speaker_model_name", type=str, default="wavlm_base_plus",
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Name of the speaker model architecture")
    parser.add_argument("--no_speaker_conditioning", action="store_true",
                        help="Disable speaker conditioning in the enhancement network")
    
    args = parser.parse_args()
    
    # Load the model
    model = load_enhanced_model(
        speaker_model_path=args.speaker_model_path,
        speaker_model_name=args.speaker_model_name,
        use_speaker_conditioning=not args.no_speaker_conditioning
    )
    
    # Load saved model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info(f"Loaded model weights from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        return
    
    # Load pretrained speaker model if provided
    pretrained_speaker_model = None
    if args.pretrained_speaker_model_path:
        try:
            pretrained_base_model, _ = load_pretrained_model(args.speaker_model_name)
            pretrained_speaker_model = SpeakerVerificationModel(
                pretrained_base_model,
                embedding_dim=768,
                num_speakers=100
            )
            pretrained_speaker_model.load_state_dict(
                torch.load(args.pretrained_speaker_model_path, map_location=device),
                strict=False
            )
            pretrained_speaker_model.eval()
            logger.info(f"Loaded pretrained speaker model from {args.pretrained_speaker_model_path}")
        except Exception as e:
            logger.error(f"Error loading pretrained speaker model: {e}")
            pretrained_speaker_model = None
    
    # Load finetuned speaker model if provided
    finetuned_speaker_model = None
    if args.finetuned_speaker_model_path:
        try:
            finetuned_base_model, _ = load_pretrained_model(args.speaker_model_name)
            finetuned_speaker_model = SpeakerVerificationModel(
                finetuned_base_model,
                embedding_dim=768,
                num_speakers=100
            )
            
            # Try to load with LoRA if direct loading fails
            try:
                finetuned_speaker_model.load_state_dict(
                    torch.load(args.finetuned_speaker_model_path, map_location=device)
                )
            except:
                logger.info("Direct loading failed, attempting to load with LoRA...")
                finetuned_speaker_model = apply_lora(finetuned_speaker_model, args.speaker_model_name)
                finetuned_speaker_model.load_state_dict(
                    torch.load(args.finetuned_speaker_model_path, map_location=device),
                    strict=False
                )
                
            finetuned_speaker_model.eval()
            logger.info(f"Loaded finetuned speaker model from {args.finetuned_speaker_model_path}")
        except Exception as e:
            logger.error(f"Error loading finetuned speaker model: {e}")
            finetuned_speaker_model = None
    
    # Evaluate model
    results_df, avg_metrics = evaluate_enhanced_model(
        model,
        args.test_metadata,
        args.output_dir,
        pretrained_speaker_model,
        finetuned_speaker_model
    )
    
    # Print summary
    logger.info("Evaluation complete!")
    logger.info("Average metrics:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 