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
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio
from model import load_combined_model
from speech_enhancement.evaluation import evaluate_separation
from train import MultiSpeakerMixtureDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

def evaluate_speaker_identification(model, test_data, output_dir):
    """
    Evaluate speaker identification performance
    
    Args:
        model: Combined model
        test_data: Test data loader
        output_dir: Directory to save results
        
    Returns:
        DataFrame with identification results
    """
    model.eval()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Evaluating speaker identification"):
            # Get batch data
            mixture = batch['mixture'].to(device)
            speaker1_id = batch['speaker1_id']
            speaker2_id = batch['speaker2_id']
            mixture_id = batch['mixture_id']
            
            # Forward pass
            enhanced_sources, source_embeddings = model(mixture)
            
            # Save enhanced sources for audio quality evaluation
            for i, (mixture_id_i, enhanced_source) in enumerate(zip(mixture_id, enhanced_sources)):
                enhanced_dir = output_dir / "enhanced" / str(mixture_id_i)
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                
                for j, source in enumerate(enhanced_source):
                    save_audio(
                        source.unsqueeze(0),
                        str(enhanced_dir / f"source{j+1}.wav")
                    )
            
            # Add to results
            for i, (mix_id, spk1, spk2) in enumerate(zip(mixture_id, speaker1_id, speaker2_id)):
                results.append({
                    'mixture_id': mix_id,
                    'true_speaker1': spk1,
                    'true_speaker2': spk2,
                    'pred_speaker1_embedding': source_embeddings[0][i].cpu().numpy(),
                    'pred_speaker2_embedding': source_embeddings[1][i].cpu().numpy() if len(source_embeddings) > 1 else None
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_dir / "speaker_identification_results.csv", index=False)
    
    return results_df

def calculate_rank1_identification_accuracy(speaker_model, reference_embeddings, 
                                           predicted_embeddings):
    """
    Calculate Rank-1 identification accuracy
    
    Args:
        speaker_model: Speaker verification model
        reference_embeddings: Reference speaker embeddings [num_speakers, embedding_dim]
        predicted_embeddings: Predicted speaker embeddings [num_predictions, embedding_dim]
        
    Returns:
        Rank-1 accuracy
    """
    # Calculate similarity matrix
    similarities = F.cosine_similarity(
        predicted_embeddings.unsqueeze(1),  # [num_predictions, 1, embedding_dim]
        reference_embeddings.unsqueeze(0),  # [1, num_speakers, embedding_dim]
        dim=2
    )
    
    # Get the most similar reference for each prediction
    top1_indices = torch.argmax(similarities, dim=1)
    
    # Ground truth indices (assuming 1-to-1 correspondence)
    gt_indices = torch.arange(len(predicted_embeddings))
    
    # Calculate accuracy
    accuracy = (top1_indices == gt_indices).float().mean().item()
    
    return accuracy

def evaluate_model(model, test_loader, output_dir, pretrained_speaker_model=None,
                  finetuned_speaker_model=None):
    """
    Evaluate the combined speech enhancement model
    
    Args:
        model: Combined model
        test_loader: DataLoader for test data
        output_dir: Directory to save evaluation results
        pretrained_speaker_model: Pretrained speaker verification model
        finetuned_speaker_model: Fine-tuned speaker verification model
        
    Returns:
        DataFrame with evaluation results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on test set
    model.eval()
    
    # Initialize results
    results = []
    
    # Process each batch
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            # Get batch data
            mixture = batch['mixture'].to(device)
            source1 = batch['source1'].to(device)
            source2 = batch['source2'].to(device)
            mixture_id = batch['mixture_id']
            
            # Forward pass (get enhanced sources)
            enhanced_sources, speaker_embeddings = model(mixture)
            
            # Evaluate speech separation quality for each item in the batch
            for i, (mix_id, enh1, enh2, src1, src2) in enumerate(zip(
                mixture_id, enhanced_sources[0], 
                enhanced_sources[1] if len(enhanced_sources) > 1 else [None] * len(mixture_id),
                source1, source2
            )):
                # Ensure we have 2D sources for evaluate_separation
                reference_sources = np.stack([src1.cpu().numpy(), src2.cpu().numpy()])
                estimated_sources = np.stack([
                    enh1.cpu().numpy(),
                    enh2.cpu().numpy() if enh2 is not None else np.zeros_like(enh1.cpu().numpy())
                ])
                
                # Evaluate separation quality
                metrics = evaluate_separation(reference_sources, estimated_sources)
                
                # Speaker identification accuracy with pretrained model
                pretrained_acc = None
                if pretrained_speaker_model is not None:
                    # Perform speaker ID with pretrained model
                    # This would need to be implemented based on how your speaker model works
                    pretrained_acc = 0.0  # Placeholder
                
                # Speaker identification accuracy with finetuned model
                finetuned_acc = None
                if finetuned_speaker_model is not None:
                    # Perform speaker ID with finetuned model
                    finetuned_acc = 0.0  # Placeholder
                
                # Add to results
                results.append({
                    'mixture_id': mix_id,
                    'SDR': metrics['SDR'],
                    'SIR': metrics['SIR'],
                    'SAR': metrics['SAR'],
                    'PESQ': metrics.get('PESQ', float('nan')),
                    'pretrained_speaker_acc': pretrained_acc,
                    'finetuned_speaker_acc': finetuned_acc
                })
                
                # Save enhanced sources
                enhanced_dir = output_dir / "enhanced" / str(mix_id)
                enhanced_dir.mkdir(parents=True, exist_ok=True)
                
                save_audio(
                    enh1.unsqueeze(0).cpu(),
                    str(enhanced_dir / "source1.wav")
                )
                
                if enh2 is not None:
                    save_audio(
                        enh2.unsqueeze(0).cpu(),
                        str(enhanced_dir / "source2.wav")
                    )
    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate combined speech enhancement model")
    parser.add_argument("--test_metadata", type=str, required=True,
                        help="Path to test metadata CSV file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--speaker_model_name", type=str, default="wavlm_base_plus",
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Name of the speaker model architecture")
    parser.add_argument("--pretrained_speaker_model", type=str, default=None,
                        help="Path to pretrained speaker model for comparison")
    parser.add_argument("--finetuned_speaker_model", type=str, default=None,
                        help="Path to finetuned speaker model for comparison")
    parser.add_argument("--output_dir", type=str, default="results/combined_pipeline",
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
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = load_combined_model(None, args.speaker_model_name)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Evaluate model
    results_df, avg_metrics = evaluate_model(
        model,
        test_loader,
        args.output_dir,
        args.pretrained_speaker_model,
        args.finetuned_speaker_model
    ) 