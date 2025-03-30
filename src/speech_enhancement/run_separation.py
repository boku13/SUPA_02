import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import mir_eval

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio
from speaker_verification.pretrained_eval import load_pretrained_model, extract_embeddings

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

def create_trial_pairs(source_paths, reference_paths):
    """
    Create trial pairs for speaker identification
    
    Args:
        source_paths: List of paths to separated source files
        reference_paths: List of paths to reference audio files
        
    Returns:
        List of trial pairs
    """
    trials = []
    
    for source_path in source_paths:
        for ref_path in reference_paths:
            # Get speaker ID from reference path
            ref_speaker = Path(ref_path).parent.name
            # Add to trials
            trials.append((source_path, ref_path, ref_speaker))
    
    return trials

def identify_speakers(model, separated_sources, reference_speakers):
    """
    Identify which separated source corresponds to which speaker
    
    Args:
        model: Pretrained speaker verification model
        separated_sources: List of paths to separated source files
        reference_speakers: Dictionary mapping speaker IDs to reference audio paths
        
    Returns:
        Dictionary mapping source paths to predicted speaker IDs
    """
    # Create reference embeddings
    reference_embeddings = {}
    for speaker_id, audio_paths in reference_speakers.items():
        # Use the first file as reference
        ref_path = audio_paths[0]
        try:
            # Load audio
            waveform, sr = load_audio(ref_path)
            
            # Extract embedding
            with torch.no_grad():
                audio = waveform.unsqueeze(0).to(device)
                embedding = extract_embeddings(model, audio)
                reference_embeddings[speaker_id] = embedding
        except Exception as e:
            logger.error(f"Error processing reference audio {ref_path}: {e}")
    
    # Identify speakers in separated sources
    source_speakers = {}
    
    for source_path in tqdm(separated_sources, desc="Identifying speakers"):
        try:
            # Load audio
            waveform, sr = load_audio(source_path)
            
            # Extract embedding
            with torch.no_grad():
                audio = waveform.unsqueeze(0).to(device)
                source_embedding = extract_embeddings(model, audio)
            
            # Compare with reference embeddings
            best_score = -np.inf
            best_speaker = None
            
            for speaker_id, ref_embedding in reference_embeddings.items():
                # Compute similarity
                similarity = torch.nn.functional.cosine_similarity(
                    source_embedding, ref_embedding, dim=1
                ).item()
                
                # Update best match
                if similarity > best_score:
                    best_score = similarity
                    best_speaker = speaker_id
            
            # Assign speaker
            source_speakers[source_path] = (best_speaker, best_score)
        except Exception as e:
            logger.error(f"Error processing source audio {source_path}: {e}")
            source_speakers[source_path] = (None, -np.inf)
    
    return source_speakers

def evaluate_separation(reference_sources, estimated_sources, sample_rate=16000):
    """
    Evaluate the quality of source separation
    
    Args:
        reference_sources: Reference source signals [num_sources, num_samples]
        estimated_sources: Estimated source signals [num_sources, num_samples]
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
    
    # PESQ (if only 2 sources and PESQ available)
    if PESQ_AVAILABLE and len(reference_sources) == 2 and len(estimated_sources) == 2:
        try:
            # Calculate PESQ for each source
            pesq_scores = []
            for i in range(len(reference_sources)):
                ref = reference_sources[i]
                est = estimated_sources_ordered[i]
                
                # Ensure same length
                min_len = min(len(ref), len(est))
                ref = ref[:min_len]
                est = est[:min_len]
                
                # Normalize to [-1, 1]
                ref = ref / np.max(np.abs(ref))
                est = est / np.max(np.abs(est))
                
                # Resample if needed (PESQ requires 8kHz or 16kHz)
                if sample_rate != 8000 and sample_rate != 16000:
                    # Using numpy resample as fallback if librosa not available
                    from scipy import signal
                    target_len = int(len(ref) * 16000 / sample_rate)
                    ref = signal.resample(ref, target_len)
                    est = signal.resample(est, target_len)
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
    else:
        # PESQ not available, add placeholder values
        metrics['PESQ'] = float('nan')
        metrics['PESQ_per_source'] = [float('nan')] * len(reference_sources)
    
    return metrics

def process_test_set(test_metadata_file, separated_dir, output_dir, model_name="wavlm_base_plus"):
    """
    Process the test set with speaker identification and separation metrics
    
    Args:
        test_metadata_file: Path to test metadata file
        separated_dir: Directory containing separated sources
        output_dir: Directory to save results
        model_name: Name of the pretrained model to use
        
    Returns:
        DataFrame with results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info(f"Loading test metadata from {test_metadata_file}")
    test_metadata = pd.read_csv(test_metadata_file)
    
    # Add debugging info
    logger.info(f"Test metadata contains {len(test_metadata)} entries")
    logger.info(f"Test metadata columns: {test_metadata.columns.tolist()}")
    logger.info(f"First few mixture_ids: {test_metadata['mixture_id'].head().tolist()}")
    
    # Load pretrained model for speaker identification
    logger.info(f"Loading pretrained model: {model_name}")
    pretrained_model, _ = load_pretrained_model(model_name)
    pretrained_model.eval()
    
    # Also load the finetuned model if available
    finetuned_model = None
    finetuned_model_path = f"models/speaker_verification/wavlm_ft/best_model.pt"
    if os.path.exists(finetuned_model_path):
        logger.info(f"Loading finetuned model from {finetuned_model_path}")
        # Import here to avoid circular imports
        from speaker_verification.finetune import SpeakerVerificationModel
        
        # Create model with same architecture
        finetuned_model, _ = load_pretrained_model(model_name)
        speaker_model = SpeakerVerificationModel(
            finetuned_model,
            embedding_dim=768,
            num_speakers=100  # Not used in evaluation
        )
        
        # Load weights
        try:
            # First try to load directly (if the model was saved completely)
            speaker_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
        except RuntimeError as e:
            logger.warning(f"Direct loading failed: {e}")
            logger.info("Trying to load LoRA weights instead...")
            
            # Try to load LoRA weights if available
            lora_weights_path = os.path.join(os.path.dirname(finetuned_model_path), "lora_weights.pt")
            if os.path.exists(lora_weights_path):
                # Import PEFT utilities for LoRA models
                try:
                    from peft import (
                        LoraConfig, 
                        get_peft_model, 
                        prepare_model_for_kbit_training,
                        TaskType
                    )
                    
                    # Apply LoRA configuration
                    from speaker_verification.finetune import apply_lora
                    speaker_model = apply_lora(speaker_model, model_name)
                    
                    # Load LoRA weights
                    speaker_model.load_state_dict(torch.load(lora_weights_path, map_location=device), strict=False)
                    logger.info("Successfully loaded LoRA weights")
                except ImportError:
                    logger.error("PEFT library not available. Cannot load LoRA weights.")
                    finetuned_model = None
                except Exception as e:
                    logger.error(f"Error loading LoRA weights: {e}")
                    finetuned_model = None
            else:
                logger.error(f"Neither complete model nor LoRA weights found. Using pretrained model only.")
                finetuned_model = None
        
        if finetuned_model is not None:
            speaker_model.eval()
    
    # Group references by speaker
    reference_speakers = {}
    for _, row in test_metadata.iterrows():
        speaker1 = row['speaker1_id']
        speaker2 = row['speaker2_id']
        source1 = row['source1_path']
        source2 = row['source2_path']
        
        if speaker1 not in reference_speakers:
            reference_speakers[speaker1] = []
        if speaker2 not in reference_speakers:
            reference_speakers[speaker2] = []
        
        reference_speakers[speaker1].append(source1)
        reference_speakers[speaker2].append(source2)
    
    # Get list of separated mixtures
    separated_metadata_file = os.path.join(separated_dir, "separated_sources.csv")
    if os.path.exists(separated_metadata_file):
        logger.info(f"Loading separated sources metadata from {separated_metadata_file}")
        separated_metadata = pd.read_csv(separated_metadata_file)
        logger.info(f"Separated metadata contains {len(separated_metadata)} entries")
        logger.info(f"Separated metadata columns: {separated_metadata.columns.tolist()}")
        logger.info(f"First few mixture_ids: {separated_metadata['mixture_id'].head().tolist()}")
    else:
        # We need to create one if it doesn't exist
        logger.info("Separated sources metadata not found, creating from directory structure")
        separated_data = []
        for mixture_dir in Path(separated_dir).glob("*"):
            if mixture_dir.is_dir():
                mixture_id = mixture_dir.name
                source_files = list(mixture_dir.glob("*_source*.wav"))
                
                if len(source_files) >= 2:
                    separated_data.append({
                        'mixture_id': mixture_id,
                        'mixture_path': str(mixture_dir / f"{mixture_id}.wav"),
                        'source1_path': str(source_files[0]),
                        'source2_path': str(source_files[1]) if len(source_files) > 1 else None
                    })
                
        separated_metadata = pd.DataFrame(separated_data)
        separated_metadata.to_csv(separated_metadata_file, index=False)
    
    # Fix: Create a mapping column in test_metadata to match the format in separated_metadata
    test_metadata['mixture_id_str'] = test_metadata['mixture_id'].apply(lambda x: f"mix_{x:05d}")
    
    logger.info("Attempting to merge test metadata with separated metadata")
    # Merge with original metadata using the string format mixture ID
    try:
        test_data = pd.merge(test_metadata, separated_metadata, 
                            left_on='mixture_id_str', right_on='mixture_id', 
                            suffixes=('_original', '_separated'))
        logger.info(f"Successfully merged dataframes, resulting in {len(test_data)} entries")
    except Exception as e:
        logger.error(f"Error merging dataframes: {e}")
        # If standard merge fails, try with more verbose logging and fallback methods
        logger.info("Attempting alternative merge method...")
        
        # Print key information for debugging
        logger.info(f"test_metadata['mixture_id_str'] values: {test_metadata['mixture_id_str'].head().tolist()}")
        logger.info(f"separated_metadata['mixture_id'] values: {separated_metadata['mixture_id'].head().tolist()}")
        
        # Try more explicit merge
        common_ids = set(test_metadata['mixture_id_str']).intersection(set(separated_metadata['mixture_id']))
        logger.info(f"Found {len(common_ids)} common IDs between datasets")
        
        if len(common_ids) > 0:
            test_data = pd.merge(
                test_metadata, 
                separated_metadata,
                left_on='mixture_id_str', 
                right_on='mixture_id',
                suffixes=('_original', '_separated'),
                how='inner'  # Only keep rows that match
            )
            logger.info(f"Inner join resulted in {len(test_data)} entries")
        else:
            logger.error("No common IDs found between test metadata and separated sources")
            return None, None
    
    # Process each mixture
    results = []
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating mixtures"):
        mixture_id = row['mixture_id_separated'] if 'mixture_id_separated' in row else row['mixture_id']
        
        try:
            # Ensure necessary paths exist to avoid file-not-found errors
            source1_original = row.get('source1_path_original', row.get('source1_path', None))
            source2_original = row.get('source2_path_original', row.get('source2_path', None))
            source1_separated = row.get('source1_path_separated', None)
            source2_separated = row.get('source2_path_separated', None)
            
            # Check if files exist
            if (not source1_original or not os.path.exists(source1_original) or 
                not source2_original or not os.path.exists(source2_original)):
                logger.warning(f"Original source files not found for mixture {mixture_id}")
                logger.warning(f"Paths: {source1_original}, {source2_original}")
                continue
                
            if (not source1_separated or not os.path.exists(source1_separated) or 
                not source2_separated or not os.path.exists(source2_separated)):
                logger.warning(f"Separated source files not found for mixture {mixture_id}")
                logger.warning(f"Paths: {source1_separated}, {source2_separated}")
                continue
            
            # Log paths for debugging
            logger.info(f"Processing mixture {mixture_id}")
            logger.info(f"Original sources: {source1_original}, {source2_original}")
            logger.info(f"Separated sources: {source1_separated}, {source2_separated}")
            
            # Load reference sources
            ref_source1, sr1 = load_audio(source1_original)
            ref_source2, sr2 = load_audio(source2_original)
            
            # Load separated sources
            sep_source1, sr3 = load_audio(source1_separated)
            sep_source2, sr4 = load_audio(source2_separated)
            
            # Convert to numpy for evaluation
            ref_sources = np.stack([ref_source1.numpy(), ref_source2.numpy()])
            sep_sources = np.stack([sep_source1.numpy(), sep_source2.numpy()])
            
            # Evaluate separation quality
            metrics = evaluate_separation(ref_sources, sep_sources, sr1)
            
            # Get speaker IDs
            speaker1_id = row.get('speaker1_id', None)
            speaker2_id = row.get('speaker2_id', None)
            
            if not speaker1_id or not speaker2_id:
                logger.warning(f"Speaker IDs not found for mixture {mixture_id}")
                continue
            
            # Identify speakers with pretrained model
            pretrained_sources = identify_speakers(
                pretrained_model, 
                [source1_separated, source2_separated],
                {speaker1_id: [source1_original], 
                 speaker2_id: [source2_original]}
            )
            
            # Calculate pretrained accuracy
            pretrained_correct = 0
            pretrained_total = 0
            pretrained_results = []
            for src_path, (pred_spk, similarity) in pretrained_sources.items():
                # For determining expected speaker, check if source1 or source2 appears in the path
                is_source1 = "source1" in Path(src_path).name
                expected_spk = speaker1_id if is_source1 else speaker2_id
                is_correct = pred_spk == expected_spk
                if is_correct:
                    pretrained_correct += 1
                pretrained_total += 1
                pretrained_results.append({
                    'source_path': src_path,
                    'predicted_speaker': pred_spk,
                    'expected_speaker': expected_spk,
                    'similarity': similarity,
                    'is_correct': is_correct
                })
            
            pretrained_acc = pretrained_correct / pretrained_total if pretrained_total > 0 else 0
            
            # Save detailed speaker identification results
            pretrained_results_df = pd.DataFrame(pretrained_results)
            pretrained_results_df.to_csv(output_dir / f"pretrained_identification_{mixture_id}.csv", index=False)
            
            # Identify speakers with finetuned model if available
            finetuned_acc = None
            finetuned_results = []
            if finetuned_model is not None:
                finetuned_sources = identify_speakers(
                    finetuned_model, 
                    [source1_separated, source2_separated],
                    {speaker1_id: [source1_original], 
                     speaker2_id: [source2_original]}
                )
                
                # Calculate finetuned accuracy
                finetuned_correct = 0
                finetuned_total = 0
                for src_path, (pred_spk, similarity) in finetuned_sources.items():
                    # For determining expected speaker, check if source1 or source2 appears in the path
                    is_source1 = "source1" in Path(src_path).name
                    expected_spk = speaker1_id if is_source1 else speaker2_id
                    is_correct = pred_spk == expected_spk
                    if is_correct:
                        finetuned_correct += 1
                    finetuned_total += 1
                    finetuned_results.append({
                        'source_path': src_path,
                        'predicted_speaker': pred_spk,
                        'expected_speaker': expected_spk,
                        'similarity': similarity,
                        'is_correct': is_correct
                    })
                
                finetuned_acc = finetuned_correct / finetuned_total if finetuned_total > 0 else 0
                
                # Save detailed speaker identification results
                finetuned_results_df = pd.DataFrame(finetuned_results)
                finetuned_results_df.to_csv(output_dir / f"finetuned_identification_{mixture_id}.csv", index=False)
            
            # Add to results
            result = {
                'mixture_id': mixture_id,
                'speaker1_id': speaker1_id,
                'speaker2_id': speaker2_id,
                'SDR': metrics['SDR'],
                'SIR': metrics['SIR'],
                'SAR': metrics['SAR'],
                'PESQ': metrics.get('PESQ', float('nan')),
                'pretrained_speaker_acc': pretrained_acc,
                'finetuned_speaker_acc': finetuned_acc
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing mixture {mixture_id}: {e}")
            # Add empty result
            results.append({
                'mixture_id': mixture_id,
                'speaker1_id': row.get('speaker1_id', None),
                'speaker2_id': row.get('speaker2_id', None),
                'SDR': float('nan'),
                'SIR': float('nan'),
                'SAR': float('nan'),
                'PESQ': float('nan'),
                'pretrained_speaker_acc': 0.0,
                'finetuned_speaker_acc': 0.0 if finetuned_model else None
            })
    
    if not results:
        logger.error("No results were generated. Check the input files and paths.")
        return None, None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = {
        'SDR': np.nanmean(results_df['SDR']),
        'SIR': np.nanmean(results_df['SIR']),
        'SAR': np.nanmean(results_df['SAR']),
        'PESQ': np.nanmean(results_df['PESQ']),
        'pretrained_speaker_acc': np.nanmean(results_df['pretrained_speaker_acc']),
    }
    
    if finetuned_model is not None:
        avg_metrics['finetuned_speaker_acc'] = np.nanmean(results_df['finetuned_speaker_acc'])
    
    # Log average metrics
    logger.info(f"Average metrics:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    
    # Save average metrics
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("Average metrics:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return results_df, avg_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process test set with separated audio files")
    parser.add_argument("--test_metadata", type=str, required=True,
                        help="Path to test metadata file")
    parser.add_argument("--separated_dir", type=str, required=True,
                        help="Directory containing separated sources")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="wavlm_base_plus",
                        help="Name of the pretrained model to use")
    
    args = parser.parse_args()
    
    # Process test set
    process_test_set(args.test_metadata, args.separated_dir, args.output_dir, args.model_name) 