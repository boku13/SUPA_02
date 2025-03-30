import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, 
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device as default_device, load_audio, setup_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

# Initialize device - this will be overridden by finetune.py when imported
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Initial pretrained_eval device setting: {device}")

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length audio inputs.
    Process each item individually without trying to stack tensors of different lengths.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Dictionary with batched labels and lists of audio samples
    """
    labels = torch.stack([sample['label'] for sample in batch])
    audio1_samples = [sample['audio1'] for sample in batch]
    audio2_samples = [sample['audio2'] for sample in batch]
    
    return {
        'label': labels,
        'audio1': audio1_samples,
        'audio2': audio2_samples
    }

class VoxCelebTrialDataset(Dataset):
    """
    Dataset for VoxCeleb trial verification
    """
    def __init__(self, trial_file, audio_root, feature_extractor=None):
        """
        Initialize dataset
        
        Args:
            trial_file (str): Path to trial file with format "label speaker1/file1 speaker2/file2"
            audio_root (str): Root directory containing audio files
            feature_extractor: Feature extractor for the model
        """
        self.audio_root = audio_root
        self.feature_extractor = feature_extractor
        
        # Parse trial file
        self.trials = []
        with open(trial_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    # Format: label file1 file2
                    label = int(parts[0])
                    file1 = os.path.join(audio_root, parts[1])
                    file2 = os.path.join(audio_root, parts[2])
                    self.trials.append((label, file1, file2))
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        label, file1, file2 = self.trials[idx]
        
        # Load audio files
        waveform1, sr1 = load_audio(file1)
        waveform2, sr2 = load_audio(file2)
        
        # Process with feature extractor if provided
        if self.feature_extractor is not None:
            inputs1 = self.feature_extractor(
                waveform1.numpy(), 
                sampling_rate=sr1, 
                return_tensors="pt"
            ).input_values.squeeze(0)
            
            inputs2 = self.feature_extractor(
                waveform2.numpy(), 
                sampling_rate=sr2, 
                return_tensors="pt"
            ).input_values.squeeze(0)
        else:
            inputs1, inputs2 = waveform1, waveform2
        
        return {
            'label': torch.tensor(label, dtype=torch.long),
            'audio1': inputs1,
            'audio2': inputs2
        }

def load_pretrained_model(model_name):
    """
    Load a pretrained model and its feature extractor.
    
    Args:
        model_name (str): Name of the pretrained model to load.
            Options: 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', 'wavlm_base_plus'
            
    Returns:
        model: The pretrained model
        feature_extractor: The corresponding feature extractor
    """
    global device
    
    # Log current device setting
    logger.info(f"Using device for model loading: {device}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    
    model_map = {
        'hubert_large': 'facebook/hubert-large-ll60k',
        'wav2vec2_xlsr': 'facebook/wav2vec2-large-xlsr-53',
        'unispeech_sat': 'microsoft/unispeech-sat-base-plus',
        'wavlm_base_plus': 'microsoft/wavlm-base-plus'
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_map.keys())}")
    
    # Get pretrained model
    pretrained_name = model_map[model_name]
    logger.info(f"Loading pretrained model: {pretrained_name}")
    
    # Load model and feature extractor
    model = AutoModel.from_pretrained(pretrained_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    
    # Move model to correct device
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    return model, feature_extractor

def extract_embeddings(model, inputs, input_lengths=None):
    """
    Extract embeddings from a model.
    
    Args:
        model: The model to extract embeddings from
        inputs: Input tensor or dict
        input_lengths: Lengths of input sequences
        
    Returns:
        embeddings: Extracted embeddings
    """
    # Move inputs to device
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Forward pass
        outputs = model(inputs) if isinstance(inputs, dict) else model(inputs)
        
        # Get embeddings - most models use different output formats
        if hasattr(outputs, 'last_hidden_state'):
            # Average pooling over time dimension
            if input_lengths is not None:
                # Use input lengths for masked mean
                masks = torch.arange(outputs.last_hidden_state.size(1)).expand(
                    outputs.last_hidden_state.size(0), -1
                ).to(device) < input_lengths.unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * masks.unsqueeze(-1)).sum(1) / input_lengths.unsqueeze(-1)
            else:
                # Simple mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            raise ValueError("Unsupported model output format")
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def compute_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding tensor
        emb2: Second embedding tensor
        
    Returns:
        similarity: Cosine similarity score
    """
    # Normalize embeddings
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(emb1 * emb2, dim=1)
    
    return similarity

def calculate_eer(labels, scores):
    """
    Calculate the Equal Error Rate (EER).
    
    Args:
        labels: Ground truth labels (0/1)
        scores: Similarity scores
        
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    # Sort scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Calculate False Accept Rate (FAR) and False Reject Rate (FRR)
    far = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)
    frr = 1 - np.cumsum(sorted_labels) / np.sum(sorted_labels)
    
    # Find the point where FAR = FRR
    min_diff_idx = np.argmin(np.abs(far - frr))
    eer = (far[min_diff_idx] + frr[min_diff_idx]) / 2
    threshold = sorted_scores[min_diff_idx]
    
    return eer, threshold

def calculate_tar_at_far(labels, scores, far_target=0.01):
    """
    Calculate True Accept Rate (TAR) at a specific False Accept Rate (FAR).
    
    Args:
        labels: Ground truth labels (0/1)
        scores: Similarity scores
        far_target: Target FAR value (default: 0.01 for 1% FAR)
        
    Returns:
        tar: True Accept Rate at target FAR
        threshold: Threshold at target FAR
    """
    # Sort scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Calculate False Accept Rate (FAR) and True Accept Rate (TAR)
    far = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)
    tar = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    
    # Find the point where FAR <= far_target
    far_target_idx = np.argmax(far >= far_target) - 1
    if far_target_idx < 0:
        far_target_idx = 0
    
    tar_at_far = tar[far_target_idx]
    threshold = sorted_scores[far_target_idx]
    
    return tar_at_far, threshold

def evaluate_pretrained_model(model_name, trial_file, audio_root, batch_size=8, max_trials=4000, saved_subset_path=None):
    """
    Evaluate a pretrained model on speaker verification task.
    
    Args:
        model_name: Name of the pretrained model to evaluate
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        batch_size: Batch size for evaluation
        max_trials: Maximum number of trials to evaluate (default: 4000)
        saved_subset_path: Path to save/load the subset of trial pairs (if None, will be derived from trial_file)
        
    Returns:
        results: Dictionary with evaluation results
    """
    logger.info(f"Evaluating pretrained model: {model_name}")
    
    # Determine the path to save/load the subset if not provided
    if saved_subset_path is None:
        saved_subset_path = os.path.join(os.path.dirname(trial_file), f"trial_subset_{max_trials}.txt")
    
    # Check if a saved subset already exists
    if os.path.exists(saved_subset_path):
        logger.info(f"Loading existing trial subset from {saved_subset_path}")
        with open(saved_subset_path, 'r') as f:
            trial_pairs = f.readlines()
        logger.info(f"Loaded {len(trial_pairs)} trial pairs from existing subset")
    else:
        # Load all trial pairs from the original file and limit to max_trials
        logger.info(f"Creating new trial subset from {trial_file}")
        with open(trial_file, 'r') as f:
            all_trial_pairs = f.readlines()
        
        # Limit to the first max_trials pairs
        trial_pairs = all_trial_pairs[:max_trials]
        logger.info(f"Limited evaluation to {len(trial_pairs)} trial pairs (from {len(all_trial_pairs)} total)")
        
        # Save the subset for future use
        with open(saved_subset_path, 'w') as f:
            f.writelines(trial_pairs)
        logger.info(f"Saved trial subset to {saved_subset_path} for consistent evaluation")
    
    # Create dataset and dataloader with custom collate function
    dataset = VoxCelebTrialDataset(saved_subset_path, audio_root, None)  # We'll process inputs separately
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Reduce worker count to avoid potential issues
        collate_fn=custom_collate_fn  # Use our custom collate function
    )
    
    # Load model and feature extractor
    model, feature_extractor = load_pretrained_model(model_name)
    model.eval()
    
    # Collect all labels and scores
    all_labels = []
    all_scores = []
    
    logger.info("Processing trial pairs...")
    for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
        labels = batch['label']
        audio1_samples = batch['audio1']  # List of audio samples
        audio2_samples = batch['audio2']  # List of audio samples
        
        # Process each sample individually
        batch_scores = []
        for i in range(len(audio1_samples)):
            # Extract embeddings for each pair
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
        
        # Collect labels and scores
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(batch_scores)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far, far_threshold = calculate_tar_at_far(all_labels, all_scores, far_target=0.01)
    
    # Calculate speaker identification accuracy (using optimal threshold)
    pred_labels = (all_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)
    
    results = {
        'model_name': model_name,
        'eer': eer * 100,  # Convert to percentage
        'tar_at_1far': tar_at_far * 100,  # Convert to percentage
        'accuracy': accuracy * 100,  # Convert to percentage
        'eer_threshold': eer_threshold,
        'optimal_far_threshold': far_threshold,
        'num_trials': len(all_labels),
        'trial_subset_path': saved_subset_path  # Save the path to the subset
    }
    
    logger.info(f"Evaluation results for {model_name} on {len(all_labels)} trials:")
    logger.info(f"  Trial subset: {saved_subset_path}")
    logger.info(f"  EER: {eer*100:.2f}%")
    logger.info(f"  TAR@1%FAR: {tar_at_far*100:.2f}%")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    
    return results

def evaluate_all_models(trial_file, audio_root, max_trials=4000):
    """
    Evaluate all pretrained models on speaker verification task.
    
    Args:
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        max_trials: Maximum number of trials to evaluate
        
    Returns:
        results_df: DataFrame with results for all models
    """
    model_names = ['hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', 'wavlm_base_plus']
    results = []
    
    # Create a consistent saved subset path
    saved_subset_path = os.path.join(os.path.dirname(trial_file), f"trial_subset_{max_trials}.txt")
    
    for model_name in model_names:
        result = evaluate_pretrained_model(model_name, trial_file, audio_root, max_trials=max_trials, 
                                          saved_subset_path=saved_subset_path)
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate pretrained models on speaker verification")
    parser.add_argument("--trial_file", type=str, required=True, help="Path to trial file")
    parser.add_argument("--audio_root", type=str, required=True, help="Root directory for audio files")
    parser.add_argument("--model_name", type=str, default=None, 
                        choices=['hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', 'wavlm_base_plus'],
                        help="Model to evaluate (default: evaluate all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_trials", type=int, default=4000, help="Maximum number of trial pairs to evaluate")
    parser.add_argument("--saved_subset_path", type=str, default=None, 
                        help="Path to save/load the subset of trial pairs")
    parser.add_argument("--output_file", type=str, default="pretrained_results.csv", 
                        help="Output file for results")
    
    args = parser.parse_args()
    
    if args.model_name:
        # Evaluate single model
        result = evaluate_pretrained_model(
            args.model_name, 
            args.trial_file, 
            args.audio_root, 
            args.batch_size,
            max_trials=args.max_trials,
            saved_subset_path=args.saved_subset_path
        )
        results_df = pd.DataFrame([result])
    else:
        # Evaluate all models
        results_df = evaluate_all_models(
            args.trial_file, 
            args.audio_root,
            max_trials=args.max_trials
        )
    
    # Save results
    results_df.to_csv(args.output_file, index=False)
    logger.info(f"Results saved to {args.output_file}")
    
    # Also save the subset path info for reference
    subset_path = args.saved_subset_path if args.saved_subset_path else os.path.join(
        os.path.dirname(args.trial_file), 
        f"trial_subset_{args.max_trials}.txt"
    )
    logger.info(f"Trial subset used: {subset_path}")
    logger.info(f"To ensure consistent evaluation, use this subset path in future runs:") 