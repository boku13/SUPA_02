# %%
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Replace tqdm.notebook with a more robust import
try:
    # Try to use tqdm.notebook for Jupyter environments
    from tqdm.notebook import tqdm
except ImportError:
    # Fall back to regular tqdm for terminal environments
    from tqdm import tqdm
import logging
import random
import time
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoFeatureExtractor, 
    AutoProcessor,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# %%
# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# %%
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# %%
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# %%
# Set seed for reproducibility
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# %%
# Define paths - validate they exist
BASE_DIR = Path("data")
VOX1_DIR = BASE_DIR / "vox1"
VOX2_DIR = BASE_DIR / "vox2"
VOX1_WAV_DIR = VOX1_DIR / "wav"
VOX2_AAC_DIR = VOX2_DIR / "aac"
VOX2_TXT_DIR = VOX2_DIR / "txt"
TRIAL_FILE = VOX1_DIR / "veri_test2.txt"
OUTPUT_DIR = Path("models/speaker_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
def validate_paths():
    """Validate that all necessary paths exist"""
    paths = [VOX1_DIR, VOX2_DIR, VOX1_WAV_DIR, VOX2_AAC_DIR, VOX2_TXT_DIR, TRIAL_FILE]
    for path in paths:
        if not path.exists():
            logger.error(f"Path does not exist: {path}")
            if "vox" in str(path).lower():
                logger.error("Please ensure the VoxCeleb datasets are correctly downloaded and extracted")
            return False
    return True

# %%
def load_audio(file_path):
    """
    Load audio file, converting from .m4a to waveform if needed
    
    Args:
        file_path: Path to the audio file (.wav or .m4a)
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    import librosa
    import torchaudio
    
    file_path = str(file_path)
    target_sr = 16000  # Target sample rate
    
    try:
        if file_path.endswith('.m4a'):
            # Load m4a files using librosa
            waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform).float()
        else:
            # Load wav files using torchaudio
            try:
                waveform, sr = torchaudio.load(file_path)
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                waveform = waveform.squeeze(0)
                # Resample if needed
                if sr != target_sr:
                    waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            except Exception as e:
                logger.warning(f"Error loading with torchaudio, falling back to librosa: {e}")
                waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
                waveform = torch.from_numpy(waveform).float()
    except Exception as e:
        logger.warning(f"Error loading audio {file_path}: {e}")
        # Return a short silent audio in case of error
        waveform = torch.zeros(target_sr)
        sr = target_sr
    
    return waveform, sr

# %%
def plot_waveform(waveform, sr, title="Waveform"):
    """Plot a waveform"""
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.numpy())
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

# %%
def save_audio(waveform, file_path, sample_rate=16000):
    """Save audio to file."""
    waveform = waveform.detach().cpu().numpy()
    if waveform.ndim > 1:
        waveform = waveform.squeeze(0)  # Remove batch dimension
    sf.write(file_path, waveform, sample_rate)
    return file_path

# %%
def get_speaker_ids(directory):
    """
    Get list of speaker IDs from a directory
    
    Args:
        directory: Path to directory containing speaker folders
        
    Returns:
        list: Sorted list of speaker IDs
    """
    try:
        # Check if directory exists
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        # List all directories (speaker IDs)
        speaker_ids = []
        for item in os.listdir(directory):
            item_path = directory / item
            if item_path.is_dir():
                speaker_ids.append(item)
        
        return sorted(speaker_ids)
    except Exception as e:
        logger.error(f"Error getting speaker IDs from {directory}: {e}")
        return []

# %%
def create_voxceleb2_metadata(speaker_ids, output_file):
    """
    Create metadata file for VoxCeleb2 dataset
    
    Args:
        speaker_ids: List of speaker IDs to include
        output_file: Path to save the metadata file
        
    Returns:
        pd.DataFrame: Metadata dataframe
    """
    metadata = []
    
    for speaker_id in tqdm(speaker_ids, desc="Processing speakers"):
        speaker_dir = VOX2_AAC_DIR / speaker_id
        
        # Skip if directory doesn't exist
        if not speaker_dir.exists():
            continue
        
        # Get all m4a files for this speaker
        for session_dir in speaker_dir.iterdir():
            if session_dir.is_dir():
                for audio_file in session_dir.glob("*.m4a"):
                    # Get relative path
                    rel_path = os.path.relpath(audio_file, VOX2_AAC_DIR)
                    
                    # Add to metadata
                    metadata.append({
                        'id': speaker_id,
                        'path': rel_path,
                        'gender': 'unknown'  # Gender information not available
                    })
    
    # Create dataframe
    df = pd.DataFrame(metadata)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, sep=' ', index=False, header=False)
    
    logger.info(f"Created metadata file with {len(df)} entries at {output_file}")
    
    return df

# %%
def create_train_test_split(metadata_file, train_ids, test_ids, train_file, test_file):
    """
    Create train/test split from metadata
    
    Args:
        metadata_file: Path to metadata file
        train_ids: List of speaker IDs for training
        test_ids: List of speaker IDs for testing
        train_file: Output path for training metadata
        test_file: Output path for test metadata
        
    Returns:
        tuple: (train_df, test_df)
    """
    # Check if metadata file exists
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return None, None
    
    # Read metadata
    try:
        # Try reading as CSV
        metadata = pd.read_csv(metadata_file, sep=' ', names=['id', 'path', 'gender'])
    except:
        # If file is not a proper CSV, read line by line
        data = []
        with open(metadata_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    speaker_id = parts[0]
                    path = parts[1]
                    gender = parts[2] if len(parts) > 2 else "unknown"
                    data.append([speaker_id, path, gender])
        metadata = pd.DataFrame(data, columns=['id', 'path', 'gender'])
    
    # Split by speaker ID
    train_df = metadata[metadata['id'].isin(train_ids)]
    test_df = metadata[metadata['id'].isin(test_ids)]
    
    # Save splits
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    train_df.to_csv(train_file, sep=' ', index=False, header=False)
    test_df.to_csv(test_file, sep=' ', index=False, header=False)
    
    logger.info(f"Created train split with {len(train_df)} files from {len(train_ids)} speakers")
    logger.info(f"Created test split with {len(test_df)} files from {len(test_ids)} speakers")
    
    return train_df, test_df

# %%
# Speaker Verification Datasets
# ============================

class VoxCelebTrialDataset(Dataset):
    """Dataset for VoxCeleb trial verification"""
    def __init__(self, trial_file, audio_root, feature_extractor=None):
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
        
        logger.info(f"Loaded {len(self.trials)} trials from {trial_file}")
    
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

# %%
def custom_collate_fn(batch):
    """Custom collate function to handle variable-length audio inputs"""
    labels = torch.stack([sample['label'] for sample in batch])
    audio1_samples = [sample['audio1'] for sample in batch]
    audio2_samples = [sample['audio2'] for sample in batch]
    
    return {
        'label': labels,
        'audio1': audio1_samples,
        'audio2': audio2_samples
    }

# %%
class VoxCelebSpeakerDataset(Dataset):
    """Dataset for VoxCeleb speaker identification"""
    def __init__(self, metadata_file, audio_root, processor=None, max_duration=5):
        self.audio_root = audio_root
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = 16000
        
        # Read metadata
        self.df = []
        try:
            # Try to read as CSV
            self.df = pd.read_csv(metadata_file, sep=' ', names=['id', 'path', 'gender']).to_dict('records')
        except:
            # If fails, read line by line
            with open(metadata_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        speaker_id = parts[0]
                        path = parts[1]
                        gender = parts[2] if len(parts) > 2 else "unknown"
                        self.df.append({'id': speaker_id, 'path': path, 'gender': gender})
        
        # Get unique speaker IDs and create label mapping
        self.speakers = sorted(set(item['id'] for item in self.df))
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(self.speakers)}
        
        logger.info(f"Loaded dataset with {len(self.df)} utterances from {len(self.speakers)} speakers")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df[idx]
        speaker_id = row['id']
        path = row['path']
        
        # Form full audio path
        audio_path = os.path.join(self.audio_root, path)
        
        # Load audio
        try:
            waveform, sr = load_audio(audio_path)
            
            # Cut if too long
            max_samples = int(self.max_duration * self.sample_rate)
            if len(waveform) > max_samples:
                start = torch.randint(0, len(waveform) - max_samples, (1,)).item()
                waveform = waveform[start:start+max_samples]
            
            # Process with processor if available
            if self.processor is not None:
                inputs = self.processor(
                    waveform.numpy(), 
                    sampling_rate=sr, 
                    return_tensors="pt"
                ).input_values.squeeze(0)
            else:
                inputs = waveform
                
            label = self.speaker_to_idx[speaker_id]
            
            return {
                'audio': inputs,
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return a placeholder in case of error
            dummy_audio = torch.zeros(self.sample_rate)
            return {
                'audio': dummy_audio,
                'label': torch.tensor(0, dtype=torch.long)
            }

# %%
# Speaker Verification Models
# ==========================

class ArcFaceLayer(nn.Module):
    """Implementation of ArcFace loss layer"""
    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.5):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # Normalize weights
        weights_norm = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(F.normalize(features, dim=1), weights_norm)
        
        if labels is None:
            return cos_theta * self.scale_factor
            
        # Clip cosine values to valid range
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Get angle
        theta = torch.acos(cos_theta)
        
        # Add margin to target classes
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        theta += one_hot * self.margin
        
        # Convert back to cosine
        output = torch.cos(theta)
        
        # Scale for better convergence
        output *= self.scale_factor
        
        return output

# %%
def load_pretrained_model(model_name):
    """
    Load a pretrained model and its feature extractor.
    
    Args:
        model_name (str): Name of the pretrained model to load.
            Options: 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', 'wavlm_base_plus'
    """
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
    
    # Move model to device
    model = model.to(device)
    
    return model, feature_extractor

# %%
class SpeakerVerificationModel(nn.Module):
    """Speaker verification model with a pretrained backbone and ArcFace head"""
    def __init__(self, pretrained_model, embedding_dim, num_speakers):
        super(SpeakerVerificationModel, self).__init__()
        self.backbone = pretrained_model
        self.arcface = ArcFaceLayer(embedding_dim, num_speakers)
        
    def forward(self, x, labels=None):
        # Extract embeddings from backbone
        outputs = self.backbone(x)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Apply ArcFace classifier if labels are provided
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
        
        # For inference, just return embeddings
        return embeddings

# %%
def extract_embeddings(model, inputs):
    """
    Extract embeddings from a model
    
    Args:
        model: The model to extract embeddings from
        inputs: Input tensor
        
    Returns:
        torch.Tensor: Embeddings
    """
    # Move inputs to device
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(inputs)
        
        # Get embeddings
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            raise ValueError("Unsupported model output format")
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

# %%
def compute_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        
    Returns:
        torch.Tensor: Similarity score
    """
    # Normalize embeddings
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(emb1 * emb2, dim=1)
    
    return similarity

# %%
# Evaluation Metrics
# =================

def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate (EER)
    
    Args:
        labels: Ground truth labels (0/1)
        scores: Similarity scores
        
    Returns:
        tuple: (eer, threshold)
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

# %%
def calculate_tar_at_far(labels, scores, far_target=0.01):
    """
    Calculate True Accept Rate at a specific False Accept Rate
    
    Args:
        labels: Ground truth labels (0/1)
        scores: Similarity scores
        far_target: Target FAR value
        
    Returns:
        tuple: (tar, threshold)
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

# %%
# Speaker Verification Evaluation
# ==============================

def evaluate_pretrained_model(model_name, trial_file, audio_root, batch_size=8, max_samples=100):
    """
    Evaluate a pretrained model on speaker verification task
    
    Args:
        model_name: Name of the pretrained model
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (for quicker testing)
        
    Returns:
        dict: Evaluation results
    """
    start_time = time.time()
    logger.info(f"Evaluating pretrained model: {model_name}")
    
    # Load model and feature extractor
    logger.info(f"Loading model {model_name}...")
    model, feature_extractor = load_pretrained_model(model_name)
    model.eval()
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Create dataset and dataloader
    logger.info(f"Creating dataset from {trial_file}...")
    dataset = VoxCelebTrialDataset(trial_file, audio_root, feature_extractor)
    
    # For quick testing, limit the number of samples
    if max_samples > 0 and max_samples < len(dataset):
        logger.info(f"Limiting evaluation to {max_samples} samples for quicker testing")
        # Create a subset of the dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        
        # Create a new dataset with only these samples
        subset_trials = [dataset.trials[i] for i in indices]
        dataset.trials = subset_trials
        
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"Dataset created with {len(dataset)} trial pairs")
    
    # Collect all labels and scores
    all_labels = []
    all_scores = []
    
    logger.info("Processing trial pairs...")
    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        labels = batch['label']
        audio1_samples = batch['audio1']
        audio2_samples = batch['audio2']
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(labels)} samples")
        
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
        
        # Log progress
        batch_time = time.time() - batch_start
        logger.info(f"Batch {batch_idx+1}/{total_batches} processed in {batch_time:.2f} seconds")
        logger.info(f"Completed {batch_idx+1}/{total_batches} batches ({(batch_idx+1)/total_batches*100:.1f}%)")
        
        # Estimate remaining time
        if batch_idx > 0:
            avg_time_per_batch = batch_time
            remaining_batches = total_batches - (batch_idx + 1)
            est_time_remaining = avg_time_per_batch * remaining_batches
            logger.info(f"Estimated time remaining: {est_time_remaining:.2f} seconds")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
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
        'optimal_far_threshold': far_threshold
    }
    
    logger.info(f"Evaluation results for {model_name}:")
    logger.info(f"  EER: {eer*100:.2f}%")
    logger.info(f"  TAR@1%FAR: {tar_at_far*100:.2f}%")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    
    # Visualize distribution of scores
    logger.info("Creating score distribution visualization...")
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores[all_labels == 1], bins=50, alpha=0.5, label='Same Speaker')
    plt.hist(all_scores[all_labels == 0], bins=50, alpha=0.5, label='Different Speakers')
    plt.axvline(x=eer_threshold, color='r', linestyle='--', label=f'EER Threshold: {eer_threshold:.3f}')
    plt.axvline(x=far_threshold, color='g', linestyle='--', label=f'FAR@1% Threshold: {far_threshold:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution - {model_name}')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, f"{model_name}_score_distribution.png")
    plt.savefig(plot_path)
    logger.info(f"Score distribution plot saved to {plot_path}")
    
    plt.show()
    
    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return results

# %%
# Model Fine-tuning
# ================

def apply_lora(model):
    """
    Apply LoRA adapters to the model
    
    Args:
        model: Model to apply LoRA to
        
    Returns:
        model: Model with LoRA adapters
    """
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"LoRA applied to model")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%} of total)")
    
    return model

# %%
def train_model(model_name, train_metadata, val_metadata, audio_root, 
                output_dir, batch_size=8, epochs=5, learning_rate=5e-5):
    """
    Fine-tune a speaker verification model using LoRA and ArcFace loss
    
    Args:
        model_name: Name of the pretrained model
        train_metadata: Path to training metadata file
        val_metadata: Path to validation metadata file
        audio_root: Root directory for audio files
        output_dir: Directory to save the fine-tuned model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        model: Fine-tuned model
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model and processor
    pretrained_model, processor = load_pretrained_model(model_name)
    
    # Create datasets
    train_dataset = VoxCelebSpeakerDataset(train_metadata, audio_root, processor)
    val_dataset = VoxCelebSpeakerDataset(val_metadata, audio_root, processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Create model
    model = SpeakerVerificationModel(
        pretrained_model, 
        768,  # Embedding dimension
        len(train_dataset.speakers)
    ).to(device)
    
    # Apply LoRA
    model = apply_lora(model)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            inputs = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(inputs, labels)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
            train_loss /= len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in progress_bar:
                    inputs = batch['audio'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits, _ = model(inputs, labels)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
                    
                    progress_bar.set_postfix({'loss': loss.item()})
            
            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logger.info(f"  Saved best model with val_loss: {val_loss:.4f}")
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train')
        plt.plot(val_accuracies, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png")
        plt.show()
        
        # Load best model
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        
        return model

# %%
def evaluate_finetuned_model(model, trial_file, audio_root, batch_size=8, max_samples=100):
    """
    Evaluate a fine-tuned model on speaker verification task
    
    Args:
        model: Fine-tuned speaker verification model
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (for quicker testing)
        
    Returns:
        dict: Evaluation results
    """
    start_time = time.time()
    logger.info("Evaluating fine-tuned model")
    
    # Get feature extractor for the same model architecture
    if hasattr(model.backbone, 'config'):
        model_name = model.backbone.config.model_type
        if 'hubert' in model_name:
            model_name = 'hubert_large'
        elif 'wav2vec2' in model_name:
            model_name = 'wav2vec2_xlsr'
        elif 'unispeech' in model_name:
            model_name = 'unispeech_sat'
        elif 'wavlm' in model_name:
            model_name = 'wavlm_base_plus'
    else:
        model_name = 'wavlm_base_plus'  # Default to WavLM if model type can't be determined
    
    logger.info(f"Using feature extractor for {model_name}")
    _, feature_extractor = load_pretrained_model(model_name)
    
    # Create dataset and dataloader
    logger.info(f"Creating dataset from {trial_file}...")
    dataset = VoxCelebTrialDataset(trial_file, audio_root, feature_extractor)
    
    # For quick testing, limit the number of samples
    if max_samples > 0 and max_samples < len(dataset):
        logger.info(f"Limiting evaluation to {max_samples} samples for quicker testing")
        # Create a subset of the dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        
        # Create a new dataset with only these samples
        subset_trials = [dataset.trials[i] for i in indices]
        dataset.trials = subset_trials
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"Dataset created with {len(dataset)} trial pairs")
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect all labels and scores
    all_labels = []
    all_scores = []
    
    logger.info("Processing trial pairs...")
    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        labels = batch['label']
        audio1_samples = batch['audio1']
        audio2_samples = batch['audio2']
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(labels)} samples")
        
        # Process each sample individually
        batch_scores = []
        for i in range(len(audio1_samples)):
            # Extract embeddings for each pair
            with torch.no_grad():
                # Process first audio
                audio1 = audio1_samples[i].unsqueeze(0).to(device)
                emb1 = model(audio1)  # For fine-tuned model, this returns embeddings directly
                
                # Process second audio
                audio2 = audio2_samples[i].unsqueeze(0).to(device)
                emb2 = model(audio2)
                
                # Compute similarity
                score = compute_similarity(emb1, emb2).item()
                batch_scores.append(score)
        
        # Collect labels and scores
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(batch_scores)
        
        # Log progress
        batch_time = time.time() - batch_start
        logger.info(f"Batch {batch_idx+1}/{total_batches} processed in {batch_time:.2f} seconds")
        logger.info(f"Completed {batch_idx+1}/{total_batches} batches ({(batch_idx+1)/total_batches*100:.1f}%)")
        
        # Estimate remaining time
        if batch_idx > 0:
            avg_time_per_batch = batch_time
            remaining_batches = total_batches - (batch_idx + 1)
            est_time_remaining = avg_time_per_batch * remaining_batches
            logger.info(f"Estimated time remaining: {est_time_remaining:.2f} seconds")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far, far_threshold = calculate_tar_at_far(all_labels, all_scores, far_target=0.01)
    
    # Calculate speaker identification accuracy (using optimal threshold)
    pred_labels = (all_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)
    
    results = {
        'model_name': 'Fine-tuned Model',
        'eer': eer * 100,  # Convert to percentage
        'tar_at_1far': tar_at_far * 100,  # Convert to percentage
        'accuracy': accuracy * 100,  # Convert to percentage
        'eer_threshold': eer_threshold,
        'optimal_far_threshold': far_threshold
    }
    
    logger.info(f"Evaluation results for fine-tuned model:")
    logger.info(f"  EER: {eer*100:.2f}%")
    logger.info(f"  TAR@1%FAR: {tar_at_far*100:.2f}%")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    
    # Visualize distribution of scores
    logger.info("Creating score distribution visualization...")
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores[all_labels == 1], bins=50, alpha=0.5, label='Same Speaker')
    plt.hist(all_scores[all_labels == 0], bins=50, alpha=0.5, label='Different Speakers')
    plt.axvline(x=eer_threshold, color='r', linestyle='--', label=f'EER Threshold: {eer_threshold:.3f}')
    plt.axvline(x=far_threshold, color='g', linestyle='--', label=f'FAR@1% Threshold: {far_threshold:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.title('Score Distribution - Fine-tuned Model')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, "finetuned_model_score_distribution.png")
    plt.savefig(plot_path)
    logger.info(f"Score distribution plot saved to {plot_path}")
    
    plt.show()
    
    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return results

# %%
def compare_models(pretrained_results, finetuned_results):
    """
    Compare pretrained and fine-tuned model performance
    
    Args:
        pretrained_results: Evaluation results for pretrained model
        finetuned_results: Evaluation results for fine-tuned model
        
    Returns:
        DataFrame: Comparison of results
    """
    # Create comparison dataframe
    comparison = pd.DataFrame([
        pretrained_results,
        finetuned_results
    ])
    
    # Display results
    logger.info("\nModel Comparison:")
    logger.info(comparison[['model_name', 'eer', 'tar_at_1far', 'accuracy']])
    
    # Save comparison to CSV
    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Create comparison chart
    metrics = ['EER (%)', 'TAR@1%FAR (%)', 'Accuracy (%)']
    values_pretrained = [
        pretrained_results['eer'],
        pretrained_results['tar_at_1far'], 
        pretrained_results['accuracy']
    ]
    values_finetuned = [
        finetuned_results['eer'], 
        finetuned_results['tar_at_1far'], 
        finetuned_results['accuracy']
    ]
    
    # For EER, lower is better, so invert it for visualization
    values_pretrained[0] = 100 - values_pretrained[0]
    values_finetuned[0] = 100 - values_finetuned[0]
    metrics[0] = 'Inverted EER (higher is better)'
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, values_pretrained, width, label='Pretrained')
    rects2 = ax.bar(x + width/2, values_finetuned, width, label='Fine-tuned')
    
    ax.set_ylabel('Performance (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    # Save comparison plot
    plot_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(plot_path)
    logger.info(f"Model comparison plot saved to {plot_path}")
    
    plt.show()
    
    # Show improvements
    eer_improvement = pretrained_results['eer'] - finetuned_results['eer']
    tar_improvement = finetuned_results['tar_at_1far'] - pretrained_results['tar_at_1far'] 
    acc_improvement = finetuned_results['accuracy'] - pretrained_results['accuracy']
    
    logger.info("\nImprovements with Fine-tuning:")
    logger.info(f"  EER: {eer_improvement:.2f}% decrease (lower is better)")
    logger.info(f"  TAR@1%FAR: {tar_improvement:.2f}% increase")
    logger.info(f"  Accuracy: {acc_improvement:.2f}% increase")
    
    return comparison

# %%
# Single Audio Inference Demo
# ==========================

def demo_inference(model, audio_path1, audio_path2=None):
    """
    Demonstrate inference with a single audio file or pair
    
    Args:
        model: Speaker verification model
        audio_path1: Path to first audio file
        audio_path2: Path to second audio file (optional)
        
    Returns:
        tuple: (embedding1, embedding2, similarity) if audio_path2 is provided,
               embedding1 otherwise
    """
    # Load audio
    waveform1, sr1 = load_audio(audio_path1)
    
    # Plot waveform
    plot_waveform(waveform1, sr1, title=f"Waveform: {os.path.basename(audio_path1)}")
    
    # Extract embedding
    model.eval()
    with torch.no_grad():
        embedding1 = model(waveform1.unsqueeze(0).to(device))
    
    logger.info(f"Extracted embedding from {os.path.basename(audio_path1)}")
    logger.info(f"Embedding shape: {embedding1.shape}")
    logger.info(f"Embedding norm: {torch.norm(embedding1).item():.4f}")
    
    # If second audio file is provided, compare them
    if audio_path2 is not None:
        # Load audio
        waveform2, sr2 = load_audio(audio_path2)
        
        # Plot waveform
        plot_waveform(waveform2, sr2, title=f"Waveform: {os.path.basename(audio_path2)}")
        
        # Extract embedding
        with torch.no_grad():
            embedding2 = model(waveform2.unsqueeze(0).to(device))
        
        logger.info(f"Extracted embedding from {os.path.basename(audio_path2)}")
        logger.info(f"Embedding shape: {embedding2.shape}")
        logger.info(f"Embedding norm: {torch.norm(embedding2).item():.4f}")
        
        # Compute similarity
        similarity = compute_similarity(embedding1, embedding2).item()
        logger.info(f"Similarity between the two audios: {similarity:.4f}")
        
        return embedding1, embedding2, similarity
    
    return embedding1

# %%
# Fix Jupyter cell magic commands that are causing linter errors
# Instead of using magic commands, use regular Python code

# Install required packages
# Uncomment and run this in the first cell of your notebook if needed
# import subprocess
# subprocess.check_call(["pip", "install", "--upgrade", "jupyter", "ipywidgets"])
# subprocess.check_call(["jupyter", "nbextension", "enable", "--py", "widgetsnbextension"])

# %%
# Set model name
model_name = 'wavlm_base_plus'

# %%
# 1. Evaluate pretrained model with limited samples first for quick testing
logger.info("Step 1: Evaluating pretrained model with limited samples")
pretrained_results_quick = evaluate_pretrained_model(
    model_name=model_name,
    trial_file=TRIAL_FILE,
    audio_root=VOX1_WAV_DIR,
    batch_size=8,
    max_samples=100  # Use a small subset for quick testing
)

# %%
# Run full evaluation if the quick test was successful
logger.info("Step 2: Running full evaluation (this may take a while)")
pretrained_results = evaluate_pretrained_model(
    model_name=model_name,
    trial_file=TRIAL_FILE,
    audio_root=VOX1_WAV_DIR,
    batch_size=8,
    max_samples=-1  # Use all samples
)

# %%



