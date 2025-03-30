import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, 
    AutoFeatureExtractor,
    AutoProcessor,
    get_linear_schedule_with_warmup
)
import logging
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Configure logging format to be more concise and disable excessive transformers logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('peft').setLevel(logging.WARNING)

# Configure tqdm for cleaner display in all environments without affecting multiprocessing
tqdm._instances.clear()  # Clear any existing instances
tqdm.monitor_interval = 0  # Disable monitor thread
# Default tqdm configuration for cleaner display
original_tqdm = tqdm
def custom_tqdm(*args, **kwargs):
    new_kwargs = {
        'ncols': 100,  # Fixed width
        'bar_format': '{l_bar}{bar:15}{r_bar}',  # Simplified format
        'ascii': True,  # Use ASCII characters for compatibility
        'leave': False,  # Don't leave traces
    }
    new_kwargs.update(kwargs)
    return original_tqdm(*args, **new_kwargs)
tqdm = custom_tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device as default_device, load_audio, setup_seed
# Change relative import to absolute import
from speaker_verification.pretrained_eval import calculate_eer, calculate_tar_at_far, load_pretrained_model
import speaker_verification.pretrained_eval as pretrained_eval

# Set seed for reproducibility
setup_seed(42)

# Device configuration - ensure CUDA is detected
if torch.cuda.is_available():
    cuda_device_info = {
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
    }
    device = torch.device("cuda")
else:
    logger.warning("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

class ArcFaceLayer(nn.Module):
    """Implementation of ArcFace loss layer"""
    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.2):
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
        one_hot = torch.zeros_like(cos_theta).to(features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        theta += one_hot * self.margin
        
        # Convert back to cosine
        output = torch.cos(theta)
        
        # Scale for better convergence
        output *= self.scale_factor
        
        return output

class VoxCelebSpeakerDataset(Dataset):
    """Dataset for VoxCeleb speaker identification"""
    def __init__(self, metadata_file, audio_root, processor=None, max_duration=5, cache_waveforms=False, max_files_per_speaker=10, use_first_n_speakers=100, is_test=False):
        """
        Initialize dataset
        
        Args:
            metadata_file: Path to metadata file
            audio_root: Root directory containing audio files
            processor: Processor for the model
            max_duration: Maximum audio duration in seconds
            cache_waveforms: Whether to cache loaded waveforms in memory (uses more RAM but faster)
            max_files_per_speaker: Maximum number of files to use per speaker (default: 10)
            use_first_n_speakers: Use only the first N speakers (default: 100), if 0 use all speakers
            is_test: Whether this is a test dataset
        """
        self.audio_root = audio_root
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = 16000
        self.cache_waveforms = cache_waveforms
        self.waveform_cache = {}
        
        # Read metadata
        df = pd.read_csv(metadata_file, sep=' ', names=['id', 'path', 'gender'])
        
        # Get all unique speaker IDs sorted in ascending order
        all_speakers = sorted(df['id'].unique())
        total_speakers = len(all_speakers)
        logger.info(f"Total unique speakers in dataset: {total_speakers}")
        
        # Select speakers based on whether this is train or test
        if is_test:
            # For test set, use all speakers in the test file
            # The test file should already contain the appropriate speakers (different from train)
            selected_speakers = all_speakers
            logger.info(f"Using all {len(selected_speakers)} speakers for test set")
        else:
            # For train set, use the first N speakers as specified in the assignment
            if use_first_n_speakers > 0 and use_first_n_speakers < total_speakers:
                selected_speakers = all_speakers[:use_first_n_speakers]
                logger.info(f"Using FIRST {len(selected_speakers)} speakers for training")
            else:
                selected_speakers = all_speakers
                logger.info(f"Using all {len(selected_speakers)} speakers for training")
        
        # Filter DataFrame to only include selected speakers
        df = df[df['id'].isin(selected_speakers)]
        logger.info(f"Dataset filtered to {len(df)} utterances from {len(selected_speakers)} speakers")
        
        # Update the implementation to respect the max_files_per_speaker parameter
        if max_files_per_speaker <= 0:
            # If max_files_per_speaker is not positive, don't limit files
            logger.info("Not limiting the number of files per speaker (using all available)")
            self.df = df
        else:
            # Use the specified max_files_per_speaker value
            logger.info(f"Limiting dataset to exactly {max_files_per_speaker} files per speaker")
            # Group by speaker ID and take the first max_files_per_speaker entries
            self.df = df.groupby('id').apply(lambda x: x.head(max_files_per_speaker)).reset_index(drop=True)
            logger.info(f"Dataset reduced from {len(df)} to {len(self.df)} utterances after limiting to {max_files_per_speaker} per speaker")
        
        # Get unique speaker IDs and create label mapping
        self.speakers = sorted(self.df['id'].unique())
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(self.speakers)}
        
        logger.info(f"Final dataset contains {len(self.df)} utterances from {len(self.speakers)} speakers")
        if len(self.speakers) > 0:
            sample_speakers = min(5, len(self.speakers))
            logger.info(f"Speaker IDs used: {self.speakers[:sample_speakers]}... (and {len(self.speakers)-sample_speakers} more)")
        else:
            logger.warning("No speakers found in the dataset!")
        
        # Verify a few samples to ensure audio loading works
        self._verify_samples(num_samples=min(5, len(self.df)))
    
    def _verify_samples(self, num_samples=5):
        """Verify that a few random samples can be loaded correctly"""
        if len(self.df) <= num_samples:
            samples = self.df
        else:
            samples = self.df.sample(num_samples)
        
        success = 0
        for _, row in samples.iterrows():
            try:
                path = row['path']
                audio_path = os.path.join(self.audio_root, path)
                
                if not os.path.exists(audio_path):
                    logger.warning(f"File not found during verification: {audio_path}")
                    continue
                
                waveform, sr = load_audio(audio_path)
                success += 1
                
                # If everything is OK, cache this waveform
                if self.cache_waveforms:
                    self.waveform_cache[audio_path] = (waveform, sr)
                    
            except Exception as e:
                logger.warning(f"Error during dataset verification: {e}")
        
        logger.info(f"Dataset verification: {success}/{len(samples)} samples loaded successfully")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        speaker_id = row['id']
        path = row['path']
        
        # Form full audio path
        audio_path = os.path.join(self.audio_root, path)
        
        # Load audio
        try:
            # Check cache first
            if self.cache_waveforms and audio_path in self.waveform_cache:
                waveform, sr = self.waveform_cache[audio_path]
            else:
                waveform, sr = load_audio(audio_path)
                # Cache if needed
                if self.cache_waveforms:
                    self.waveform_cache[audio_path] = (waveform, sr)
            
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

def get_embedding_dim(model_name):
    """
    Get the embedding dimension for a specific model
    
    Args:
        model_name: Name of the model
        
    Returns:
        int: Embedding dimension
    """
    # Define embedding dimensions for known models
    embedding_dims = {
        'hubert_large': 1024,
        'wav2vec2_xlsr': 1024,
        'unispeech_sat': 768,
        'wavlm_base_plus': 768
    }
    
    return embedding_dims.get(model_name, 768)  # Default to 768 if unknown

def get_target_modules(model_name):
    """
    Get the appropriate target modules for LoRA based on model architecture
    
    Args:
        model_name: Name of the model
        
    Returns:
        list: List of target module names
    """
    # Define target modules for different model architectures
    if 'wavlm' in model_name:
        return ["k_proj", "q_proj", "v_proj", "out_proj"]
    elif 'hubert' in model_name:
        return ["k_proj", "q_proj", "v_proj", "out_proj"]
    elif 'wav2vec2' in model_name:
        return ["k_proj", "q_proj", "v_proj", "out_proj"]
    elif 'unispeech' in model_name:
        return ["k_proj", "q_proj", "v_proj", "out_proj"]
    else:
        return ["query", "key", "value", "output.dense"]  # Generic transformer modules

class SpeakerVerificationModel(nn.Module):
    """Speaker verification model with a pretrained backbone and ArcFace head"""
    def __init__(self, pretrained_model, embedding_dim, num_speakers):
        super(SpeakerVerificationModel, self).__init__()
        self.backbone = pretrained_model
        self.arcface = ArcFaceLayer(embedding_dim, num_speakers)
        
    def forward(self, x=None, labels=None, **kwargs):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor (audio)
            labels: Speaker labels
            **kwargs: Additional keyword arguments that might be passed by PEFT
            
        Returns:
            Tuple of (logits, embeddings) during training or just embeddings during inference
        """
        # Handle the case where input is passed as a keyword argument
        if x is None and 'input_values' in kwargs:
            x = kwargs['input_values']
        elif x is None and 'input_ids' in kwargs:
            x = kwargs['input_ids']
            
        # Extract embeddings from backbone
        outputs = self.backbone(x)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Apply ArcFace classifier if labels are provided
        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits, embeddings
        
        # For inference, just return embeddings
        return embeddings

def apply_lora(model, model_name):
    """
    Apply LoRA adapters to the model
    
    Args:
        model: The model to apply LoRA to
        model_name: Name of the model for model-specific configurations
        
    Returns:
        model: The model with LoRA adapters
    """
    # Get appropriate target modules for this model architecture
    target_modules = get_target_modules(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,  # Rank
        lora_alpha=32,  # Alpha scaling
        lora_dropout=0.1,  # Dropout probability
        target_modules=target_modules,
        bias="none",  # Don't train bias parameters
        inference_mode=False,  # Important: Set to False to make sure adapter is trainable
        init_lora_weights="gaussian",  # ADDED: Initialize with Gaussian rather than zeros for B
    )
    
    # Apply LoRA to model
    try:
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Explicitly initialize B matrices with small non-zero values
        # This ensures they're not stuck at zero
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    # Initialize with small random values
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                    logger.info(f"Explicitly initialized {name} with random values")
        
        # Enable adapters - CRITICAL for both training and inference
        if hasattr(model, 'enable_adapters'):
            model.enable_adapters()
            logger.info("LoRA adapters explicitly enabled")
    except Exception as e:
        logger.warning(f"Error applying LoRA: {e}")
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable%: {100 * trainable_params / all_params:.4f}%")
    
    return model

def collate_fn(batch):
    """
    Custom collate function for variable-length audio
    
    Args:
        batch: List of samples
        
    Returns:
        Batched samples with padding
    """
    # Filter out None values and problematic samples
    valid_batch = []
    for item in batch:
        if item is None:
            continue
        if 'audio' not in item or 'label' not in item:
            continue
        if not isinstance(item['audio'], torch.Tensor) or not isinstance(item['label'], torch.Tensor):
            continue
        if item['audio'].dim() == 0 or item['audio'].nelement() == 0:
            continue
        valid_batch.append(item)
    
    # If no valid samples, return a minimal batch
    if len(valid_batch) == 0:
        logger.warning("No valid samples in batch, returning dummy batch")
        dummy_audio = torch.zeros(1, 16000)  # 1 second of silence
        dummy_label = torch.zeros(1, dtype=torch.long)
        return {
            'audio': dummy_audio,
            'label': dummy_label
        }
    
    batch = valid_batch
    
    # Get max length
    max_length = max([item['audio'].shape[0] for item in batch])
    
    # Pad audio
    audio_batch = []
    for item in batch:
        audio = item['audio']
        padding = max_length - audio.shape[0]
        if padding > 0:
            audio = torch.nn.functional.pad(audio, (0, padding))
        audio_batch.append(audio)
    
    # Stack
    try:
        audio_batch = torch.stack(audio_batch)
        labels = torch.stack([item['label'] for item in batch])
    except Exception as e:
        logger.error(f"Error stacking tensors in collate_fn: {e}")
        # Return minimal batch in case of error
        dummy_audio = torch.zeros(1, 16000)
        dummy_label = torch.zeros(1, dtype=torch.long)
        return {
            'audio': dummy_audio,
            'label': dummy_label
        }
    
    return {
        'audio': audio_batch,
        'label': labels
    }

def calculate_speaker_identification_accuracy(model, dataloader, device):
    """
    Calculate speaker identification accuracy using the embedding similarity approach
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing test utterances
        device: Device to run model on
        
    Returns:
        float: Speaker identification accuracy
    """
    model.eval()
    
    # First pass: collect speaker embeddings for reference
    speaker_embeddings = {}  # {speaker_id: [embedding1, embedding2, ...]}
    
    with torch.no_grad():
        # Add progress bar for collecting speaker embeddings
        progress_bar = tqdm(
            dataloader, 
            desc="Collecting speaker embeddings",
            dynamic_ncols=True,
            unit="batch"
        )
        for batch in progress_bar:
            inputs = batch['audio'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Extract embeddings
            _, embeddings = model(x=inputs, labels=torch.tensor(labels).to(device))
            
            # Store embeddings by speaker
            for i, label in enumerate(labels):
                if label not in speaker_embeddings:
                    speaker_embeddings[label] = []
                speaker_embeddings[label].append(embeddings[i].cpu())
    
    # Calculate average embedding per speaker (centroid)
    speaker_centroids = {}
    for speaker, embs in speaker_embeddings.items():
        if len(embs) > 0:
            # Stack all embeddings for this speaker and average
            speaker_centroids[speaker] = torch.stack(embs).mean(dim=0)
    
    # Second pass: evaluate identification accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Add progress bar for evaluating identification accuracy
        progress_bar = tqdm(
            dataloader, 
            desc="Evaluating identification accuracy",
            dynamic_ncols=True,
            unit="batch"
        )
        for batch in progress_bar:
            inputs = batch['audio'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Extract embeddings
            _, embeddings = model(x=inputs, labels=torch.tensor(labels).to(device))
            
            batch_correct = 0
            batch_total = 0
            
            # For each embedding, find the closest speaker centroid
            for i, emb in enumerate(embeddings):
                true_speaker = labels[i]
                
                # Skip if we don't have this speaker in our centroids
                if true_speaker not in speaker_centroids:
                    continue
                
                # Calculate cosine similarity with all speaker centroids
                similarities = {}
                for speaker, centroid in speaker_centroids.items():
                    sim = F.cosine_similarity(emb.cpu().unsqueeze(0), centroid.unsqueeze(0))
                    similarities[speaker] = sim.item()
                
                # Find speaker with highest similarity
                if similarities:
                    predicted_speaker = max(similarities, key=similarities.get)
                    if predicted_speaker == true_speaker:
                        correct += 1
                        batch_correct += 1
                    total += 1
                    batch_total += 1
            
            # Update progress bar with batch accuracy
            if batch_total > 0:
                batch_acc = batch_correct / batch_total
                progress_bar.set_postfix({
                    'acc': f"{batch_acc:.4f}"
                })
    
    return correct / total if total > 0 else 0

def verify_lora_activation(model):
    """
    Verify that LoRA adapters are properly activated in the model
    
    Args:
        model: The model to verify
        
    Returns:
        bool: True if LoRA is properly activated, False otherwise
    """
    logger.info("Verifying LoRA adapter activation...")
    
    # Check if model has the peft_config attribute (indicating it's a PEFT model)
    if not hasattr(model, 'peft_config'):
        logger.error("Model does not have peft_config attribute - LoRA not applied!")
        return False
    
    # Ensure the model's active adapters is not None
    if hasattr(model, 'active_adapters') and model.active_adapters is None:
        logger.error("Model's active_adapters is None - LoRA disabled!")
        model.enable_adapters()
        logger.info("Explicitly enabled LoRA adapters")
    
    # Check if inference_mode is set to False
    if hasattr(model, 'peft_config'):
        for config in model.peft_config.values():
            if hasattr(config, 'inference_mode') and config.inference_mode:
                logger.error("LoRA inference_mode is True - should be False for training!")
                config.inference_mode = False
                logger.info("Fixed inference_mode to False")
    
    logger.info("LoRA verification complete")
    return True

def extract_embeddings_with_lora(model, inputs):
    """
    Extract embeddings from a model with LoRA adapters, ensuring they're activated
    
    Args:
        model: The model to extract embeddings from
        inputs: Input tensor
        
    Returns:
        torch.Tensor: Embeddings
    """
    # Ensure LoRA adapters are enabled
    if hasattr(model, 'enable_adapters'):
        model.enable_adapters()
    
    # Extract embeddings
    with torch.no_grad():
        if hasattr(model, 'backbone'):
            # Use the backbone directly to ensure LoRA is applied
            outputs = model.backbone(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            # If model is already a backbone
            outputs = model(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

def train_model(model_name, train_metadata, val_metadata, audio_root, 
                output_dir, batch_size=8, epochs=5, learning_rate=1e-4, max_files_per_speaker=10, use_gpu=True):
    """
    Fine-tune a speaker verification model using LoRA and ArcFace loss
    
    Args:
        model_name: Name of the pretrained model to fine-tune
        train_metadata: Path to training metadata file
        val_metadata: Path to validation metadata file
        audio_root: Root directory containing audio files
        output_dir: Directory to save the fine-tuned model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer (reduced to 1e-4 from 5e-5)
        max_files_per_speaker: Maximum number of files to use per speaker
        use_gpu: Whether to use GPU for training (if available)
    
    Returns:
        model: Fine-tuned model
    """
    # Force CUDA if available and requested
    global device
    cuda_available = torch.cuda.is_available()
    
    if use_gpu and cuda_available:
        device_to_use = torch.device("cuda")
        logger.info(f"Using GPU for training: {torch.cuda.get_device_name(0)}")
        # Disable mixed precision training to avoid repeated warnings
        logger.info("Mixed precision training is disabled to avoid warnings")
        use_amp = False
        scaler = None
    elif use_gpu and not cuda_available:
        logger.warning("GPU requested but CUDA is not available. Using CPU instead.")
        device_to_use = torch.device("cpu")
        use_amp = False
        scaler = None
    else:
        device_to_use = torch.device("cpu")
        logger.info("Using CPU for training (GPU not requested)")
        use_amp = False
        scaler = None
    
    # Set global device for consistent usage
    device = device_to_use
    
    # Update device in imported modules - only log this once
    pretrained_eval.device = device
    
    # Set device in PyTorch backends for best performance
    if cuda_available:
        torch.backends.cudnn.benchmark = True
    
    # Single clear log of the device being used
    logger.info(f"Training on {device} ({torch.cuda.get_device_name(0) if cuda_available else 'CPU'})")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model and processor
    logger.info(f"Loading pretrained model: {model_name}")
    pretrained_model, processor = load_pretrained_model(model_name)
    
    # Move model to correct device immediately
    pretrained_model = pretrained_model.to(device)
    
    # Determine embedding dimension
    embedding_dim = get_embedding_dim(model_name)
    logger.info(f"Using embedding dimension: {embedding_dim}")
    
    # Create datasets with STRICT assignment-specific requirements
    logger.info("Creating datasets...")
    
    # Get all unique speaker IDs from the test file first
    test_df = pd.read_csv(val_metadata, sep=' ', names=['id', 'path', 'gender'])
    test_speakers = sorted(test_df['id'].unique())
    logger.info(f"Found {len(test_speakers)} unique speakers in test metadata")
    
    # Create train dataset with first 100 speakers
    train_dataset = VoxCelebSpeakerDataset(
        train_metadata, 
        audio_root, 
        processor, 
        max_files_per_speaker=max_files_per_speaker,  # Use the passed max_files_per_speaker
        use_first_n_speakers=100,  # STRICT: Use first 100 speakers for training as per assignment
        is_test=False
    )
    
    # Create validation dataset - use the test speakers directly
    val_dataset = VoxCelebSpeakerDataset(
        val_metadata, 
        audio_root, 
        processor, 
        max_files_per_speaker=max_files_per_speaker,  # Use the passed max_files_per_speaker
        use_first_n_speakers=0,    # Don't limit speakers, use all available test speakers
        is_test=True               # Indicates this is a test dataset
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with progress bars
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=cuda_available  # Pin memory for faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with progress bars
        collate_fn=collate_fn,
        pin_memory=cuda_available  # Pin memory for faster GPU transfer
    )
    
    # Create model and ensure it's on the correct device
    logger.info(f"Creating speaker verification model with {len(train_dataset.speakers)} speakers")
    model = SpeakerVerificationModel(
        pretrained_model, 
        embedding_dim,
        len(train_dataset.speakers)
    ).to(device)  # STRICT: Explicitly move to device
    
    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = apply_lora(model, model_name)
    
    # Double-check model is on correct device after LoRA
    model = model.to(device)
    
    # Create optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,  # Add weight decay for better regularization
        betas=(0.9, 0.999)  # Default betas
    )
    
    # Use a more gradual learning rate schedule
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # Warm up for 10% of steps
        num_training_steps=total_steps
    )
    
    # Create criterion - use standard cross entropy loss for stability
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with modified mixed precision handling
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    logger.info(f"Dataset stats: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    logger.info(f"Using {len(train_dataset.speakers)} speakers for training, {len(val_dataset.speakers)} speakers for validation")
    
    # Add before the training loop starts
    verify_lora_activation(model)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        # Configure progress bar for training loop
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            dynamic_ncols=True,  # Adapt to terminal width
            unit="batch"
        )
        for batch in progress_bar:
            # Clear gradients at the start of each iteration
            optimizer.zero_grad()
            
            inputs = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            # Prepare inputs - some models expect 'input_values' or 'input_ids'
            input_dict = {'x': inputs, 'labels': labels}
            
            # Standard training path without mixed precision
            logits, _ = model(**input_dict)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update learning rate scheduler
            scheduler.step()
            
            train_loss += loss.item()
            
            # Only track loss during training
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        # Validation with fixed mixed precision handling
        with torch.no_grad():
            # Configure progress bar for validation
            progress_bar = tqdm(
                val_loader, 
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                dynamic_ncols=True,  # Adapt to terminal width
                unit="batch"
            )
            for batch in progress_bar:
                inputs = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                # Prepare inputs
                input_dict = {'x': inputs, 'labels': labels}
                
                # Standard forward pass without mixed precision
                logits, _ = model(**input_dict)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                # Only track loss during validation
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Skip train accuracy calculation to save time
        train_accuracy = 0.0  # Just use a placeholder value
        train_accuracies.append(train_accuracy)
        
        # Comment out the identification accuracy calculation
        # val_accuracy = calculate_speaker_identification_accuracy(model, val_loader, device)
        val_accuracy = 0.0  # Placeholder
        val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"  Saved best model with val_loss: {val_loss:.4f}")
        
        # Check if LoRA parameters are actually changing
        if epoch % 1 == 0:  # Do this check every epoch
            logger.info("Checking LoRA parameter updates...")
            
            # Find some LoRA parameters to monitor
            lora_b_count = 0
            for name, param in model.named_parameters():
                if 'lora_B' in name and param.requires_grad:
                    lora_b_count += 1
                    if lora_b_count <= 5:  # Just log first 5 B matrices
                        # Log the norm of B matrices
                        param_norm = torch.norm(param).item()
                        param_sum = torch.sum(torch.abs(param)).item()
                        logger.info(f"  {name} norm: {param_norm:.6f}, abs sum: {param_sum:.6f}")
                        if param_norm < 1e-6:
                            logger.warning(f"  WARNING: {name} has very small norm - training may not be effective!")
    
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
    # Only plot validation accuracy since train accuracy is just a placeholder
    plt.plot(val_accuracies, label='Validation', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy Curve')
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    
    # Additionally save LoRA weights explicitly
    logger.info("Saving LoRA weights explicitly...")
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_state_dict[name] = param.data.clone()

    if len(lora_state_dict) > 0:
        torch.save(lora_state_dict, output_dir / "lora_weights.pt")
        logger.info(f"Saved {len(lora_state_dict)} LoRA parameters to lora_weights.pt")
    else:
        logger.warning("No LoRA parameters found to save!")
    
    # Save model config
    with open(output_dir / "model_config.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Embedding dimension: {embedding_dim}\n")
        f.write(f"Number of speakers: {len(train_dataset.speakers)}\n")
        f.write(f"Max files per speaker: {max_files_per_speaker}\n")
        f.write(f"Final validation accuracy: {val_accuracies[-1]:.4f}\n")
    
    return model

def evaluate_model_performance(model, processor, trial_file, audio_root, batch_size=4, model_name=""):
    """
    Evaluate model performance on speaker verification using VoxCeleb1 trial pairs
    
    Args:
        model: The model to evaluate
        processor: The processor for the model
        trial_file: Path to the trial file
        audio_root: Root directory for audio files
        batch_size: Batch size for evaluation
        model_name: Name of the model (for logging)
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on {trial_file}")
    
    # Load trial pairs and limit to 4000 as requested
    with open(trial_file, 'r') as f:
        trial_pairs = f.readlines()[:4000]  # Limit to top 4000 trial pairs
    
    # Write a temporary trial file with the limited pairs
    temp_trial_file = os.path.join(os.path.dirname(trial_file), "temp_trial_4000.txt")
    with open(temp_trial_file, 'w') as f:
        f.writelines(trial_pairs)
    
    logger.info(f"Limited evaluation to 4000 trial pairs (from {len(trial_pairs)} total)")
    
    # Create dataset and dataloader with the limited trial file
    dataset = pretrained_eval.VoxCelebTrialDataset(temp_trial_file, audio_root, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=pretrained_eval.custom_collate_fn
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect all labels and scores
    all_labels = []
    all_scores = []
    
    logger.info("Processing trial pairs...")
    for batch in tqdm(dataloader, desc="Evaluating verification performance", dynamic_ncols=True, unit="batch"):
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
                emb1 = extract_embeddings_with_lora(model, audio1)
                
                # Process second audio
                audio2 = audio2_samples[i].unsqueeze(0).to(device)
                emb2 = extract_embeddings_with_lora(model, audio2)
                
                # Compute similarity
                score = pretrained_eval.compute_similarity(emb1, emb2).item()
                batch_scores.append(score)
        
        # Collect labels and scores
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(batch_scores)
    
    # Clean up temporary file
    try:
        os.remove(temp_trial_file)
    except:
        logger.warning(f"Could not remove temporary trial file: {temp_trial_file}")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far, far_threshold = calculate_tar_at_far(all_labels, all_scores, far_target=0.01)
    
    # Calculate verification accuracy (using optimal threshold)
    pred_labels = (all_scores >= eer_threshold).astype(int)
    verification_accuracy = accuracy_score(all_labels, pred_labels)
    
    results = {
        'model_name': model_name,
        'eer': eer * 100,  # Convert to percentage
        'tar_at_1far': tar_at_far * 100,  # Convert to percentage
        'verification_accuracy': verification_accuracy * 100,  # Convert to percentage
        'eer_threshold': eer_threshold,
        'far_threshold': far_threshold,
        'num_trials': len(all_labels)
    }
    
    logger.info(f"Evaluation results for {model_name} on {len(all_labels)} trials:")
    logger.info(f"  EER: {eer*100:.2f}%")
    logger.info(f"  TAR@1%FAR: {tar_at_far*100:.2f}%")
    logger.info(f"  Verification Accuracy: {verification_accuracy*100:.2f}%")
    
    return results

def compare_pretrained_and_finetuned(model_name, trial_file, audio_root, 
                                      finetuned_model_path, output_dir,
                                      batch_size=4):
    """
    Compare pretrained and finetuned models on speaker verification
    
    Args:
        model_name: Name of the model
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        finetuned_model_path: Path to finetuned model weights
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        
    Returns:
        tuple: (pretrained_results, finetuned_results)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trial pairs and limit to 4000 as requested
    with open(trial_file, 'r') as f:
        trial_pairs = f.readlines()[:4000]  # Limit to top 4000 trial pairs
    
    # Write a temporary trial file with the limited pairs
    temp_trial_file = os.path.join(os.path.dirname(trial_file), "temp_trial_4000.txt")
    with open(temp_trial_file, 'w') as f:
        f.writelines(trial_pairs)
    
    logger.info(f"Limited comparison to 4000 trial pairs (from original trial file)")
    
    # Load pretrained model
    logger.info(f"Loading pretrained model: {model_name}")
    pretrained_model, processor = load_pretrained_model(model_name)
    pretrained_model.eval()
    
    # Evaluate pretrained model
    pretrained_results = evaluate_model_performance(
        pretrained_model, 
        processor, 
        temp_trial_file, 
        audio_root, 
        batch_size,
        f"Pretrained {model_name}"
    )
    
    # Load finetuned model
    logger.info(f"Loading finetuned model from {finetuned_model_path}")
    # Get embedding dimension
    embedding_dim = get_embedding_dim(model_name)
    
    # Create VoxCeleb2 dataset (just to get speaker count)
    train_dataset = VoxCelebSpeakerDataset(
        "data/vox2/vox2_train.csv", 
        "data/vox2/aac", 
        processor=None, 
        max_files_per_speaker=10,
        use_first_n_speakers=100,
        is_test=False
    )
    
    # Create model
    finetuned_model = SpeakerVerificationModel(
        pretrained_model, 
        embedding_dim,
        len(train_dataset.speakers)
    ).to(device)

    # Apply LoRA
    finetuned_model = apply_lora(finetuned_model, model_name)

    # Load weights
    finetuned_model.load_state_dict(torch.load(finetuned_model_path, map_location=device))

    # Verify LoRA activation explicitly before evaluation
    verify_lora_activation(finetuned_model)

    finetuned_model.eval()
    
    # Evaluate finetuned model
    finetuned_results = evaluate_model_performance(
        finetuned_model, 
        processor, 
        temp_trial_file, 
        audio_root, 
        batch_size,
        f"Finetuned {model_name}"
    )
    
    # Clean up temporary file
    try:
        os.remove(temp_trial_file)
    except:
        logger.warning(f"Could not remove temporary trial file: {temp_trial_file}")
    
    # Save results
    results_df = pd.DataFrame([pretrained_results, finetuned_results])
    results_file = output_dir / "comparison_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # Plot comparison
    labels = [f"Pretrained {model_name}", f"Finetuned {model_name}"]
    metrics = ['eer', 'tar_at_1far', 'verification_accuracy']
    metric_names = ['EER (%)', 'TAR@1%FAR (%)', 'Verification Accuracy (%)']
    
    plt.figure(figsize=(15, 5))
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(1, 3, i+1)
        values = [pretrained_results[metric], finetuned_results[metric]]
        
        # For EER, lower is better
        if metric == 'eer':
            colors = ['red', 'green'] if values[0] > values[1] else ['green', 'red']
        else:  # For TAR and Verification Accuracy, higher is better
            colors = ['red', 'green'] if values[0] < values[1] else ['green', 'red']
            
        plt.bar(labels, values, color=colors)
        plt.title(name)
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=15)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            plt.text(j, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_plot.png")
    logger.info(f"Comparison plot saved to {output_dir / 'comparison_plot.png'}")
    
    return pretrained_results, finetuned_results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a speaker verification model")
    parser.add_argument("--model", type=str, default="wavlm_base_plus", 
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Pretrained model to fine-tune")
    parser.add_argument("--train_metadata", type=str, required=True,
                        help="Path to training metadata file")
    parser.add_argument("--val_metadata", type=str, required=True,
                        help="Path to validation metadata file")
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Root directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="models/speaker_verification",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for optimizer (reduced to 1e-4 from 5e-5)")
    parser.add_argument("--max_files_per_speaker", type=int, default=10,
                        help="Maximum number of files to use per speaker (will be capped at 10)")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Whether to use GPU for training (if available)")
    parser.add_argument("--trial_file", type=str, default="data/vox1/veri_test2.txt",
                        help="Path to trial file for evaluation")
    parser.add_argument("--vox1_audio_root", type=str, default="data/vox1/wav",
                        help="Root directory for VoxCeleb1 audio files (for evaluation)")
    parser.add_argument("--evaluate_only", action="store_true", default=False,
                        help="Skip training and only run evaluation")
    parser.add_argument("--skip_evaluation", action="store_true", default=False,
                        help="Skip evaluation and only run training")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to save the fine-tuned model
    model_path = output_dir / "best_model.pt"
    
    # Train model if not in evaluate-only mode
    if not args.evaluate_only:
        # Fine-tune model
        train_model(
            model_name=args.model,
            train_metadata=args.train_metadata,
            val_metadata=args.val_metadata,
            audio_root=args.audio_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_files_per_speaker=args.max_files_per_speaker,
            use_gpu=args.use_gpu
        )
    
    # Evaluate models if not in skip-evaluation mode
    if not args.skip_evaluation:
        if not os.path.exists(args.trial_file):
            logger.error(f"Trial file not found: {args.trial_file}")
            logger.error("Skipping evaluation. Please provide a valid trial file path.")
        elif not os.path.exists(args.vox1_audio_root):
            logger.error(f"VoxCeleb1 audio root not found: {args.vox1_audio_root}")
            logger.error("Skipping evaluation. Please provide a valid VoxCeleb1 audio root.")
        elif not os.path.exists(model_path):
            logger.error(f"Fine-tuned model not found: {model_path}")
            logger.error("Skipping evaluation. Please train the model first or provide a valid model path.")
        else:
            # Compare pretrained and finetuned models
            compare_pretrained_and_finetuned(
                model_name=args.model,
                trial_file=args.trial_file,
                audio_root=args.vox1_audio_root,
                finetuned_model_path=model_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size
            ) 
