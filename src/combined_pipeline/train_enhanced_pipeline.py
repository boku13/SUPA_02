import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import mir_eval

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio
from speech_enhancement.sepformer import SepFormerWrapper
from speech_enhancement.evaluation import evaluate_separation
from speaker_verification.pretrained_eval import load_pretrained_model, extract_embeddings
from speaker_verification.finetune import SpeakerVerificationModel, apply_lora

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
    logger.info("PESQ module is available for evaluation")
except ImportError:
    PESQ_AVAILABLE = False
    logger.warning("PESQ module not available, skipping PESQ calculation")

class EnhancedCombinedModel(nn.Module):
    """
    Enhanced combined model that integrates speaker identification with 
    speech separation for better speaker-aware speech enhancement.
    """
    def __init__(self, 
                speaker_model, 
                sepformer_wrapper,
                embedding_dim=768,
                num_speakers=2,
                use_speaker_conditioning=True):
        """
        Initialize the enhanced combined model
        
        Args:
            speaker_model: Pretrained/finetuned speaker verification model
            sepformer_wrapper: Wrapper for the SepFormer model
            embedding_dim: Dimension of speaker embeddings
            num_speakers: Number of speakers to separate
            use_speaker_conditioning: Whether to use speaker conditioning in enhancement
        """
        super(EnhancedCombinedModel, self).__init__()
        
        # Speaker verification model (embedded as a torch module)
        self.speaker_model = speaker_model
        
        # SepFormer model (through wrapper)
        self.sepformer_wrapper = sepformer_wrapper
        
        # Whether to use speaker embeddings to condition enhancement
        self.use_speaker_conditioning = use_speaker_conditioning
        
        # Speaker-conditional enhancement network
        if use_speaker_conditioning:
            self.speaker_conditioning = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256)
            )
        
        # Post-processing enhancement network (improves separated sources)
        self.enhancement_network = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=3, padding=1),
            ) for _ in range(num_speakers)
        ])
        
        # Freeze speaker model weights
        for param in self.speaker_model.parameters():
            param.requires_grad = False
        
        # The sepformer model is not a nn.Module, so we don't freeze it here
        # It will be used for inference only
    
    def forward(self, mixture, reference_embeddings=None):
        """
        Forward pass
        
        Args:
            mixture: Mixed audio signal [batch_size, audio_length]
            reference_embeddings: Optional reference speaker embeddings
                                [num_speakers, embedding_dim]
            
        Returns:
            enhanced_sources: Enhanced separated sources
            source_embeddings: Extracted speaker embeddings for each source
        """
        # Get batch size and ensure directory exists
        batch_size = mixture.shape[0]
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)
        
        # Process each mixture separately
        separated_sources_list = []
        
        for i in range(batch_size):
            # Create a valid temporary WAV file for this batch item
            tmp_mixture_path = tmp_dir / f"tmp_mixture_{i}.wav"
            
            # Ensure the tensor is on CPU and properly shaped for torchaudio
            mix_cpu = mixture[i].cpu().unsqueeze(0)
            
            # Save to temporary file
            torchaudio.save(
                str(tmp_mixture_path),  # Convert to string for torchaudio
                mix_cpu,
                16000  # Default sample rate for SepFormer
            )
            
            try:
                # Run separation on this file
                separated_sources = self.sepformer_wrapper.separate(
                    str(tmp_mixture_path),
                    save_results=False
                )
                
                # Check the shape and transpose if needed
                # SepFormer returns [audio_length, num_sources]
                # We need [num_sources, audio_length]
                if separated_sources.shape[1] < separated_sources.shape[0]:
                    # First dimension is likely audio_length, second is num_sources
                    separated_sources = separated_sources.transpose(0, 1)
                
                # Add to our list
                separated_sources_list.append(separated_sources)
                
            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                # Create dummy output as fallback (2 sources of same length as input)
                audio_length = mixture.shape[1]
                dummy_sources = torch.zeros((2, audio_length), device='cpu')
                separated_sources_list.append(dummy_sources)
            
            finally:
                # Clean up temp file
                if tmp_mixture_path.exists():
                    try:
                        os.remove(tmp_mixture_path)
                    except:
                        pass
        
        # Find the common shape among all separated sources
        # Each item in separated_sources_list is [num_sources, audio_length]
        num_sources = separated_sources_list[0].shape[0]  # Usually 2 for SepFormer
        
        # Find the minimum audio length to ensure consistent tensor sizes
        min_audio_length = min([sources.shape[1] for sources in separated_sources_list])
        
        # Initialize tensors for the batch with the minimum length
        # Shape: [num_sources, batch_size, min_audio_length]
        separated_sources_batch = torch.zeros(
            (num_sources, batch_size, min_audio_length),
            device=device
        )
        
        # Fill in the batch tensor (trimming if necessary)
        for i, separated_sources in enumerate(separated_sources_list):
            for j in range(num_sources):
                # Handle potential shape issues
                if j < separated_sources.shape[0] and min_audio_length <= separated_sources.shape[1]:
                    # Ensure we're slicing correctly based on shape
                    separated_sources_batch[j, i] = separated_sources[j, :min_audio_length].to(device)
                else:
                    # In case of shape mismatch, use zeros
                    logger.warning(f"Shape mismatch for source {j} in batch item {i}. Filling with zeros.")
                    separated_sources_batch[j, i] = torch.zeros(min_audio_length, device=device)
        
        # Extract speaker embeddings for each separated source
        source_embeddings = []
        for i in range(num_sources):
            # Shape: [batch_size, min_audio_length]
            source_audio = separated_sources_batch[i]
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.speaker_model(source_audio)
                source_embeddings.append(embeddings)
        
        # If reference embeddings are provided, match sources to references
        if reference_embeddings is not None:
            # Calculate similarity matrix: [batch_size, num_sources, num_refs]
            similarities = torch.zeros(
                (batch_size, num_sources, reference_embeddings.shape[0]),
                device=device
            )
            
            for i in range(batch_size):
                for j in range(num_sources):
                    for k in range(reference_embeddings.shape[0]):
                        # Compute cosine similarity
                        similarities[i, j, k] = F.cosine_similarity(
                            source_embeddings[j][i].unsqueeze(0),
                            reference_embeddings[k].unsqueeze(0),
                            dim=1
                        )
            
            # For each item in the batch, decide optimal assignment
            assignments = []
            for i in range(batch_size):
                # For 2 speakers, use a simple comparison of total similarity
                if num_sources == 2 and reference_embeddings.shape[0] == 2:
                    sim_1 = similarities[i, 0, 0] + similarities[i, 1, 1]
                    sim_2 = similarities[i, 0, 1] + similarities[i, 1, 0]
                    
                    if sim_1 >= sim_2:
                        assignments.append([0, 1])  # Default order
                    else:
                        assignments.append([1, 0])  # Swapped order
                else:
                    # For more speakers, use a more sophisticated assignment algorithm
                    # Not implemented in this version
                    assignments.append(list(range(num_sources)))
            
            # Reorder sources and embeddings based on assignments
            reordered_sources = torch.zeros_like(separated_sources_batch)
            reordered_embeddings = []
            
            for i in range(batch_size):
                for j in range(num_sources):
                    # Get source index for this reference
                    source_idx = assignments[i].index(j)
                    reordered_sources[j, i] = separated_sources_batch[source_idx, i]
            
            # Update separated sources with reordered version
            separated_sources_batch = reordered_sources
            
            # Also reorder embeddings
            for i in range(num_sources):
                batch_embeddings = []
                for j in range(batch_size):
                    source_idx = assignments[j].index(i)
                    batch_embeddings.append(source_embeddings[source_idx][j])
                reordered_embeddings.append(torch.stack(batch_embeddings))
            
            source_embeddings = reordered_embeddings
        
        # Apply enhancement network to each source
        enhanced_sources = []
        for i in range(num_sources):
            # Reshape for 1D convolution: [batch_size, 1, audio_length]
            source_reshaped = separated_sources_batch[i].unsqueeze(1)
            
            # Apply enhancement network
            enhanced = self.enhancement_network[i](source_reshaped)
            
            # Reshape back to [batch_size, audio_length]
            enhanced = enhanced.squeeze(1)
            
            enhanced_sources.append(enhanced)
        
        return enhanced_sources, source_embeddings

class MultiSpeakerMixtureDataset(Dataset):
    """Dataset for multi-speaker mixtures"""
    def __init__(self, metadata_file, max_length=160000):
        """
        Initialize dataset
        
        Args:
            metadata_file: Path to metadata CSV file
            max_length: Maximum audio length in samples
        """
        self.metadata = pd.read_csv(metadata_file)
        self.max_length = max_length
        
        logger.info(f"Loaded dataset with {len(self.metadata)} mixtures")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load mixture
        mixture_path = row['mixture_path']
        mixture, sr = load_audio(mixture_path)
        
        # Load source 1
        source1_path = row['source1_path']
        source1, _ = load_audio(source1_path)
        
        # Load source 2
        source2_path = row['source2_path']
        source2, _ = load_audio(source2_path)
        
        # Get speaker IDs
        speaker1_id = row['speaker1_id']
        speaker2_id = row['speaker2_id']
        
        # Pad or trim audio to max_length
        if mixture.shape[0] > self.max_length:
            # Randomly crop
            start = torch.randint(0, mixture.shape[0] - self.max_length, (1,)).item()
            mixture = mixture[start:start + self.max_length]
            source1 = source1[start:start + self.max_length]
            source2 = source2[start:start + self.max_length]
        else:
            # Pad with zeros
            padding = torch.zeros(self.max_length - mixture.shape[0])
            mixture = torch.cat([mixture, padding])
            source1 = torch.cat([source1, padding])
            source2 = torch.cat([source2, padding])
        
        return {
            'mixture': mixture,
            'source1': source1,
            'source2': source2,
            'speaker1_id': speaker1_id,
            'speaker2_id': speaker2_id,
            'mixture_id': row['mixture_id']
        }

def waveform_loss(estimated, target, alpha=0.5):
    """
    Combined L1 and L2 loss for waveform comparison
    
    Args:
        estimated: Estimated waveform
        target: Target waveform
        alpha: Weight for L1 loss (1-alpha for L2)
        
    Returns:
        Combined loss value
    """
    l1_loss = F.l1_loss(estimated, target)
    l2_loss = F.mse_loss(estimated, target)
    
    return alpha * l1_loss + (1 - alpha) * l2_loss

def train_enhanced_model(model, train_loader, val_loader, output_dir, 
                         epochs=10, learning_rate=1e-4):
    """
    Train the enhanced combined model
    
    Args:
        model: Enhanced combined model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        output_dir: Directory to save the model
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Only optimize the parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Training with {len(trainable_params)} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Get batch data
            mixture = batch['mixture'].to(device)
            source1 = batch['source1'].to(device)
            source2 = batch['source2'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            enhanced_sources, _ = model(mixture)
            
            # Calculate loss
            loss1 = waveform_loss(enhanced_sources[0], source1)
            loss2 = waveform_loss(enhanced_sources[1], source2)
            loss = loss1 + loss2
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in progress_bar:
                # Get batch data
                mixture = batch['mixture'].to(device)
                source1 = batch['source1'].to(device)
                source2 = batch['source2'].to(device)
                
                # Forward pass
                enhanced_sources, _ = model(mixture)
                
                # Calculate loss
                loss1 = waveform_loss(enhanced_sources[0], source1)
                loss2 = waveform_loss(enhanced_sources[1], source2)
                loss = loss1 + loss2
                
                # Update progress bar
                val_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
        }, output_dir / f"checkpoint_epoch{epoch+1}.pt")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(output_dir / "training_curves.png")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    
    return model

def load_enhanced_model(speaker_model_path=None, speaker_model_name="wavlm_base_plus", 
                      embedding_dim=768, num_speakers=2, use_speaker_conditioning=True):
    """
    Load the enhanced speech separation model
    
    Args:
        speaker_model_path: Path to pretrained/finetuned speaker model
        speaker_model_name: Name of the speaker model architecture
        embedding_dim: Dimension of speaker embeddings
        num_speakers: Number of speakers to separate
        use_speaker_conditioning: Whether to use speaker conditioning
        
    Returns:
        model: Enhanced speech separation model
    """
    # Load pretrained speaker model
    pretrained_model, _ = load_pretrained_model(speaker_model_name)
    
    # Create speaker verification model
    speaker_model = SpeakerVerificationModel(
        pretrained_model,
        embedding_dim,
        100  # Placeholder for num_speakers, not used in inference
    )
    
    # Load finetuned weights if provided
    if speaker_model_path and os.path.exists(speaker_model_path):
        logger.info(f"Loading speaker model weights from {speaker_model_path}")
        try:
            # First try direct loading
            speaker_model.load_state_dict(torch.load(speaker_model_path, map_location=device))
        except:
            # Try LoRA weights
            logger.info("Direct loading failed, attempting to load with LoRA...")
            speaker_model = apply_lora(speaker_model, speaker_model_name)
            speaker_model.load_state_dict(torch.load(speaker_model_path, map_location=device), strict=False)
    
    # Set to evaluation mode
    speaker_model.eval()
    
    # Load SepFormer model through wrapper
    sepformer_wrapper = SepFormerWrapper()
    
    # Create combined model
    model = EnhancedCombinedModel(
        speaker_model,
        sepformer_wrapper,
        embedding_dim,
        num_speakers,
        use_speaker_conditioning
    )
    
    return model.to(device)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train enhanced combined speech separation model")
    parser.add_argument("--train_metadata", type=str, required=True,
                        help="Path to training metadata CSV file")
    parser.add_argument("--val_metadata", type=str, required=True,
                        help="Path to validation metadata CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--speaker_model_path", type=str, default=None,
                        help="Path to pretrained/finetuned speaker model")
    parser.add_argument("--speaker_model_name", type=str, default="wavlm_base_plus",
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Name of the speaker model architecture")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--no_speaker_conditioning", action="store_true",
                        help="Disable speaker conditioning in the enhancement network")
    
    args = parser.parse_args()
    
    # Load model
    model = load_enhanced_model(
        speaker_model_path=args.speaker_model_path,
        speaker_model_name=args.speaker_model_name,
        use_speaker_conditioning=not args.no_speaker_conditioning
    )
    
    # Create datasets
    train_dataset = MultiSpeakerMixtureDataset(args.train_metadata)
    val_dataset = MultiSpeakerMixtureDataset(args.val_metadata)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Train model
    trained_model = train_enhanced_model(
        model,
        train_loader,
        val_loader,
        args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"Model training complete. Best model saved to {args.output_dir}/best_model.pt")

if __name__ == "__main__":
    main() 