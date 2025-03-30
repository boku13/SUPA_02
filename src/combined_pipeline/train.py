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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio
from model import load_combined_model, CombinedSpeechEnhancementModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

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

def train_combined_model(model, train_loader, val_loader, output_dir, 
                         epochs=10, learning_rate=1e-4):
    """
    Train the combined speech enhancement model
    
    Args:
        model: Combined model
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
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
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

def custom_collate_fn(batch):
    """
    Custom collate function for variable length audio
    
    Args:
        batch: Batch of data
        
    Returns:
        Collated batch
    """
    # Find the longest sequence
    max_len = max([item['mixture'].shape[0] for item in batch])
    
    # Pad all sequences to the longest
    for item in batch:
        padding_len = max_len - item['mixture'].shape[0]
        if padding_len > 0:
            padding = torch.zeros(padding_len)
            item['mixture'] = torch.cat([item['mixture'], padding])
            item['source1'] = torch.cat([item['source1'], padding])
            item['source2'] = torch.cat([item['source2'], padding])
    
    # Collate the batch
    collated_batch = {
        'mixture': torch.stack([item['mixture'] for item in batch]),
        'source1': torch.stack([item['source1'] for item in batch]),
        'source2': torch.stack([item['source2'] for item in batch]),
        'speaker1_id': [item['speaker1_id'] for item in batch],
        'speaker2_id': [item['speaker2_id'] for item in batch],
        'mixture_id': [item['mixture_id'] for item in batch]
    }
    
    return collated_batch

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train combined speech enhancement model")
    parser.add_argument("--train_metadata", type=str, required=True,
                        help="Path to training metadata CSV file")
    parser.add_argument("--val_metadata", type=str, required=True,
                        help="Path to validation metadata CSV file")
    parser.add_argument("--speaker_model_path", type=str, default=None,
                        help="Path to pretrained/finetuned speaker model")
    parser.add_argument("--speaker_model_name", type=str, default="wavlm_base_plus",
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Name of the speaker model architecture")
    parser.add_argument("--output_dir", type=str, default="models/combined_pipeline",
                        help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--max_length", type=int, default=160000,
                        help="Maximum audio length in samples (10 seconds at 16kHz)")
    
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = MultiSpeakerMixtureDataset(args.train_metadata, args.max_length)
    val_dataset = MultiSpeakerMixtureDataset(args.val_metadata, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # Load model
    model = load_combined_model(
        args.speaker_model_path,
        args.speaker_model_name
    )
    
    # Train model
    trained_model = train_combined_model(
        model,
        train_loader,
        val_loader,
        args.output_dir,
        args.epochs,
        args.lr
    ) 