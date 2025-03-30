import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from speechbrain.pretrained import SepformerSeparation

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, setup_seed
from speaker_verification.finetune import SpeakerVerificationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

class CombinedSpeechEnhancementModel(nn.Module):
    """
    Combined model that integrates speaker identification with speech separation
    for enhanced speaker-aware speech separation.
    """
    def __init__(self, 
                speaker_model, 
                sepformer_model,
                embedding_dim=768,
                num_speakers=2):
        """
        Initialize the combined model
        
        Args:
            speaker_model: Pretrained speaker verification model
            sepformer_model: Pretrained SepFormer model
            embedding_dim: Dimension of speaker embeddings
            num_speakers: Number of speakers to separate
        """
        super(CombinedSpeechEnhancementModel, self).__init__()
        
        # Speaker verification model
        self.speaker_model = speaker_model
        
        # SepFormer model
        self.sepformer_model = sepformer_model
        
        # Speaker-conditional enhancement
        self.speaker_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # Post-processing modules (source-specific enhancement)
        self.post_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=3, padding=1),
            ) for _ in range(num_speakers)
        ])
        
        # Freeze speaker model and sepformer model weights
        for param in self.speaker_model.parameters():
            param.requires_grad = False
            
        # The sepformer model is not a torch.nn.Module, so we handle it differently
        
    def forward(self, mixture, reference_embeddings=None):
        """
        Forward pass
        
        Args:
            mixture: Mixed audio signal
            reference_embeddings: Optional reference speaker embeddings
            
        Returns:
            enhanced_sources: Enhanced separated sources
            predicted_speakers: Predicted speaker embeddings for each source
        """
        # Separate sources using SepFormer
        # Note: In a real implementation, you'd need to handle this differently
        # since SepFormer is not a torch.nn.Module
        separated_sources = self.separate_sources(mixture)
        
        # Get speaker embeddings for each separated source
        source_embeddings = []
        for source in separated_sources:
            # Extract speaker embedding
            with torch.no_grad():
                embedding = self.speaker_model(source.unsqueeze(0))
            source_embeddings.append(embedding)
        
        # If reference embeddings are provided, associate each source with a reference
        if reference_embeddings is not None:
            # Calculate similarity between separated source embeddings and reference embeddings
            similarities = []
            for source_emb in source_embeddings:
                source_similarities = []
                for ref_emb in reference_embeddings:
                    # Compute cosine similarity
                    sim = F.cosine_similarity(source_emb, ref_emb, dim=1)
                    source_similarities.append(sim.item())
                similarities.append(source_similarities)
            
            # Get the best matching reference for each source
            matched_indices = []
            for i in range(len(similarities)):
                matched_ref_idx = torch.argmax(torch.tensor(similarities[i])).item()
                matched_indices.append(matched_ref_idx)
            
            # Reorder sources and embeddings to match reference order
            ordered_sources = []
            ordered_embeddings = []
            for i in range(len(reference_embeddings)):
                # Find the source matched to this reference
                try:
                    source_idx = matched_indices.index(i)
                    ordered_sources.append(separated_sources[source_idx])
                    ordered_embeddings.append(source_embeddings[source_idx])
                except ValueError:
                    # No source matched to this reference
                    logger.warning(f"No source matched to reference {i}")
                    # Use a placeholder
                    ordered_sources.append(torch.zeros_like(separated_sources[0]))
                    ordered_embeddings.append(torch.zeros_like(source_embeddings[0]))
            
            separated_sources = ordered_sources
            source_embeddings = ordered_embeddings
        
        # Apply speaker-conditional enhancement
        enhanced_sources = []
        for i, (source, embedding) in enumerate(zip(separated_sources, source_embeddings)):
            # Process embedding for conditioning
            cond = self.speaker_conditioning(embedding)
            
            # Reshape source for 1D convolution [batch, channels, length]
            source_reshaped = source.unsqueeze(1)
            
            # Apply post-processing
            enhanced = self.post_processor[i % len(self.post_processor)](source_reshaped)
            
            # Reshape back to [batch, length]
            enhanced = enhanced.squeeze(1)
            
            enhanced_sources.append(enhanced)
        
        return enhanced_sources, source_embeddings
    
    def separate_sources(self, mixture):
        """
        Separate sources using the SepFormer model
        
        Args:
            mixture: Mixed audio signal
            
        Returns:
            separated_sources: Separated sources
        """
        # This is a placeholder for actual SepFormer separation
        # In a real implementation, you'd call the actual model
        
        # For now, let's assume it returns a tensor with shape [num_sources, audio_length]
        # We'll implement this properly in the train.py file
        separated_sources = torch.zeros((2, mixture.shape[1]), device=mixture.device)
        
        return separated_sources

def load_combined_model(speaker_model_path, speaker_model_name="wavlm_base_plus", 
                      embedding_dim=768, num_speakers=2):
    """
    Load the combined speech enhancement model
    
    Args:
        speaker_model_path: Path to pretrained/finetuned speaker model
        speaker_model_name: Name of the speaker model architecture
        embedding_dim: Dimension of speaker embeddings
        num_speakers: Number of speakers to separate
        
    Returns:
        model: Combined speech enhancement model
    """
    # Load pretrained speaker model
    from speaker_verification.pretrained_eval import load_pretrained_model
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
        speaker_model.load_state_dict(torch.load(speaker_model_path, map_location=device), strict=False)
    
    # Load SepFormer model
    sepformer_model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wham",
        savedir="models/speech_enhancement/sepformer"
    )
    
    # Create combined model
    model = CombinedSpeechEnhancementModel(
        speaker_model,
        sepformer_model,
        embedding_dim,
        num_speakers
    )
    
    return model.to(device)

if __name__ == "__main__":
    # Test the model
    import argparse
    
    parser = argparse.ArgumentParser(description="Test combined speech enhancement model")
    parser.add_argument("--speaker_model_path", type=str, default=None,
                        help="Path to pretrained/finetuned speaker model")
    parser.add_argument("--speaker_model_name", type=str, default="wavlm_base_plus",
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Name of the speaker model architecture")
    
    args = parser.parse_args()
    
    # Load model
    model = load_combined_model(args.speaker_model_path, args.speaker_model_name)
    
    # Print model architecture
    logger.info(f"Model architecture:")
    logger.info(model) 