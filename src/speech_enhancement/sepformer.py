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
from speechbrain.pretrained import SepformerSeparation

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

class SepFormerWrapper:
    """Wrapper for pretrained SepFormer model"""
    def __init__(self, model_path=None):
        """
        Initialize SepFormer model
        
        Args:
            model_path: Path to pretrained model (None for default)
        """
        logger.info("Loading SepFormer model...")
        
        # Create run_opts to avoid symlink issues
        run_opts = {"device": "cpu"}  # First download to CPU to avoid CUDA issues
        
        try:
            # Make sure the directory exists
            os.makedirs("models/speech_enhancement/sepformer", exist_ok=True)
            
            # First step: Download the model files
            if model_path:
                # Load from local path if provided
                source_path = model_path
            else:
                # Use HuggingFace Hub
                source_path = "speechbrain/sepformer-wham"
            
            # Download model files only
            logger.info(f"Downloading model files from {source_path}")
            SepformerSeparation.from_hparams(
                source=source_path,
                savedir="models/speech_enhancement/sepformer",
                run_opts=run_opts,
                download_only=True
            )
            
            # Second step: Actually load the model
            logger.info("Loading the model into memory")
            self.model = SepformerSeparation.from_hparams(
                source=source_path,
                savedir="models/speech_enhancement/sepformer",
                run_opts={"device": device}  # Use the specified device for actual model
            )
            
            logger.info("Testing if model is loaded properly")
            if self.model is None:
                raise ValueError("Model failed to load properly")
            
            # Verify the model has the required methods
            if not hasattr(self.model, 'separate_file'):
                raise AttributeError("Model doesn't have 'separate_file' method")
                
            logger.info("SepFormer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SepFormer model: {e}")
            raise
    
    def separate(self, mixture_path, output_dir=None, save_results=False):
        """
        Separate a mixed audio file into individual sources
        
        Args:
            mixture_path: Path to mixed audio file
            output_dir: Directory to save separated sources
            save_results: Whether to save separated sources to files
            
        Returns:
            est_sources: Estimated sources (tensor)
        """
        # Verify model is loaded
        if self.model is None:
            raise ValueError("SepFormer model not loaded. Please initialize properly.")
        
        # Verify file exists
        if not os.path.exists(mixture_path):
            raise FileNotFoundError(f"Mixture file not found: {mixture_path}")
        
        try:
            # Load mixture
            mixture, fs = torchaudio.load(mixture_path)
            
            # Separate sources
            logger.info(f"Separating sources from {mixture_path}")
            est_sources = self.model.separate_file(mixture_path)
            
            # Save results if requested
            if save_results and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get filename without extension
                filename = Path(mixture_path).stem
                
                # Save each source
                for i, source in enumerate(est_sources):
                    output_path = output_dir / f"{filename}_source{i+1}.wav"
                    save_audio(source.unsqueeze(0), str(output_path))
            
            return est_sources
        except Exception as e:
            logger.error(f"Error separating mixture {mixture_path}: {e}")
            raise

def batch_separate(input_dir, output_dir, metadata_file=None):
    """
    Apply SepFormer separation to a directory of mixed audio files
    
    Args:
        input_dir: Directory containing mixed audio files
        output_dir: Directory to save separated sources
        metadata_file: Path to metadata file (optional)
        
    Returns:
        dataframe with paths to original mixtures and separated sources
    """
    # Initialize SepFormer model
    sepformer = SepFormerWrapper()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of mixture files
    if metadata_file:
        # Use metadata file
        metadata = pd.read_csv(metadata_file)
        mixture_files = metadata['mixture_path'].tolist()
    else:
        # Use all wav files in input directory
        input_dir = Path(input_dir)
        mixture_files = list(input_dir.glob("*.wav"))
    
    # Process each mixture
    results = []
    for mixture_path in tqdm(mixture_files, desc="Separating mixtures"):
        # Get mixture ID from filename
        if isinstance(mixture_path, str):
            mixture_id = Path(mixture_path).stem
        else:
            mixture_id = mixture_path.stem
        
        # Create output subdirectory for this mixture
        mixture_output_dir = output_dir / mixture_id
        mixture_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Separate sources
            est_sources = sepformer.separate(
                str(mixture_path), 
                mixture_output_dir, 
                save_results=True
            )
            
            # Create source paths
            source_paths = [
                str(mixture_output_dir / f"{mixture_id}_source{i+1}.wav") 
                for i in range(est_sources.shape[0])
            ]
            
            # Add to results
            results.append({
                'mixture_id': mixture_id,
                'mixture_path': str(mixture_path),
                'source1_path': source_paths[0],
                'source2_path': source_paths[1] if len(source_paths) > 1 else None
            })
        except Exception as e:
            logger.error(f"Error processing {mixture_path}: {e}")
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_dir / "separated_sources.csv", index=False)
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply SepFormer separation to mixed audio files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing mixed audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save separated sources")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to metadata file (optional)")
    
    args = parser.parse_args()
    
    # Run batch separation
    batch_separate(args.input_dir, args.output_dir, args.metadata_file) 