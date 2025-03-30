import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from speechbrain.inference.separation import SepformerSeparation

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

def separate_mixture(mixture_path, output_dir, model):
    """
    Separate a mixed audio file into individual sources using SepFormer
    
    Args:
        mixture_path: Path to mixed audio file
        output_dir: Directory to save separated sources
        model: SepFormer model
        
    Returns:
        List of paths to separated sources
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get mixture ID from filename
    mixture_id = Path(mixture_path).stem
    
    # Separate sources directly
    # Note: SepFormer expects 8kHz audio
    logger.info(f"Separating mixture: {mixture_path}")
    est_sources = model.separate_file(path=str(mixture_path))
    logger.info(f"Separation complete, got {est_sources.shape[2]} sources")
    
    # Save separated sources
    source_paths = []
    for i in range(est_sources.shape[2]):  # Number of sources
        source = est_sources[:, :, i].detach().cpu()
        
        # Save the separated source
        output_path = output_dir / f"{mixture_id}_source{i+1}.wav"
        torchaudio.save(str(output_path), source, 8000)
        logger.info(f"Saved separated source {i+1} to {output_path}")
        
        source_paths.append(str(output_path))
    
    return source_paths

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
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SepFormer model for WHAMR dataset (instead of WHAM)
    # Provide run_opts to avoid symlink issues
    logger.info("Loading SepFormer model...")
    
    # Need to create a proper savedir that can be accessed without symlinks
    savedir = "models/sepformer_direct"
    os.makedirs(savedir, exist_ok=True)
    
    run_opts = {"device": device}
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr", 
        savedir=savedir,
        run_opts=run_opts
    )
    
    logger.info("Model loaded successfully")
    
    # Get list of mixture files
    if metadata_file:
        # Use metadata file
        logger.info(f"Loading metadata from {metadata_file}")
        metadata = pd.read_csv(metadata_file)
        mixture_files = metadata['mixture_path'].tolist()
        logger.info(f"Found {len(mixture_files)} mixtures in metadata file")
    else:
        # Use all wav files in input directory
        input_dir = Path(input_dir)
        logger.info(f"Scanning directory {input_dir} for WAV files")
        mixture_files = list(input_dir.glob("*.wav"))
        logger.info(f"Found {len(mixture_files)} WAV files in directory")
    
    # Process each mixture
    results = []
    for mixture_path in tqdm(mixture_files, desc="Separating mixtures"):
        mixture_id = Path(mixture_path).stem if isinstance(mixture_path, str) else mixture_path.stem
        logger.info(f"Processing mixture: {mixture_id}")
        
        # Create output subdirectory for this mixture
        mixture_output_dir = output_dir / mixture_id
        mixture_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Resample to 8kHz for SepFormer
            logger.info(f"Loading and resampling audio: {mixture_path}")
            waveform, sr = load_audio(str(mixture_path))
            resampled_path = mixture_output_dir / f"{mixture_id}_8k.wav"
            
            if sr != 8000:
                logger.info(f"Resampling from {sr}Hz to 8000Hz")
                resampled = torchaudio.functional.resample(waveform, sr, 8000)
                torchaudio.save(str(resampled_path), resampled.unsqueeze(0) if resampled.dim() == 1 else resampled, 8000)
            else:
                # Just save as is if already 8kHz
                logger.info("Audio already at 8kHz, no resampling needed")
                torchaudio.save(str(resampled_path), waveform.unsqueeze(0) if waveform.dim() == 1 else waveform, 8000)
            
            # Separate sources
            logger.info(f"Performing source separation")
            source_paths = separate_mixture(resampled_path, mixture_output_dir, model)
            
            # Add to results
            results.append({
                'mixture_id': mixture_id,
                'mixture_path': str(mixture_path),
                'source1_path': source_paths[0] if len(source_paths) > 0 else None,
                'source2_path': source_paths[1] if len(source_paths) > 1 else None
            })
        except Exception as e:
            logger.error(f"Error processing {mixture_path}: {e}")
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv_path = output_dir / "separated_sources.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved metadata to {results_csv_path}")
    
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
    
    # Modify output directory to save in results directory
    if not args.output_dir.startswith("results/"):
        original_output_dir = args.output_dir
        args.output_dir = f"results/speech_enhancement/{Path(original_output_dir).name}"
        logger.info(f"Changed output directory from {original_output_dir} to {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output will be saved to {args.output_dir}")
    
    # Run batch separation
    batch_separate(args.input_dir, args.output_dir, args.metadata_file) 