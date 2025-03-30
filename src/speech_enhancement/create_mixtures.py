import os
import sys
import torch
import numpy as np
import pandas as pd
import random
import soundfile as sf
import torchaudio
from pathlib import Path
from tqdm import tqdm
import itertools
import logging
from pydub import AudioSegment

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import device, load_audio, setup_seed, save_audio, get_voxceleb_speaker_ids

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

def prepare_speaker_utterances(metadata_file, audio_root, speaker_ids):
    """
    Group utterances by speaker
    
    Args:
        metadata_file: Path to metadata file
        audio_root: Root directory containing audio files
        speaker_ids: List of speaker IDs to include
        
    Returns:
        Dict mapping speaker IDs to lists of audio file paths
    """
    # Read metadata
    df = pd.read_csv(metadata_file, sep=' ', names=['id', 'path', 'gender'])
    
    # Filter by speaker IDs
    df = df[df['id'].isin(speaker_ids)]
    
    # Group by speaker
    speaker_utterances = {}
    for speaker_id in speaker_ids:
        speaker_df = df[df['id'] == speaker_id]
        speaker_utterances[speaker_id] = [
            os.path.join(audio_root, path) for path in speaker_df['path'].tolist()
        ]
    
    return speaker_utterances

def create_mixture(audio_path1, audio_path2, snr=0, target_sr=16000):
    """
    Create a mixture of two audio files with a given SNR
    
    Args:
        audio_path1: Path to first audio file
        audio_path2: Path to second audio file
        snr: Signal-to-noise ratio in dB
        target_sr: Target sample rate
        
    Returns:
        mixed_audio: Mixed audio waveform
        clean1: First clean audio waveform
        clean2: Second clean audio waveform
    """
    # Load audio files
    waveform1, sr1 = load_audio(audio_path1, target_sr)
    waveform2, sr2 = load_audio(audio_path2, target_sr)
    
    # Convert to numpy for processing
    audio1 = waveform1.numpy()
    audio2 = waveform2.numpy()
    
    # Determine the shorter audio
    len1, len2 = len(audio1), len(audio2)
    min_len = min(len1, len2)
    
    # Crop or pad audios to the same length
    if len1 > min_len:
        start = random.randint(0, len1 - min_len)
        audio1 = audio1[start:start + min_len]
    elif len1 < min_len:
        padding = np.zeros(min_len - len1)
        audio1 = np.concatenate([audio1, padding])
    
    if len2 > min_len:
        start = random.randint(0, len2 - min_len)
        audio2 = audio2[start:start + min_len]
    elif len2 < min_len:
        padding = np.zeros(min_len - len2)
        audio2 = np.concatenate([audio2, padding])
    
    # Calculate RMS
    rms1 = np.sqrt(np.mean(audio1 ** 2))
    rms2 = np.sqrt(np.mean(audio2 ** 2))
    
    # Calculate scaling factor for SNR
    gain = rms1 / (rms2 * (10 ** (snr / 20)))
    
    # Apply gain to second audio
    audio2_scaled = audio2 * gain
    
    # Mix the two audios
    mixed_audio = audio1 + audio2_scaled
    
    # Normalize to avoid clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio = mixed_audio / max_val
        audio1 = audio1 / max_val
        audio2_scaled = audio2_scaled / max_val
    
    # Convert back to torch tensors
    mixed_audio_tensor = torch.from_numpy(mixed_audio)
    clean1_tensor = torch.from_numpy(audio1)
    clean2_tensor = torch.from_numpy(audio2_scaled)
    
    return mixed_audio_tensor, clean1_tensor, clean2_tensor

def create_multi_speaker_dataset(metadata_file, audio_root, output_dir, 
                                speaker_ids, num_mixtures=1000, snr_range=(-5, 5)):
    """
    Create a multi-speaker dataset by mixing utterances from different speakers
    
    Args:
        metadata_file: Path to metadata file
        audio_root: Root directory containing audio files
        output_dir: Directory to save mixed audio files
        speaker_ids: List of speaker IDs to include
        num_mixtures: Number of mixtures to create
        snr_range: Range of SNR values for mixing
        
    Returns:
        DataFrame with metadata for the mixed dataset
    """
    # Create output directories
    output_dir = Path(output_dir)
    mixed_dir = output_dir / "mixed"
    sources_dir = output_dir / "sources"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare speaker utterances
    speaker_utterances = prepare_speaker_utterances(metadata_file, audio_root, speaker_ids)
    
    # Create metadata dataframe
    metadata = []
    
    # Generate all possible speaker pairs
    all_pairs = list(itertools.combinations(speaker_ids, 2))
    
    # Shuffle pairs to ensure diversity
    random.shuffle(all_pairs)
    
    # Create mixtures
    progress_bar = tqdm(range(num_mixtures), desc="Creating mixtures")
    for i in progress_bar:
        # Get a random speaker pair
        pair_idx = i % len(all_pairs)
        speaker1, speaker2 = all_pairs[pair_idx]
        
        # Sample random utterances from each speaker
        utterance1 = random.choice(speaker_utterances[speaker1])
        utterance2 = random.choice(speaker_utterances[speaker2])
        
        # Generate random SNR within the specified range
        snr = random.uniform(snr_range[0], snr_range[1])
        
        # Create a mixture
        try:
            mixed, clean1, clean2 = create_mixture(utterance1, utterance2, snr)
            
            # Save audio files
            mixture_path = mixed_dir / f"mix_{i:05d}.wav"
            source1_path = sources_dir / f"s1_{i:05d}.wav"
            source2_path = sources_dir / f"s2_{i:05d}.wav"
            
            save_audio(mixed.unsqueeze(0), str(mixture_path))
            save_audio(clean1.unsqueeze(0), str(source1_path))
            save_audio(clean2.unsqueeze(0), str(source2_path))
            
            # Add to metadata
            metadata.append({
                'mixture_id': i,
                'mixture_path': str(mixture_path),
                'source1_path': str(source1_path),
                'source2_path': str(source2_path),
                'speaker1_id': speaker1,
                'speaker2_id': speaker2,
                'snr': snr
            })
        except Exception as e:
            logger.error(f"Error creating mixture {i}: {e}")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    
    logger.info(f"Created {len(metadata_df)} mixtures in {output_dir}")
    
    return metadata_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create multi-speaker mixtures")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to metadata file")
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Root directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save mixed audio files")
    parser.add_argument("--speaker_start", type=int, default=0,
                        help="Starting index for speaker IDs")
    parser.add_argument("--num_speakers", type=int, default=50,
                        help="Number of speakers to include")
    parser.add_argument("--num_mixtures", type=int, default=1000,
                        help="Number of mixtures to create")
    parser.add_argument("--snr_min", type=float, default=-5,
                        help="Minimum SNR value")
    parser.add_argument("--snr_max", type=float, default=5,
                        help="Maximum SNR value")
    
    args = parser.parse_args()
    
    # Get speaker IDs
    all_speakers = get_voxceleb_speaker_ids(args.metadata_file, sort=True)
    selected_speakers = all_speakers[args.speaker_start:args.speaker_start + args.num_speakers]
    
    # Create multi-speaker dataset
    create_multi_speaker_dataset(
        args.metadata_file,
        args.audio_root,
        args.output_dir,
        selected_speakers,
        args.num_mixtures,
        (args.snr_min, args.snr_max)
    ) 