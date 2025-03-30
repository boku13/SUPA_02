import os
import torch
import numpy as np
import pandas as pd
import torchaudio
import librosa
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Optional
from pathlib import Path
import random
import av
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def setup_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_voxceleb_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Parse VoxCeleb metadata file and return as DataFrame
    
    Args:
        metadata_path: Path to the metadata file (txt format)
        
    Returns:
        DataFrame with columns: id, path, gender
    """
    data = []
    with open(metadata_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:  # Ensure we have at least ID and path
                speaker_id = parts[0]
                path = parts[1]
                gender = parts[2] if len(parts) > 2 else None
                data.append((speaker_id, path, gender))
    
    return pd.DataFrame(data, columns=['id', 'path', 'gender'])

def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and convert to desired sample rate
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (waveform tensor, sample rate)
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        if audio_path.endswith('.m4a'):
            # Use PyAV for m4a files
            try:
                container = av.open(audio_path)
                audio = container.streams.audio[0]
                # Decode all frames
                signal = []
                for frame in container.decode(audio):
                    signal.append(frame.to_ndarray())
                
                # Concatenate frames
                if signal:
                    signal = np.concatenate(signal, axis=1).reshape(-1)
                    waveform = torch.tensor(signal, dtype=torch.float32)
                    sample_rate = audio.sample_rate
                else:
                    logger.warning(f"No audio frames found in {audio_path}, returning empty tensor")
                    waveform = torch.zeros(target_sr, dtype=torch.float32)  # 1 second of silence
                    sample_rate = target_sr
                
                # Convert sample rate if needed
                if sample_rate != target_sr:
                    waveform = torchaudio.functional.resample(
                        waveform,
                        sample_rate,
                        target_sr
                    )
                return waveform, target_sr
            except Exception as e:
                logger.warning(f"PyAV failed to load {audio_path}: {e}. Trying with librosa...")
                # Fall back to librosa
                signal, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                waveform = torch.tensor(signal, dtype=torch.float32)
                return waveform, target_sr
        else:
            # Use torchaudio for other formats
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                waveform = waveform.squeeze(0)  # Remove channel dimension
                
                # Convert sample rate if needed
                if sample_rate != target_sr:
                    waveform = torchaudio.functional.resample(
                        waveform,
                        sample_rate,
                        target_sr
                    )
                return waveform, target_sr
            except Exception as e:
                logger.warning(f"torchaudio failed to load {audio_path}: {e}. Trying with librosa...")
                # Fall back to librosa
                signal, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                waveform = torch.tensor(signal, dtype=torch.float32)
                return waveform, target_sr
    
    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        # Return a small amount of silence in case of error
        return torch.zeros(target_sr, dtype=torch.float32), target_sr

def extract_speaker_embeddings(model, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Extract speaker embeddings using a pretrained model
    
    Args:
        model: Pretrained speaker verification model
        audio: Audio waveform tensor
        sample_rate: Sample rate of the audio
        
    Returns:
        Speaker embedding tensor
    """
    # Ensure input is on the correct device
    audio = audio.to(device)
    
    # Make sure audio is in correct shape (batch_size, seq_len)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    with torch.no_grad():
        # Forward pass through the model
        embedding = model(audio)
        
    return embedding

def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform") -> None:
    """
    Plot a waveform
    
    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        title: Title for the plot
    """
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.numpy())
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(waveform: torch.Tensor, sample_rate: int, title: str = "Spectrogram") -> None:
    """
    Plot a spectrogram
    
    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        title: Title for the plot
    """
    waveform = waveform.numpy()
    
    # Generate spectrogram
    D = librosa.stft(waveform)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_voxceleb_speaker_ids(metadata_path: str, n_speakers: int = None, 
                            sort: bool = True) -> List[str]:
    """
    Get a list of speaker IDs from VoxCeleb metadata
    
    Args:
        metadata_path: Path to the metadata file
        n_speakers: Number of speakers to return (None for all)
        sort: Whether to sort speaker IDs
        
    Returns:
        List of speaker IDs
    """
    df = get_voxceleb_metadata(metadata_path)
    unique_ids = df['id'].unique()
    
    if sort:
        unique_ids.sort()
    
    if n_speakers is not None:
        unique_ids = unique_ids[:n_speakers]
    
    return unique_ids.tolist()

def save_audio(waveform: torch.Tensor, file_path: str, sample_rate: int = 16000) -> None:
    """
    Save audio waveform to file
    
    Args:
        waveform: Audio waveform tensor
        file_path: Path to save the audio file
        sample_rate: Sample rate of the audio
    """
    # Ensure waveform is 2D for torchaudio.save
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    torchaudio.save(file_path, waveform, sample_rate) 