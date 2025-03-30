#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify audio loading from VoxCeleb2 dataset.
This helps diagnose issues with audio loading before finetuning.
"""

import os
import sys
import torch
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_audio_loading(metadata_file, audio_root, num_samples=10):
    """
    Test audio loading from metadata file
    
    Args:
        metadata_file: Path to metadata file
        audio_root: Root directory containing audio files
        num_samples: Number of random samples to test
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    logger.info(f"Testing audio loading from {metadata_file}")
    logger.info(f"Audio root directory: {audio_root}")
    
    # Read metadata
    try:
        df = pd.read_csv(metadata_file, sep=' ', names=['id', 'path', 'gender'])
        logger.info(f"Loaded metadata with {len(df)} entries")
    except Exception as e:
        logger.error(f"Error reading metadata file: {e}")
        return False
    
    # Select random samples
    if len(df) > num_samples:
        samples = df.sample(num_samples)
    else:
        samples = df
    
    success_count = 0
    
    # Test each sample
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="Testing audio loading"):
        try:
            path = row['path']
            audio_path = os.path.join(audio_root, path)
            
            logger.info(f"Testing loading of {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"File not found: {audio_path}")
                continue
            
            # Try loading the audio
            waveform, sr = load_audio(audio_path)
            
            # Log success
            logger.info(f"Successfully loaded {audio_path}")
            logger.info(f"  Waveform shape: {waveform.shape}")
            logger.info(f"  Sample rate: {sr}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
    
    # Report success rate
    success_rate = success_count / len(samples) * 100
    logger.info(f"Audio loading test complete: {success_count}/{len(samples)} successful ({success_rate:.2f}%)")
    
    return success_count == len(samples)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio loading from VoxCeleb2 dataset")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata file")
    parser.add_argument("--audio_root", type=str, required=True, help="Root directory containing audio files")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of random samples to test")
    
    args = parser.parse_args()
    
    test_audio_loading(args.metadata, args.audio_root, args.num_samples) 