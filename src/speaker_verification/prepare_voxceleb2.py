#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to prepare VoxCeleb2 dataset for training.
This script creates metadata files and train/test splits.
"""

import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path("data")
VOX1_DIR = BASE_DIR / "vox1"
VOX2_DIR = BASE_DIR / "vox2"
VOX1_WAV_DIR = VOX1_DIR / "wav"
VOX2_AAC_DIR = VOX2_DIR / "aac"
VOX2_TXT_DIR = VOX2_DIR / "txt"
TRIAL_FILE = VOX1_DIR / "veri_test2.txt"
OUTPUT_DIR = Path("models/speaker_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        # List all directories (speaker IDs)
        speaker_ids = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                speaker_ids.append(item)
        
        return sorted(speaker_ids)
    except Exception as e:
        logger.error(f"Error getting speaker IDs from {directory}: {e}")
        return []

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
    
    # Process each speaker
    for speaker_id in tqdm(speaker_ids, desc="Processing speakers"):
        speaker_dir = os.path.join(VOX2_AAC_DIR, speaker_id)
        
        # Skip if directory doesn't exist
        if not os.path.exists(speaker_dir):
            logger.warning(f"Speaker directory not found: {speaker_dir}")
            continue
        
        # Get all m4a files for this speaker using os.walk for robustness
        try:
            for root, dirs, files in os.walk(speaker_dir):
                for file in files:
                    if file.endswith('.m4a'):
                        # Get full path
                        audio_file = os.path.join(root, file)
                        
                        # Get relative path
                        rel_path = os.path.relpath(audio_file, VOX2_AAC_DIR)
                        
                        # Add to metadata
                        metadata.append({
                            'id': speaker_id,
                            'path': rel_path,
                            'gender': 'unknown'  # Gender information not available
                        })
        except Exception as e:
            logger.error(f"Error processing speaker {speaker_id}: {e}")
            continue
    
    # Create dataframe
    df = pd.DataFrame(metadata)
    
    if len(df) == 0:
        logger.error("No metadata entries found. Check your dataset structure.")
        return df
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, sep=' ', index=False, header=False)
    
    logger.info(f"Created metadata file with {len(df)} entries at {output_file}")
    
    return df

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
    except Exception as e:
        logger.error(f"Error reading metadata file: {e}")
        # If file is not a proper CSV, read line by line
        try:
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
        except Exception as e2:
            logger.error(f"Second attempt to read metadata failed: {e2}")
            return None, None
    
    if len(metadata) == 0:
        logger.error("Metadata file is empty")
        return None, None
    
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

def prepare_voxceleb2():
    """
    Prepare VoxCeleb2 dataset for training
    
    Returns:
        tuple: (train_file, val_file)
    """
    # Check if paths exist
    if not os.path.exists(VOX2_AAC_DIR):
        logger.error(f"VoxCeleb2 audio directory not found: {VOX2_AAC_DIR}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Looking for: {os.path.abspath(VOX2_AAC_DIR)}")
        return None, None
    
    logger.info(f"Found VoxCeleb2 directory at: {VOX2_AAC_DIR}")
    
    # Get all speaker IDs
    logger.info("Getting speaker IDs from VoxCeleb2...")
    try:
        all_speaker_ids = get_speaker_ids(VOX2_AAC_DIR)
        
        if not all_speaker_ids:
            logger.error("No speaker IDs found in VoxCeleb2 directory")
            return None, None
            
        logger.info(f"Found {len(all_speaker_ids)} speakers in VoxCeleb2")
        
        # Take first 100 speakers for training, rest for testing
        train_speaker_ids = all_speaker_ids[:100]
        test_speaker_ids = all_speaker_ids[100:118]  # Take next 18 speakers for testing
        
        logger.info(f"Using {len(train_speaker_ids)} speakers for training")
        logger.info(f"Using {len(test_speaker_ids)} speakers for testing")
        
        # Create metadata file
        metadata_file = os.path.join(VOX2_DIR, "vox2_metadata.csv")
        
        train_file = os.path.join(VOX2_DIR, "vox2_train.csv")
        test_file = os.path.join(VOX2_DIR, "vox2_test.csv")
        
        # Create metadata and splits
        logger.info("Creating VoxCeleb2 metadata file...")
        metadata_df = create_voxceleb2_metadata(all_speaker_ids, metadata_file)
        
        if metadata_df is None or len(metadata_df) == 0:
            logger.error("Failed to create metadata - empty dataframe returned")
            return None, None
            
        logger.info("Creating train/test splits...")
        train_df, test_df = create_train_test_split(
            metadata_file, 
            train_speaker_ids, 
            test_speaker_ids,
            train_file,
            test_file
        )
        
        if train_df is None or test_df is None:
            logger.error("Failed to create train/test splits")
            return None, None
        
        logger.info("Successfully prepared VoxCeleb2 dataset!")
        return train_file, test_file
        
    except Exception as e:
        logger.error(f"Error preparing VoxCeleb2 dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

if __name__ == "__main__":
    # Only run when script is executed directly
    logger.info("Starting VoxCeleb2 dataset preparation...")
    train_file, test_file = prepare_voxceleb2()
    
    if train_file and test_file:
        logger.info(f"Successfully created training file: {train_file}")
        logger.info(f"Successfully created testing file: {test_file}")
        
        # Show sample entries
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                lines = f.readlines()[:5]  # Read first 5 lines
                if lines:
                    logger.info("Sample training data:")
                    for line in lines:
                        logger.info(f"  {line.strip()}")
    else:
        logger.error("Failed to prepare VoxCeleb2 dataset")
        sys.exit(1)
    
    logger.info("VoxCeleb2 dataset preparation completed successfully") 