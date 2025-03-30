import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
import soundfile as sf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

def extract_mfcc_features(audio_file, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from an audio file
    
    Args:
        audio_file: Path to audio file
        n_mfcc: Number of MFCC coefficients to extract
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        mfcc_features: MFCC features (n_mfcc x time)
    """
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return mfcc_features, sr

def visualize_mfcc(mfcc_features, sr, hop_length=512, title="MFCC Spectrogram"):
    """
    Visualize MFCC features
    
    Args:
        mfcc_features: MFCC features
        sr: Sample rate
        hop_length: Hop length used for extraction
        title: Plot title
    """
    # Create figure
    plt.figure(figsize=(10, 4))
    
    # Display MFCC
    librosa.display.specshow(
        mfcc_features,
        x_axis='time',
        sr=sr,
        hop_length=hop_length
    )
    
    # Add colorbar and title
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def process_audio_dataset(data_dir, output_dir, selected_languages=None, 
                          n_mfcc=13, max_samples_per_language=100):
    """
    Process audio dataset and extract MFCC features
    
    Args:
        data_dir: Directory containing audio files
        output_dir: Directory to save extracted features and visualizations
        selected_languages: List of selected languages to process (None for all)
        n_mfcc: Number of MFCC coefficients
        max_samples_per_language: Maximum number of samples to process per language
        
    Returns:
        DataFrame with audio metadata and features
    """
    # Create output directories
    output_dir = Path(output_dir)
    features_dir = output_dir / "features"
    plots_dir = output_dir / "plots"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    data_dir = Path(data_dir)
    
    # Initialize results
    results = []
    
    # Process each language folder
    for language_dir in sorted(data_dir.iterdir()):
        if not language_dir.is_dir():
            continue
        
        language = language_dir.name
        
        # Skip if not in selected languages
        if selected_languages and language not in selected_languages:
            continue
        
        logger.info(f"Processing language: {language}")
        
        # Create language-specific directories
        language_features_dir = features_dir / language
        language_plots_dir = plots_dir / language
        
        language_features_dir.mkdir(exist_ok=True)
        language_plots_dir.mkdir(exist_ok=True)
        
        # Process audio files for this language
        audio_files = list(language_dir.glob("*.wav"))
        
        # Limit number of samples
        if max_samples_per_language and len(audio_files) > max_samples_per_language:
            audio_files = audio_files[:max_samples_per_language]
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"Processing {language}"):
            try:
                # Extract MFCC
                mfcc_features, sr = extract_mfcc_features(audio_file, n_mfcc=n_mfcc)
                
                # Save features
                feature_file = language_features_dir / f"{audio_file.stem}_mfcc.npy"
                np.save(feature_file, mfcc_features)
                
                # Create visualization
                fig = visualize_mfcc(
                    mfcc_features, 
                    sr, 
                    title=f"MFCC Spectrogram - {language} - {audio_file.stem}"
                )
                
                # Save plot
                plot_file = language_plots_dir / f"{audio_file.stem}_mfcc.png"
                fig.savefig(plot_file)
                plt.close(fig)
                
                # Calculate statistics
                mfcc_mean = np.mean(mfcc_features, axis=1)
                mfcc_var = np.var(mfcc_features, axis=1)
                
                # Add to results
                results.append({
                    'file_path': str(audio_file),
                    'language': language,
                    'feature_path': str(feature_file),
                    'plot_path': str(plot_file),
                    'mfcc_mean': mfcc_mean.tolist(),
                    'mfcc_var': mfcc_var.tolist(),
                    'duration': librosa.get_duration(y=None, sr=sr, filename=str(audio_file))
                })
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save metadata
    results_df.to_csv(output_dir / "mfcc_features_metadata.csv", index=False)
    
    logger.info(f"Processed {len(results_df)} audio files")
    
    return results_df

def prepare_feature_dataset(metadata_file, output_dir):
    """
    Prepare dataset for classification
    
    Args:
        metadata_file: Path to metadata CSV file
        output_dir: Directory to save prepared dataset
        
    Returns:
        X_train, X_test, y_train, y_test: Train-test split
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    
    # Initialize lists
    X = []
    y = []
    
    # Process each file
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Preparing dataset"):
        # Load MFCC features
        feature_path = row['feature_path']
        mfcc_features = np.load(feature_path)
        
        # Calculate statistics (mean and variance along time axis)
        mfcc_mean = np.mean(mfcc_features, axis=1)
        mfcc_var = np.var(mfcc_features, axis=1)
        
        # Concatenate mean and variance for feature vector
        feature_vector = np.concatenate((mfcc_mean, mfcc_var))
        
        # Add to dataset
        X.append(feature_vector)
        y.append(row['language'])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, output_dir / "scaler.pkl")
    
    # Save dataset
    np.save(output_dir / "X_train.npy", X_train_scaled)
    np.save(output_dir / "X_test.npy", X_test_scaled)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    
    logger.info(f"Prepared dataset with {len(X_train)} training samples and {len(X_test)} test samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def statistical_analysis(metadata_file, output_dir, selected_languages=None):
    """
    Perform statistical analysis on MFCC features
    
    Args:
        metadata_file: Path to metadata CSV file
        output_dir: Directory to save analysis results
        selected_languages: List of selected languages to analyze (None for all)
        
    Returns:
        DataFrame with statistical analysis
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    
    # Filter selected languages
    if selected_languages:
        metadata = metadata[metadata['language'].isin(selected_languages)]
    
    # Initialize results
    results = []
    
    # Group by language
    for language, group in metadata.groupby('language'):
        # Calculate statistics for each MFCC coefficient
        mfcc_means = np.array([eval(row['mfcc_mean']) for _, row in group.iterrows()])
        mfcc_vars = np.array([eval(row['mfcc_var']) for _, row in group.iterrows()])
        
        # Calculate statistics across all samples
        language_mfcc_mean = np.mean(mfcc_means, axis=0)
        language_mfcc_std = np.std(mfcc_means, axis=0)
        language_mfcc_var_mean = np.mean(mfcc_vars, axis=0)
        
        # Add to results
        results.append({
            'language': language,
            'num_samples': len(group),
            'mfcc_mean': language_mfcc_mean.tolist(),
            'mfcc_std': language_mfcc_std.tolist(),
            'mfcc_var_mean': language_mfcc_var_mean.tolist(),
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_dir / "language_statistics.csv", index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot MFCC means for each language
    for idx, row in results_df.iterrows():
        language = row['language']
        mfcc_mean = np.array(eval(row['mfcc_mean']) if isinstance(row['mfcc_mean'], str) else row['mfcc_mean'])
        plt.plot(range(len(mfcc_mean)), mfcc_mean, label=language, marker='o')
    
    plt.title("Average MFCC Coefficients by Language")
    plt.xlabel("MFCC Coefficient")
    plt.ylabel("Mean Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mfcc_means_comparison.png")
    
    # Create a boxplot of MFCC coefficient distributions
    plt.figure(figsize=(12, 8))
    all_languages = []
    all_coeffs = []
    
    for idx, row in results_df.iterrows():
        language = row['language']
        mfcc_mean = np.array(eval(row['mfcc_mean']) if isinstance(row['mfcc_mean'], str) else row['mfcc_mean'])
        for i, val in enumerate(mfcc_mean):
            all_languages.append(language)
            all_coeffs.append(i)
    
    plt.title("Distribution of MFCC Coefficients by Language")
    plt.xlabel("Language")
    plt.ylabel("MFCC Coefficient Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "mfcc_distributions.png")
    
    logger.info(f"Performed statistical analysis on {len(results_df)} languages")
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract MFCC features from audio dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted features and visualizations")
    parser.add_argument("--selected_languages", type=str, nargs='+', default=None,
                        help="List of selected languages to process (None for all)")
    parser.add_argument("--n_mfcc", type=int, default=13,
                        help="Number of MFCC coefficients")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to process per language")
    
    args = parser.parse_args()
    
    # Process audio dataset
    metadata = process_audio_dataset(
        args.data_dir,
        args.output_dir,
        args.selected_languages,
        args.n_mfcc,
        args.max_samples
    )
    
    # Prepare feature dataset
    X_train, X_test, y_train, y_test = prepare_feature_dataset(
        os.path.join(args.output_dir, "mfcc_features_metadata.csv"),
        os.path.join(args.output_dir, "dataset")
    )
    
    # Perform statistical analysis
    statistics = statistical_analysis(
        os.path.join(args.output_dir, "mfcc_features_metadata.csv"),
        os.path.join(args.output_dir, "analysis"),
        args.selected_languages
    ) 