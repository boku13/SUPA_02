# Speech Understanding Assignment

This repository contains the implementation for a speech enhancement and speaker verification assignment using VoxCeleb datasets, as well as an Indian language classification task.

## Project Structure

```
.
├── data/                   # Data directory
│   ├── vox1/               # VoxCeleb1 dataset
│   └── vox2/               # VoxCeleb2 dataset
├── models/                 # Saved models
├── src/                    # Source code
│   ├── speaker_verification/  # Speaker verification modules
│   │   ├── prepare_voxceleb2.py    # Data preparation script
│   │   ├── finetune.py            # Fine-tuning script
│   │   ├── pretrained_eval.py     # Evaluation of pretrained models
│   │   └── test_audio_loading.py  # Utility for testing audio loading
│   ├── speech_enhancement/  # Speech enhancement modules
│   │   ├── create_mixtures.py  # Create multi-speaker mixtures
│   │   ├── sepformer.py    # SepFormer implementation
│   │   ├── evaluation.py   # Evaluation metrics (SIR, SAR, SDR, PESQ)
│   ├── combined_pipeline/  # Combined pipeline
│   │   ├── model.py        # Combined model architecture
│   │   ├── train.py        # Training pipeline
│   │   ├── evaluate.py     # Evaluation pipeline
│   └── indian_language_classification/  # Indian language classification
│       ├── mfcc_extraction.py  # Extract MFCC features
│       ├── language_classifier.py  # Language classification models
├── results/                # Results directory
├── notebooks/              # Jupyter notebooks for visualization and analysis
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Requirements

```
torch>=1.10.0
torchaudio>=0.10.0
transformers>=4.18.0
speechbrain>=0.5.12
peft>=0.4.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
librosa>=0.8.1
scipy>=1.7.0
pystoi>=0.3.3
pesq>=0.0.3
mir_eval>=0.6
scikit-learn>=1.0.0
seaborn>=0.11.0
pydub>=0.25.1
soundfile>=0.10.3
tqdm>=4.62.0
```

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the VoxCeleb datasets

## Tasks

### 1. Speech Enhancement and Speaker Verification (Question 1)

#### 1.1 Speaker Verification Evaluation and Fine-tuning

This task involves evaluating pre-trained speaker verification models on VoxCeleb1 and fine-tuning selected models using LoRA and ArcFace loss.

```bash
# Evaluate pretrained model
python src/speaker_verification/pretrained_eval.py \
  --model wavlm_base_plus \
  --trial_file data/vox1/trials_cleaned.txt \
  --audio_root data/vox1/wav \
  --output_dir results/speaker_verification/pretrained

# Fine-tune model
python src/speaker_verification/finetune.py \
  --model wavlm_base_plus \
  --train_metadata data/vox2/train_metadata.txt \
  --val_metadata data/vox2/val_metadata.txt \
  --audio_root data/vox2/wav \
  --output_dir models/speaker_verification/finetuned \
  --batch_size 8 \
  --epochs 5 \
  --lr 5e-5
```

#### 1.2 Multi-speaker Scenario Creation

Create a multi-speaker dataset by mixing utterances from different speakers.

```bash
# Create training mixtures
python src/speech_enhancement/create_mixtures.py \
  --metadata_file data/vox2/train_metadata.txt \
  --audio_root data/vox2/wav \
  --output_dir data/vox2/mixtures/train \
  --speaker_start 0 \
  --num_speakers 50 \
  --num_mixtures 1000

# Create test mixtures
python src/speech_enhancement/create_mixtures.py \
  --metadata_file data/vox2/train_metadata.txt \
  --audio_root data/vox2/wav \
  --output_dir data/vox2/mixtures/test \
  --speaker_start 50 \
  --num_speakers 50 \
  --num_mixtures 100
```

#### 1.3 Speech Enhancement with SepFormer

Apply SepFormer to perform speaker separation and speech enhancement.

```bash
# Apply SepFormer to test mixtures
python src/speech_enhancement/sepformer.py \
  --input_dir data/vox2/mixtures/test/mixed \
  --output_dir results/speech_enhancement/sepformer \
  --metadata_file data/vox2/mixtures/test/metadata.csv

# Evaluate separation performance
python src/speech_enhancement/evaluation.py \
  --ground_truth data/vox2/mixtures/test/metadata.csv \
  --separated_dir results/speech_enhancement/sepformer \
  --output_file results/speech_enhancement/evaluation.csv
```

#### 1.4 Combined Pipeline

Train and evaluate a novel pipeline that combines speaker identification with speech separation.

```bash
# Train combined model
python src/combined_pipeline/train.py \
  --train_metadata data/vox2/mixtures/train/metadata.csv \
  --val_metadata data/vox2/mixtures/test/metadata.csv \
  --speaker_model_path models/speaker_verification/finetuned/best_model.pt \
  --speaker_model_name wavlm_base_plus \
  --output_dir models/combined_pipeline \
  --batch_size 4 \
  --epochs 10 \
  --lr 1e-4

# Evaluate combined model
python src/combined_pipeline/evaluate.py \
  --test_metadata data/vox2/mixtures/test/metadata.csv \
  --model_path models/combined_pipeline/best_model.pt \
  --speaker_model_name wavlm_base_plus \
  --pretrained_speaker_model models/speaker_verification/pretrained/model.pt \
  --finetuned_speaker_model models/speaker_verification/finetuned/best_model.pt \
  --output_dir results/combined_pipeline
```

### 2. MFCC Feature Extraction and Indian Language Classification (Question 2)

#### 2.1 MFCC Feature Extraction

Extract MFCC features from Indian language audio samples.

```bash
# Extract MFCC features
python src/indian_language_classification/mfcc_extraction.py \
  --data_dir data/indian_languages \
  --output_dir results/indian_language_classification/mfcc \
  --selected_languages Hindi Tamil Telugu \
  --n_mfcc 13 \
  --max_samples 100
```

#### 2.2 Indian Language Classification

Train and evaluate classifiers for Indian language identification based on MFCC features.

```bash
# Train and evaluate classifiers
python src/indian_language_classification/language_classifier.py \
  --dataset_dir results/indian_language_classification/mfcc/dataset \
  --output_dir results/indian_language_classification/models \
  --label_map results/indian_language_classification/mfcc/dataset/label_encoder.pkl
```

## Results

Detailed analysis and results for each task can be found in the `results` directory. Visual representations of the results are saved as figures.

## Authors

- Your Name

## Acknowledgments

- VoxCeleb Dataset
- SpeechBrain SepFormer
- Hugging Face Transformers Library
- PyTorch 

# Speaker Verification System

This project implements a speaker verification system using pretrained audio models with fine-tuning on the VoxCeleb2 dataset.

## Overview

The speaker verification system uses state-of-the-art transformer-based audio models (WavLM, HuBERT, Wav2Vec2, etc.) and fine-tunes them for speaker recognition tasks using Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA (Low-Rank Adaptation). The system achieves high accuracy on speaker identification with minimal training resources.

## Project Structure

```
├── data/
│   ├── vox1/             # VoxCeleb1 dataset
│   └── vox2/             # VoxCeleb2 dataset with metadata
├── models/
│   └── speaker_verification/  # Fine-tuned models
├── src/
│   ├── speaker_verification/
│   │   ├── prepare_voxceleb2.py    # Data preparation script
│   │   ├── finetune.py            # Fine-tuning script
│   │   ├── pretrained_eval.py     # Evaluation of pretrained models
│   │   └── test_audio_loading.py  # Utility for testing audio loading
│   └── utils.py                   # Utility functions
└── README.md
```

## Features

- **Multiple Pretrained Models**: Supports WavLM Base Plus, HuBERT Large, Wav2Vec2 XLSR-53, and UniSpeech-SAT
- **Parameter-Efficient Fine-Tuning**: Uses LoRA for memory-efficient adaptation
- **ArcFace Loss**: Implements ArcFace for improved speaker embedding discrimination
- **Comprehensive Evaluation**: EER and TAR@FAR metrics for verification tasks
- **Robust Audio Loading**: Handles different audio formats with fallback mechanisms

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Set up datasets:
   - Download VoxCeleb2 dataset from official website
   - Place the dataset files in `data/vox2/aac/` directory

## Usage

### Data Preparation

Prepare the VoxCeleb2 dataset by generating metadata and splits:

```bash
python src/speaker_verification/prepare_voxceleb2.py
```

This will create:
- `data/vox2/vox2_metadata.csv`: Complete metadata
- `data/vox2/vox2_train.csv`: Training split
- `data/vox2/vox2_test.csv`: Testing split

### Testing Audio Loading

Before fine-tuning, test that audio files can be loaded correctly:

```bash
python src/speaker_verification/test_audio_loading.py --metadata data/vox2/vox2_train.csv --audio_root data/vox2/aac --num_samples 20
```

### Fine-tuning

Fine-tune a pretrained model:

```bash
python src/speaker_verification/finetune.py \
  --model wavlm_base_plus \
  --train_metadata data/vox2/vox2_train.csv \
  --val_metadata data/vox2/vox2_test.csv \
  --audio_root data/vox2/aac \
  --output_dir models/speaker_verification/wavlm_ft \
  --batch_size 8 \
  --epochs 5 \
  --lr 5e-5
```

## Model Details

### WavLM Base Plus

WavLM is a self-supervised speech model pre-trained on large-scale unlabeled audio data. The Base+ variant has 94.9M parameters and provides 768-dimensional embeddings.

### ArcFace Loss

ArcFace introduces an additive angular margin in the softmax loss to enhance discriminative power for face verification, which is adapted for speaker verification in this project. The loss encourages larger margins between different speakers.

### LoRA Fine-tuning

LoRA enables efficient fine-tuning by adding low-rank decomposition matrices to transformer layers that reduce the number of trainable parameters by 90%+ compared to full fine-tuning.

## Performance

Typical performance metrics on VoxCeleb2 test set:

| Model | Accuracy | EER |
|-------|----------|-----|
| WavLM Base Plus | ~95% | ~2.5% |
| HuBERT Large | ~94% | ~3.0% |
| Wav2Vec2 XLSR | ~93% | ~3.5% |

## Troubleshooting

If you encounter issues with audio loading:
1. Check that PyAV is installed correctly for m4a file handling
2. Ensure the audio path in metadata points to valid files
3. Try running the `test_audio_loading.py` script to diagnose problems

## References

1. Chen, S. et al. (2022). [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
2. Deng, J. et al. (2019). [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
3. Hu, E. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
4. Chung, J.S. et al. (2018). [VoxCeleb2: Deep Speaker Recognition](https://arxiv.org/abs/1806.05622)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 