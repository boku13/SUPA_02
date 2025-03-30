# Speaker Verification with WavLM

This repository contains the code for fine-tuning and evaluating speaker verification models using the VoxCeleb1 and VoxCeleb2 datasets.

## Project Description

This project implements speaker verification using pretrained speech models. It includes:

1. **Fine-tuning** a pretrained model (WavLM, HuBERT, Wav2Vec2, or UniSpeech-SAT) on VoxCeleb2 dataset
2. **Evaluating** both the pretrained and fine-tuned models on VoxCeleb1 trial pairs
3. **Comparing** their performance using metrics like EER, TAR@1%FAR, and Speaker Identification Accuracy

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers library
- PEFT (Parameter-Efficient Fine-Tuning) library
- tqdm, pandas, numpy, matplotlib, sklearn
- GPU with CUDA support (recommended)

### Installation

Install the required dependencies using pip:

```bash
pip install torch torchaudio transformers peft pandas numpy matplotlib scikit-learn tqdm
```

### Data Preparation

1. Download the VoxCeleb1 dataset:
   - Audio files should be placed in `data/vox1/wav`
   - Trial file should be located at `data/vox1/veri_test2.txt`

2. Download the VoxCeleb2 dataset:
   - Audio files should be placed in `data/vox2/aac`
   - Train metadata file should be located at `data/vox2/vox2_train.csv`
   - Test metadata file should be located at `data/vox2/vox2_test.csv`

## Usage

### Running the Full Pipeline

To run both fine-tuning and evaluation in sequence:

```bash
bash run_speaker_verification.sh
```

This script will:
1. Fine-tune the model using the first 100 identities from VoxCeleb2
2. Evaluate both the pretrained and fine-tuned models on VoxCeleb1 trial pairs
3. Save results and comparison plots to the output directory

### Evaluation Only

If you already have a fine-tuned model and want to evaluate it:

```bash
bash evaluate_only.sh
```

### Custom Configuration

You can modify the scripts to change parameters:
- `MODEL`: Choose from "hubert_large", "wav2vec2_xlsr", "unispeech_sat", or "wavlm_base_plus"
- `BATCH_SIZE`: Adjust based on your GPU memory
- `EPOCHS`: Number of training epochs
- `MAX_FILES_PER_SPEAKER`: Limit the number of files per speaker (max 10)

## Key Components

- `src/speaker_verification/finetune.py`: Main script for fine-tuning models
- `src/speaker_verification/evaluate_models.py`: Script for evaluation and comparison
- `src/speaker_verification/pretrained_eval.py`: Utilities for evaluating pretrained models

## Evaluation Metrics

The comparison includes:
- **EER (Equal Error Rate)**: Lower is better
- **TAR@1%FAR (True Accept Rate at 1% False Accept Rate)**: Higher is better
- **Speaker Identification Accuracy**: Higher is better

## Results

After running the evaluation, results will be saved in:
- CSV file: `models/speaker_verification/wavlm_ft/comparison_results.csv`
- Plot: `models/speaker_verification/wavlm_ft/comparison_plot.png`

## Troubleshooting

- **GPU Memory Issues**: Reduce batch size or use a smaller model
- **Missing Dependencies**: Install required libraries using pip
- **File Not Found Errors**: Check path configurations in the scripts

## Citation

If you use this code, please cite the relevant papers:
- WavLM: Chen, S., et al. (2022). WavLM: Large-scale self-supervised pre-training for full stack speech processing.
- ArcFace: Deng, J., et al. (2019). ArcFace: Additive angular margin loss for deep face recognition.
- VoxCeleb: Nagrani, A., et al. (2017 & 2020). VoxCeleb: A large-scale speaker identification dataset. 