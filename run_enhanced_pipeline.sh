#!/bin/bash

# Enhanced Speaker-Aware Speech Separation Pipeline
# This script runs the complete pipeline for multi-speaker separation and identification

# Set bash to exit on error
set -e

# Create required directories
mkdir -p data/vox2/mixtures_train
mkdir -p data/vox2/mixtures_test
mkdir -p data/vox2/separated_test
mkdir -p models/combined_pipeline
mkdir -p results/combined_pipeline

# Step 1: Create training mixtures
echo "Step 1: Creating training mixtures..."
python src/speech_enhancement/create_mixtures.py \
    --input_dir data/vox2/dev \
    --output_dir data/vox2/mixtures_train \
    --num_mixtures 1000 \
    --test_ratio 0.0 \
    > logs/step1_mixtures_train.log 2>&1

# Step 2: Create test mixtures
echo "Step 2: Creating test mixtures..."
python src/speech_enhancement/create_mixtures.py \
    --input_dir data/vox2/test \
    --output_dir data/vox2/mixtures_test \
    --num_mixtures 200 \
    --test_ratio 1.0 \
    > logs/step2_mixtures_test.log 2>&1

# Step 3: Run SepFormer on test set for baseline
echo "Step 3: Running SepFormer on test set for baseline..."
python src/speech_enhancement/run_separation.py \
    --test_metadata data/vox2/mixtures_test/metadata.csv \
    --separated_dir data/vox2/separated_test \
    --output_dir results/sepformer_baseline \
    --model_name wavlm_base_plus \
    > logs/step3_baseline_separation.log 2>&1

# Step 4: Train the enhanced combined model
echo "Step 4: Training enhanced combined model..."
python src/combined_pipeline/train_enhanced_pipeline.py \
    --train_metadata data/vox2/mixtures_train/metadata.csv \
    --val_metadata data/vox2/mixtures_test/metadata.csv \
    --output_dir models/combined_pipeline \
    --speaker_model_path models/speaker_verification/wavlm_ft/best_model.pt \
    --speaker_model_name wavlm_base_plus \
    --batch_size 4 \
    --epochs 10 \
    --learning_rate 1e-4 \
    > logs/step4_training.log 2>&1

# Step 5: Evaluate the enhanced combined model
echo "Step 5: Evaluating enhanced combined model..."
python src/combined_pipeline/evaluate_enhanced_pipeline.py \
    --test_metadata data/vox2/mixtures_test/metadata.csv \
    --model_path models/combined_pipeline/best_model.pt \
    --output_dir results/combined_pipeline \
    --speaker_model_path models/speaker_verification/wavlm_ft/best_model.pt \
    --pretrained_speaker_model_path models/speaker_verification/wavlm_pretrained.pt \
    --finetuned_speaker_model_path models/speaker_verification/wavlm_ft/best_model.pt \
    --speaker_model_name wavlm_base_plus \
    > logs/step5_evaluation.log 2>&1

# Step 6: Generate comparison report
echo "Step 6: Generating comparison report..."
python src/combined_pipeline/generate_comparision_report.py \
    --baseline_results results/sepformer_baseline/evaluation_results.csv \
    --enhanced_results results/combined_pipeline/evaluation_results.csv \
    --output_dir results/comparison \
    > logs/step6_report.log 2>&1

echo "Pipeline completed successfully!"
echo "Results available in results/combined_pipeline and results/comparison" 