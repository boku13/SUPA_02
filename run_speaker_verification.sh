#!/bin/bash

# Set variables
MODEL="wavlm_base_plus"
TRAIN_METADATA="data/vox2/vox2_train.csv"
VAL_METADATA="data/vox2/vox2_test.csv"
VOX2_AUDIO_ROOT="data/vox2/aac"
VOX1_AUDIO_ROOT="data/vox1/wav"
TRIAL_FILE="data/vox1/veri_test2.txt"
OUTPUT_DIR="models/speaker_verification/wavlm_ft"
BATCH_SIZE=4
EPOCHS=2
MAX_FILES_PER_SPEAKER=10

# Create output directory
mkdir -p $OUTPUT_DIR

# 1. Fine-tune the model
echo "Starting fine-tuning process..."
python src/speaker_verification/finetune.py \
    --model $MODEL \
    --train_metadata $TRAIN_METADATA \
    --val_metadata $VAL_METADATA \
    --audio_root $VOX2_AUDIO_ROOT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --max_files_per_speaker $MAX_FILES_PER_SPEAKER \
    --use_gpu \
    --skip_evaluation

# Check if fine-tuning was successful
if [ ! -f "$OUTPUT_DIR/best_model.pt" ]; then
    echo "Fine-tuning failed! No model file was created."
    exit 1
fi

# 2. Evaluate and compare models
echo "Starting model evaluation and comparison..."
python src/speaker_verification/evaluate_models.py \
    --model $MODEL \
    --trial_file $TRIAL_FILE \
    --vox1_audio_root $VOX1_AUDIO_ROOT \
    --train_metadata $TRAIN_METADATA \
    --finetuned_model_path "$OUTPUT_DIR/best_model.pt" \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE

echo "Complete! Results are available in $OUTPUT_DIR directory." 