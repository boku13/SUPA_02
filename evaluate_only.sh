#!/bin/bash

# Set variables
MODEL="wavlm_base_plus"
TRAIN_METADATA="data/vox2/vox2_train.csv"
VOX1_AUDIO_ROOT="data/vox1/wav"
TRIAL_FILE="data/vox1/veri_test2.txt"
OUTPUT_DIR="models/speaker_verification/wavlm_ft"
FINETUNED_MODEL_PATH="$OUTPUT_DIR/best_model.pt"
BATCH_SIZE=4

# Check if the fine-tuned model exists
if [ ! -f "$FINETUNED_MODEL_PATH" ]; then
    echo "Error: Fine-tuned model not found at $FINETUNED_MODEL_PATH"
    echo "Please run the fine-tuning process first or provide a valid model path."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Evaluate and compare models
echo "Starting model evaluation and comparison..."
python src/speaker_verification/evaluate_models.py \
    --model $MODEL \
    --trial_file $TRIAL_FILE \
    --vox1_audio_root $VOX1_AUDIO_ROOT \
    --train_metadata $TRAIN_METADATA \
    --finetuned_model_path $FINETUNED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE

echo "Evaluation complete! Results are available in $OUTPUT_DIR directory." 