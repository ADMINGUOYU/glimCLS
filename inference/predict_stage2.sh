#!/bin/bash

# Inference script for Stage 2 text reconstruction model

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data path - output from Stage 1 inference (predict_glim_parallel_and_pack.py)
DATA_PATH="./data/zuco_preprocessed_dataframe/stage2.df"

# Model checkpoint - trained Stage 2 model
CHECKPOINT="./logs/...."

# Use metadata (prompts: dataset, task, subject)
USE_METADATA=false

# Output path
OUTPUT_DIR="..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${OUTPUT_DIR}/predictions_stage2_${TIMESTAMP}.csv"

# Model configuration (should match training)
TEXT_MODEL="google/flan-t5-large"
FREEZE_STRATEGY="lora"  # Options: "lora" or "full_freeze_llm" [will read from model state dict, no need to specify here]
LORA_RANK=8

# Label configuration (should match training)
SENTIMENT_LABELS=("non_neutral" "neutral")
TOPIC_LABELS=("Biographies and Factual Knowledge" "Movie Reviews and Sentiment")

# Generation parameters
MAX_LENGTH=50

# Inference settings
BATCH_SIZE=72
DEVICE="cuda:0"
SPLIT="test"         # Options: "train", "val", "test", "all"

# Verbosity
VERBOSE=true
NUM_SAMPLES_TO_PRINT=5

# ============================================================================

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Stage 2 Inference Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Checkpoint: $CHECKPOINT"
echo "  Use Metadata: $USE_METADATA"
echo "  Output Path: $OUTPUT_PATH"
echo "  Text Model: $TEXT_MODEL"
echo "  Freeze Strategy: $FREEZE_STRATEGY"
echo "  Split: $SPLIT"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Length: $MAX_LENGTH"
echo "  Device: $DEVICE"
echo "=========================================="
echo

# Build command
CMD="python -m inference.predict_stage2 \
    --data_path \"$DATA_PATH\" \
    --checkpoint \"$CHECKPOINT\" \
    --output_path \"$OUTPUT_PATH\" \
    --text_model \"$TEXT_MODEL\" \
    --freeze_strategy \"$FREEZE_STRATEGY\" \
    --lora_rank $LORA_RANK \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --device \"$DEVICE\" \
    --split \"$SPLIT\""
CMD+=(--sentiment_labels "${SENTIMENT_LABELS[@]}")
CMD+=(--topic_labels "${TOPIC_LABELS[@]}")

# Add use_metadata flag
if [ "$USE_METADATA" = true ]; then
    CMD="$CMD --use_metadata"
fi

# Run inference
echo "Running inference..."
eval $CMD

echo
echo "Done!  Predictions saved to: $OUTPUT_PATH"