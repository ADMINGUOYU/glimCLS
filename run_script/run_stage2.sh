#!/bin/bash

# Training script for Stage 2 text reconstruction model
# Flan-T5 with LoRA for EEG-to-text reconstruction
#
# Output directory structure:
#   ./logs/stage2_<timestamp>/
#     ├── tensorboard/    # TensorBoard logs
#     ├── checkpoints/    # Model checkpoints
#     ├── training.log    # Console output log
#     └── training_error.log  # Error log

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data
# Set DATA_PATH to use real dataset, leave empty ("") for mock data
DATA_PATH=""  # e.g., "./data/zuco_preprocessed_dataframe/stage2.df"
DATA_SIZE=3000  # Only used when DATA_PATH is empty (mock data)
BATCH_SIZE=72
NUM_WORKERS=4
SEED=42

# Label configuration (only used when DATA_PATH is set)
SENTIMENT_LABELS=("non_neutral" "neutral")
TOPIC_LABELS=("Biographies and Factual Knowledge" "Movie Reviews and Sentiment")

# Model Architecture
TEXT_MODEL="google/flan-t5-large"
FREEZE_STRATEGY="full_freeze_llm"  # Options: "lora" or "full_freeze_llm"
LORA_RANK=8

# Label Embedding Initialization (Optional)
# Leave empty ("") to use random initialization
# Provide path to checkpoint containing pre-trained label embeddings
LABEL_EMBED_INIT=""

# Training
MAX_EPOCHS=10
LR=1e-4
MIN_LR=1e-6
WARMUP_EPOCHS=0
WEIGHT_DECAY=0.01

# Hardware
DEVICE="cuda:0"

# Logging
LOG_DIR="./logs"
EXPERIMENT_NAME="stage2"

# ============================================================================

echo "=========================================="
echo "Stage 2 Training Configuration:"
echo "  Text Model: $TEXT_MODEL"
echo "  Freeze Strategy: $FREEZE_STRATEGY"
echo "  LoRA Rank: $LORA_RANK"
if [ -n "$DATA_PATH" ]; then
    echo "  Data Path: $DATA_PATH"
    echo "  Sentiment Labels: ${SENTIMENT_LABELS[@]}"
    echo "  Topic Labels: ${TOPIC_LABELS[@]}"
else
    echo "  Data: Mock dataset"
    echo "  Data Size: $DATA_SIZE"
fi
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Learning Rate: $LR (max), $MIN_LR (min)"
echo "  Warmup Epochs: $WARMUP_EPOCHS"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Device: $DEVICE"
if [ -n "$LABEL_EMBED_INIT" ]; then
    echo "  Label Embed Init: $LABEL_EMBED_INIT"
else
    echo "  Label Embed Init: Random initialization"
fi
echo "=========================================="
echo

# Build command with optional label embedding argument
CMD="python -m train.train_stage2 \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --weight_decay $WEIGHT_DECAY \
    --text_model \"$TEXT_MODEL\" \
    --freeze_strategy \"$FREEZE_STRATEGY\" \
    --lora_rank $LORA_RANK \
    --device \"$DEVICE\" \
    --log_dir \"$LOG_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\" \
    --seed $SEED"

# Add data path or data size
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path \"$DATA_PATH\""
    CMD="$CMD --sentiment_labels ${SENTIMENT_LABELS[@]}"
    CMD="$CMD --topic_labels ${TOPIC_LABELS[@]}"
else
    CMD="$CMD --data_size $DATA_SIZE"
fi

# Add label embedding initialization if provided
if [ -n "$LABEL_EMBED_INIT" ]; then
    CMD="$CMD --label_embed_init \"$LABEL_EMBED_INIT\""
fi

# Execute command
eval $CMD