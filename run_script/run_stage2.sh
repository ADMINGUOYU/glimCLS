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

# Model Architecture
TEXT_MODEL="google/flan-t5-large"
FREEZE_STRATEGY="full_freeze_llm"  # Options: "lora" or "full_freeze_llm"
LORA_RANK=8

# Label Embedding Initialization (Optional)
# Leave empty ("") to use random initialization
# Provide path to checkpoint containing pre-trained label embeddings
LABEL_EMBED_INIT=""

# Data
DATA_SIZE=3000
BATCH_SIZE=8
SEED=42

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
echo "  Data Size: $DATA_SIZE"
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
    --data_size $DATA_SIZE \
    --batch_size $BATCH_SIZE \
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

if [ -n "$LABEL_EMBED_INIT" ]; then
    CMD="$CMD --label_embed_init \"$LABEL_EMBED_INIT\""
fi

# Execute command
eval $CMD
