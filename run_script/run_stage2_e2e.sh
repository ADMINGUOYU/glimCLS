#!/bin/bash

# Training script for End-to-End Stage 2 model
# Combines glim_parallel encoder with T5 decoder for text reconstruction
#
# Output directory structure:
#   ./logs/stage2_e2e_<timestamp>/
#     ├── tensorboard/    # TensorBoard logs
#     ├── checkpoints/    # Model checkpoints
#     ├── training.log    # Console output log
#     └── training_error.log  # Error log

# Set Hugging Face cache directory
export HF_HOME="/mnt/afs/250010218/hf_cache"
export TRANSFORMERS_CACHE="/mnt/afs/250010218/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"

# Create cache directory if it doesn't exist
mkdir -p "$HF_HOME"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data
DATA_PATH="./data/zuco_preprocessed_dataframe/zuco_merged.df"
BATCH_SIZE=16
VAL_BATCH_SIZE=16
TEST_BATCH_SIZE=16
NUM_WORKERS=4

# Checkpoint
STAGE1_CHECKPOINT="./logs/glim_parallel_best/checkpoints/best.ckpt"  # REQUIRED: Path to stage1 checkpoint
RESUME_CHECKPOINT=""  # Optional: Path to resume training from

# Encoder Architecture (from glim_parallel)
INPUT_EEG_LEN=1280
HIDDEN_EEG_LEN=96
INPUT_TEXT_LEN=96
INPUT_DIM=128
HIDDEN_DIM=128
EMBED_DIM=1024
N_IN_BLOCKS=6
N_OUT_BLOCKS=6
NUM_HEADS=8
MLP_RATIO=4
DROPOUT=0.0
USE_CHANNEL_WEIGHTS=false
USE_PROMPT=true

# Decoder Architecture (from stage2)
TEXT_MODEL="google/flan-t5-large"
FREEZE_STRATEGY="full_freeze_llm"  # Options: "full_freeze_llm", "lora", "full_trainable_llm"
LORA_RANK=8
USE_EI=true
USE_PROJECTOR=true
USE_METADATA=true

# Label configuration
SENTIMENT_LABELS=("non_neutral" "neutral")
TOPIC_LABELS=("Biographies and Factual Knowledge" "Movie Reviews and Sentiment")

# Trainability Control
ENCODER_TRAINABLE_MODE="aligner_only"  # Options: "all", "encoder_aligner", "aligner_only"

# Training
MAX_EPOCHS=15
LR=1e-4
MIN_LR=1e-6
WARMUP_EPOCHS=0

# Hardware
DEVICES=(0)  # GPU device IDs
ACCELERATOR="gpu"
STRATEGY="auto"  # Options: "auto", "ddp", etc.
PRECISION="bf16-mixed"

# Logging
LOG_DIR="./logs"
EXPERIMENT_NAME="stage2_e2e"

# ============================================================================

echo "=========================================="
echo "End-to-End Stage 2 Training Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Stage1 Checkpoint: $STAGE1_CHECKPOINT"
echo "  Encoder Trainable Mode: $ENCODER_TRAINABLE_MODE"
echo "  Text Model: $TEXT_MODEL"
echo "  Freeze Strategy: $FREEZE_STRATEGY"
echo "  Use Global EEG (ei): $USE_EI"
echo "  Use Projection Layer: $USE_PROJECTOR"
echo "  Use Metadata: $USE_METADATA"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Learning Rate: $LR (max), $MIN_LR (min)"
echo "  Warmup Epochs: $WARMUP_EPOCHS"
echo "  Devices: ${DEVICES[@]}"
echo "  Precision: $PRECISION"
echo "=========================================="
echo

# Build command
CMD="python -m train.train_stage2_e2e \
    --data_path \"$DATA_PATH\" \
    --stage1_checkpoint \"$STAGE1_CHECKPOINT\" \
    --batch_size $BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --input_eeg_len $INPUT_EEG_LEN \
    --hidden_eeg_len $HIDDEN_EEG_LEN \
    --input_text_len $INPUT_TEXT_LEN \
    --input_dim $INPUT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --embed_dim $EMBED_DIM \
    --n_in_blocks $N_IN_BLOCKS \
    --n_out_blocks $N_OUT_BLOCKS \
    --num_heads $NUM_HEADS \
    --mlp_ratio $MLP_RATIO \
    --dropout $DROPOUT \
    --text_model \"$TEXT_MODEL\" \
    --freeze_strategy \"$FREEZE_STRATEGY\" \
    --lora_rank $LORA_RANK \
    --encoder_trainable_mode \"$ENCODER_TRAINABLE_MODE\" \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --accelerator \"$ACCELERATOR\" \
    --strategy \"$STRATEGY\" \
    --precision \"$PRECISION\" \
    --log_dir \"$LOG_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\""

# Add device IDs
CMD="$CMD --devices ${DEVICES[@]}"

# Add sentiment labels
CMD="$CMD --sentiment_labels"
for label in "${SENTIMENT_LABELS[@]}"; do
    CMD="$CMD \"$label\""
done

# Add topic labels
CMD="$CMD --topic_labels"
for label in "${TOPIC_LABELS[@]}"; do
    CMD="$CMD \"$label\""
done

# Add optional flags
if [ "$USE_CHANNEL_WEIGHTS" = true ]; then
    CMD="$CMD --use_channel_weights"
fi

if [ "$USE_PROMPT" = false ]; then
    CMD="$CMD --do_not_use_prompt"
fi

if [ "$USE_EI" = true ]; then
    CMD="$CMD --use_ei"
fi

if [ "$USE_PROJECTOR" = true ]; then
    CMD="$CMD --use_projector"
fi

if [ "$USE_METADATA" = true ]; then
    CMD="$CMD --use_metadata"
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_checkpoint \"$RESUME_CHECKPOINT\""
fi

# Execute command
echo "Executing: $CMD"
echo
eval $CMD
