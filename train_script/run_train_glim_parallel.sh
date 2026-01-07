#!/bin/bash

# Training script for GLIM_PARALLEL model
# Multi-task: 2 classification (sentiment, topic) + 2 regression (length, surprisal)
#
# Output directory structure:
#   ./logs/<experiment_name>_<timestamp>/
#     ├── tensorboard/    # TensorBoard logs
#     ├── checkpoints/    # Model checkpoints
#     ├── training.log    # Console output log
#     └── training_error.log  # Error log

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data
DATA_PATH="data/zuco_preprocessed_dataframe/zuco_merged_with_variants.df"

# Classification tasks
SENTIMENT_LABELS=("non_neutral" "neutral")
TOPIC_LABELS=("Biographies and Factual Knowledge" "Movie Reviews and Sentiment")

# Model Architecture
TEXT_MODEL="google/flan-t5-large"
HIDDEN_DIM=128
EMBED_DIM=1024
N_IN_BLOCKS=6
N_OUT_BLOCKS=6
NUM_HEADS=8
ENCODER_DROPOUT=0.1
MLP_HIDDEN_DIMS="512 256"
MLP_DROPOUT=0.3

# Loss Weights (6 total)
CLIP_LOSS_WEIGHT=0.5
LM_LOSS_WEIGHT=0.5
COMMITMENT_LOSS_WEIGHT=0.7
SENTIMENT_LOSS_WEIGHT=0.3
TOPIC_LOSS_WEIGHT=0.3
LENGTH_LOSS_WEIGHT=0.3
SURPRISAL_LOSS_WEIGHT=0.3

# Training
BATCH_SIZE=72
VAL_BATCH_SIZE=24
MAX_EPOCHS=50
LR=2e-4
MIN_LR=1e-5
WARMUP_EPOCHS=15
SEED=42

# Hardware
ACCELERATOR="auto"
DEVICE=(0)
PRECISION="bf16-mixed"
NUM_WORKERS=4

# Logging
LOG_DIR="./logs"
EXPERIMENT_NAME="glim_parallel"

# ============================================================================

echo "=========================================="
echo "Training Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Sentiment Labels: ${SENTIMENT_LABELS[@]}"
echo "  Topic Labels: ${TOPIC_LABELS[@]}"
echo "  Data Path: $DATA_PATH"
echo "=========================================="
echo

python -m train.train_glim_parallel \
    --data_path "$DATA_PATH" \
    --sentiment_labels "${SENTIMENT_LABELS[@]}" \
    --topic_labels "${TOPIC_LABELS[@]}" \
    --text_model "$TEXT_MODEL" \
    --hidden_dim $HIDDEN_DIM \
    --embed_dim $EMBED_DIM \
    --n_in_blocks $N_IN_BLOCKS \
    --n_out_blocks $N_OUT_BLOCKS \
    --num_heads $NUM_HEADS \
    --encoder_dropout $ENCODER_DROPOUT \
    --mlp_hidden_dims $MLP_HIDDEN_DIMS \
    --mlp_dropout $MLP_DROPOUT \
    --clip_loss_weight $CLIP_LOSS_WEIGHT \
    --lm_loss_weight $LM_LOSS_WEIGHT \
    --commitment_loss_weight $COMMITMENT_LOSS_WEIGHT \
    --sentiment_loss_weight $SENTIMENT_LOSS_WEIGHT \
    --topic_loss_weight $TOPIC_LOSS_WEIGHT \
    --length_loss_weight $LENGTH_LOSS_WEIGHT \
    --surprisal_loss_weight $SURPRISAL_LOSS_WEIGHT \
    --batch_size $BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --log_dir "$LOG_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --accelerator "$ACCELERATOR" \
    --device "${DEVICE[@]}" \
    --precision "$PRECISION" \
    --seed $SEED \
    --num_workers $NUM_WORKERS

# Optional flags (uncomment as needed):
# --do_not_use_prompt    # Disable prompt embeddings
# --freeze_encoder       # Freeze encoder weights (only train MLPs)
# --early_stopping       # Enable early stopping
# --patience 10          # Early stopping patience
# --checkpoint ""        # Path to checkpoint to resume training
