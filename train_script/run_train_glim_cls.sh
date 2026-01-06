#!/bin/bash

# Training script for GLIM_CLS model
# Combined GLIM encoder + MLP classifier for classification tasks
#
# Output directory structure:
#   ./logs/<experiment_name>_<timestamp>/
#     ├── tensorboard/    # TensorBoard logs
#     ├── checkpoints/    # Model checkpoints
#     ├── training.log    # Console output log
#     └── training_error.log  # Error log

# ============================================================================
# TASK CONFIGURATION - Change this to switch between tasks
# ============================================================================
# Options: "sentiment" or "topic"
TASK_TYPE="sentiment"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Task-specific configuration
if [ "$TASK_TYPE" = "sentiment" ]; then
    CLASSIFICATION_LABEL_KEY="sentiment label"
    CLASSIFICATION_LABELS=("non_neutral" "neutral")
    DATA_PATH="data/zuco_preprocessed_dataframe/zuco_merged_with_variants.df"
elif [ "$TASK_TYPE" = "topic" ]; then
    CLASSIFICATION_LABEL_KEY="topic_label"
    CLASSIFICATION_LABELS=("Biographies and Factual Knowledge" "Movie Reviews and Sentiment")
    DATA_PATH="data/zuco_preprocessed_dataframe/zuco_merged_with_variants.df"
else
    echo "Error: TASK_TYPE must be 'sentiment' or 'topic'"
    exit 1
fi

# Model Architecture
TEXT_MODEL="google/flan-t5-large"
HIDDEN_DIM=128
EMBED_DIM=1024
N_IN_BLOCKS=6
N_OUT_BLOCKS=6
NUM_HEADS=8
ENCODER_DROPOUT=0.1
MLP_HIDDEN_DIMS="512 256 128"
MLP_DROPOUT=0.3

# Loss Weights
CLIP_LOSS_WEIGHT=0.5
LM_LOSS_WEIGHT=0.5
COMMITMENT_LOSS_WEIGHT=0.7
MLP_LOSS_WEIGHT=0.3

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
DEVICE=0
PRECISION="bf16-mixed"
NUM_WORKERS=4

# Logging
LOG_DIR="./logs"
EXPERIMENT_NAME="glim_cls_${TASK_TYPE}"

# ============================================================================

echo "=========================================="
echo "Training Configuration:"
echo "  Task Type: $TASK_TYPE"
echo "  Label Key: $CLASSIFICATION_LABEL_KEY"
echo "  Labels: "${CLASSIFICATION_LABELS[@]}""
echo "  Data Path: $DATA_PATH"
echo "=========================================="
echo

python -m train.train_glim_cls \
    --data_path "$DATA_PATH" \
    --classification_label_key "$CLASSIFICATION_LABEL_KEY" \
    --classification_labels "${CLASSIFICATION_LABELS[@]}"  \
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
    --mlp_loss_weight $MLP_LOSS_WEIGHT \
    --batch_size $BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --log_dir "$LOG_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --accelerator "$ACCELERATOR" \
    --device $DEVICE \
    --precision "$PRECISION" \
    --seed $SEED \
    --num_workers $NUM_WORKERS

# Optional flags (uncomment as needed):
# --do_not_use_prompt    # Disable prompt embeddings
# --freeze_encoder       # Freeze encoder weights (only train MLP)
# --early_stopping       # Enable early stopping
# --patience 10          # Early stopping patience
# --checkpoint ""        # Path to checkpoint to resume training
