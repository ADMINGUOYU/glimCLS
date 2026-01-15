#!/bin/bash

# Set Hugging Face cache directory
export HF_HOME="/mnt/afs/250010218/hf_cache"
export TRANSFORMERS_CACHE="/mnt/afs/250010218/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"

# Stage 1 Configuration
STAGE1_CHECKPOINT="/mnt/afs/250010218/glimCLS/logs/glim_parallel_20260110_175544/checkpoints/model-epoch43-acc_topic0.9072.ckpt"
FREEZE_STAGE1=false

# Stage 2 Configuration
TEXT_MODEL="google/flan-t5-large"
MODEL_CACHE_DIR="/mnt/afs/250010218/hf_cache"
FREEZE_STRATEGY="full_trainable_llm" #options: "lora" or "full_freeze_llm" or "full_trainable_llm"
LORA_RANK=8
USE_EI=true
USE_PROJECTOR=true
PROMPT_TYPE="default"

# Loss Configuration
USE_ALIGN_LOSS=true  # Disable to save memory
W_ALIGN=0.2
USE_AUX_LOSS=true     # Enable auxiliary losses
# LLM loss weight
W_LLM=1.5
# s1 downstream task loss
W_SENTIMENT=0.25
W_TOPIC=0.25
W_LENGTH=0.25
W_SURPRISAL=0.25

# Optimizer Configuration
S1_LR=8e-5      # Lower LR for fine-tuning Stage 1
PROJ_LR=2e-4    # Higher LR for new projector
LLM_LR=1e-5     # Standard LR for LoRA
WEIGHT_DECAY=0.01
WARMUP_EPOCHS=3
MIN_LR=1e-6

# Data Configuration
EEG_DATA_PATH="./data/ZUCO1-2_FOR_GLIMCLS/zuco_merged_with_variants.df"
LABELS_DATA_PATH="./data/zuco_preprocessed_dataframe/zuco2best.df"
USE_MTV=true  # Enable MTV for 8x training data augmentation
BATCH_SIZE=24
NUM_WORKERS=4

# Training Configuration
MAX_EPOCHS=20
DEVICE=0
LOG_DIR="./logs/e2e"
EXPERIMENT_NAME="glim_stage2_e2e_mtv"

# Build command
CMD="python -m train.train_e2e \
    --stage1_checkpoint $STAGE1_CHECKPOINT \
    --text_model $TEXT_MODEL \
    --freeze_strategy $FREEZE_STRATEGY \
    --lora_rank $LORA_RANK \
    --prompt_type $PROMPT_TYPE \
    --w_llm $W_LLM \
    --w_sentiment $W_SENTIMENT \
    --w_topic $W_TOPIC \
    --w_length $W_LENGTH \
    --w_surprisal $W_SURPRISAL \
    --s1_lr $S1_LR \
    --proj_lr $PROJ_LR \
    --llm_lr $LLM_LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --min_lr $MIN_LR \
    --eeg_data_path $EEG_DATA_PATH \
    --labels_data_path $LABELS_DATA_PATH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_epochs $MAX_EPOCHS \
    --device $DEVICE \
    --log_dir $LOG_DIR \
    --experiment_name $EXPERIMENT_NAME"

# Add conditional flags
if [ "$FREEZE_STAGE1" = true ]; then
    CMD="$CMD --freeze_stage1"
fi

if [ "$USE_EI" = true ]; then
    CMD="$CMD --use_ei"
fi

if [ "$USE_PROJECTOR" = true ]; then
    CMD="$CMD --use_projector"
fi

if [ "$USE_ALIGN_LOSS" = true ]; then
    CMD="$CMD --use_align_loss --w_align $W_ALIGN"
fi

if [ "$USE_AUX_LOSS" = true ]; then
    CMD="$CMD --use_aux_loss"
fi

if [ "$USE_MTV" = true ]; then
    CMD="$CMD --use_mtv"
fi

if [ -n "$MODEL_CACHE_DIR" ]; then
    CMD="$CMD --model_cache_dir $MODEL_CACHE_DIR"
fi

# Execute
echo "Running E2E training with command:"
echo "$CMD"
eval $CMD
