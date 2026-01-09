#!/bin/bash
# Run multi-checkpoint prediction script

# Set paths to your checkpoints
SENTIMENT_CKPT="./logs/...ckpt"
TOPIC_CKPT="./logs/...ckpt"
LENGTH_CKPT="./logs/...ckpt"
SURPRISAL_CKPT="./logs/...ckpt"

# Set data path
DATA_PATH="./data/zuco_preprocessed_dataframe/zuco_merged_with_variants.df"

# Set output path
OUTPUT_PATH="./data/zuco_preprocessed_dataframe/stage2.df"

# Create output directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Run prediction
python -m inference.predict_glim_parallel_and_pack \
    --data_path "$DATA_PATH" \
    --sentiment_checkpoint "$SENTIMENT_CKPT" \
    --topic_checkpoint "$TOPIC_CKPT" \
    --length_checkpoint "$LENGTH_CKPT" \
    --surprisal_checkpoint "$SURPRISAL_CKPT" \
    --output_path "$OUTPUT_PATH" \
    --device 0 \
    --batch_size 72 \
    --split all # run predict on all 'train', 'test' and 'val' splits