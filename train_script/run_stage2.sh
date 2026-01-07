#!/bin/bash

# Stage 2 Training Script
# Text reconstruction from EEG features using Flan-T5 with LoRA

python -m train.train_stage2 \
    --data_size 3000 \
    --batch_size 8 \
    --max_epochs 10 \
    --lr 1e-4 \
    --freeze_strategy lora \
    --lora_rank 8 \
    --device cuda:0 \
    --log_dir ./logs \
    --experiment_name stage2 \
    --seed 42
