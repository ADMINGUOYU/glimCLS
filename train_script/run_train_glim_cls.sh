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

python -m train.train_glim_cls \
    --data_path /nfs/usrhome2/yguoco/glim_cls/tmp/zuco_merged_with_variants.df \
    --classification_label_key "sentiment label" \
    --classification_labels non_neutral neutral \
    --text_model google/flan-t5-large \
    --hidden_dim 128 \
    --embed_dim 1024 \
    --n_in_blocks 6 \
    --n_out_blocks 6 \
    --num_heads 8 \
    --encoder_dropout 0.1 \
    --mlp_hidden_dims 512 256 \
    --mlp_dropout 0.3 \
    --clip_loss_weight 0.5 \
    --lm_loss_weight 0.5 \
    --commitment_loss_weight 0.5 \
    --mlp_loss_weight 0.5 \
    --batch_size 24 \
    --val_batch_size 24 \
    --max_epochs 50 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_steps 0 \
    --log_dir ./logs \
    --experiment_name glim_cls \
    --accelerator auto \
    --device 0 \
    --precision bf16-mixed \
    --seed 42 \
    --num_workers 4

# Optional flags (uncomment as needed):
# --do_not_use_prompt    # Disable prompt embeddings
# --freeze_encoder       # Freeze encoder weights (only train MLP)
# --early_stopping       # Enable early stopping
# --patience 10          # Early stopping patience
# --checkpoint ""        # Path to checkpoint to resume training