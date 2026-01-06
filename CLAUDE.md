# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLIM_CLS is a PyTorch Lightning-based deep learning project for EEG-based text classification. It combines a GLIM (Generalized Language-Image Model) encoder with an MLP classifier to perform classification tasks on EEG data from the ZuCo dataset.

**Core Architecture:**
- **GLIM Encoder**: Processes EEG signals (128 channels, 1280 length) into embeddings using transformer-based architecture with prompt embeddings
- **Text Model Integration**: Uses pre-trained language models (T5, BART) for text encoding
- **MLP Classifier**: Trainable classification head on top of frozen or fine-tuned encoder
- **Multi-loss Training**: Supports CLIP loss, language model loss, commitment loss, and classification loss with configurable weights

## Environment Setup

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate glim

# The environment includes:
# - PyTorch with Lightning 2.4.0
# - Transformers 4.40.*
# - PEFT 0.12.0 for parameter-efficient fine-tuning
# - TensorBoard for logging
# - sentence_transformers and keybert for embeddings
```

## Training

**Main training command:**
```bash
bash train_script/run_train_glim_cls.sh
```

**Direct Python invocation:**
```bash
python -m train.train_glim_cls \
    --data_path <path_to_merged_dataframe> \
    --classification_label_key "sentiment label" \
    --classification_labels non_neutral neutral \
    --text_model google/flan-t5-large \
    --batch_size 24 \
    --max_epochs 50 \
    --lr 1e-4 \
    --device 0
```

**Key training arguments:**
- `--classification_label_key`: Column name in dataframe for labels (e.g., "sentiment label", "topic_label")
- `--classification_labels`: Space-separated list of label names
- `--freeze_encoder`: Only train MLP classifier, freeze encoder weights
- `--mlp_hidden_dims`: Hidden layer sizes for MLP (e.g., `512 256`)
- `--device`: GPU indices as list (e.g., `0` or `0 1 2`)
- `--checkpoint`: Path to resume training from checkpoint

**Output structure:**
```
./logs/<experiment_name>_<timestamp>/
├── tensorboard/           # TensorBoard logs
├── checkpoints/          # Model checkpoints (top-3 + last)
├── training.log          # Console output
└── training_error.log    # Error logs
```

## Data Preprocessing Pipeline

The preprocessing must be run in order:

1. **preprocess_mat.py**: Convert raw .mat EEG files to dataframe format
   - Output: `zuco_eeg_128ch_1280len.df`

2. **preprocess_gen_lbl.py**: Generate classification labels (sentiment/topic)
   - Uses `generate_sentiment.py` for sentiment analysis
   - Uses `generate_embedding.py` for SBERT embeddings and keyword extraction
   - Output: `zuco_label_input_text.df`

3. **preprocess_merge.py**: Merge EEG data with labels
   - Handles typo corrections
   - Creates train/val/test splits (80/10/10)
   - Output: `zuco_merged.df`

4. **preprocess_merge_topic.py**: Merge topic labels (if using topic classification)
   - Output: `zuco_merged_with_topic.df`

**Important preprocessing notes:**
- Modify `tmp_path` variable in preprocessing scripts to set output directory
- Default preprocessing uses weighted sampling to handle class imbalance
- SBERT embeddings are 768-dimensional (all-mpnet-base-v2 model)

## Code Architecture

**Model hierarchy (model/):**
- `glim_cls.py`: Main GLIM_CLS Lightning module combining encoder + classifier
- `modules.py`: Core components (PromptEmbedder, EEGEncoder, Aligner)

**Data pipeline (data/):**
- `datamodule.py`: Lightning DataModule with custom samplers
  - `GLIMSampler`: Groups samples by text UID for batch consistency
  - `WeightedGLIMSampler`: Adds class balancing via weighted sampling
  - Uses `CLS_LABEL = 'topic_label'` constant for classification key

**Training (train/):**
- `train_glim_cls.py`: Main training script with comprehensive logging and metrics
  - Displays label distribution before training
  - Computes confusion matrix after testing
  - Uses cosine LR scheduler with warmup
  - TeeLogger redirects stdout/stderr to files

## Model Configuration

**Supported text models:**
- `google/flan-t5-xl`
- `google/flan-t5-large` (default)
- `facebook/bart-large-cnn`
- `jbochi/madlad400-3b-mt`

**Default dimensions:**
- Input EEG length: 1280
- Hidden EEG length: 96
- Input text length: 96
- Input dimension: 128
- Hidden dimension: 128
- Embedding dimension: 1024

**Prompt embeddings:**
- Three-level prompts: (task, dataset, subject)
- Default: (3, 3, 31) prompt numbers
- Configurable dropout per prompt level
- Can be disabled with `--do_not_use_prompt`

## Development Notes

- The codebase recently transitioned from hardcoded "sentiment label" to configurable classification labels
- Weighted sampling is enabled by default in DataModule to handle class imbalance
- Training uses bfloat16 precision by default for T5 models
- The model supports both joint training (encoder + MLP) and transfer learning (frozen encoder)
- Checkpoints save top-3 models by validation accuracy plus the last epoch
