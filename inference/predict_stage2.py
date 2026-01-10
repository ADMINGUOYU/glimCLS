#!/usr/bin/env python
"""
Inference Script for Stage 2 Text Reconstruction Model
-> You can select which set ('train', 'val', 'test', 'all') to use

This script loads a trained Stage 2 model checkpoint and generates text predictions
from EEG embeddings (ei, Zi) and classification labels.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.stage2_model import Stage2ReconstructionModel
from data.stage2_dataset import Stage2ReconstructionDataset

# Import torchmetrics for evaluation (following GLIM approach)
from torchmetrics.functional.text import bleu_score, rouge_score, word_error_rate
from torchmetrics.functional.classification import multiclass_accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text predictions using trained Stage 2 model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Stage 2 DataFrame (output from predict_glim_parallel_and_pack.py)"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained Stage 2 model checkpoint (.pt/.ckpt file)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the predictions. Defaults to predictions_stage2_<timestamp>.csv"
    )
    
    # Model arguments
    parser.add_argument(
        "--text_model",
        type=str,
        default="google/flan-t5-large",
        help="Pre-trained text model (should match training)"
    )
    parser.add_argument(
        "--freeze_strategy",
        type=str,
        default="lora",
        choices=["lora", "full_freeze_llm"],
        help="Freeze strategy (should match training)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (should match training)"
    )
    
    # Label arguments
    parser.add_argument(
        "--sentiment_labels",
        type=str,
        nargs="+",
        default=["non_neutral", "neutral"],
        help="Sentiment label names (should match training)"
    )
    parser.add_argument(
        "--topic_labels",
        type=str,
        nargs="+",
        default=["Biographies and Factual Knowledge", "Movie Reviews and Sentiment"],
        help="Topic label names (should match training)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search (1 = greedy)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy/beam search"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used if --do_sample)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling (only used if --do_sample)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (only used if --do_sample)"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda: 0",
        help="Device to use for inference"
    )
    
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    
    # Dataset split
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which data split to predict on (default: test)"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample predictions during inference"
    )
    parser.add_argument(
        "--num_samples_to_print",
        type=int,
        default=5,
        help="Number of sample predictions to print (if --verbose)"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    text_model: str,
    freeze_strategy: str,
    lora_rank: int,
    device: str
) -> Stage2ReconstructionModel:
    """
    Load a Stage 2 model from a checkpoint.
    
    Args: 
        checkpoint_path: Path to the checkpoint file
        text_model: Pre-trained text model name
        freeze_strategy: Freeze strategy used during training
        lora_rank: LoRA rank used during training
        device: Device to load the model on
        
    Returns: 
        Loaded Stage2ReconstructionModel in eval mode
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to get hyperparameters from checkpoint args
    saved_args = checkpoint.get("args", {})
    
    # Use saved args if available, otherwise use provided args
    model_name = saved_args.get("text_model", text_model)
    strategy = saved_args.get("freeze_strategy", freeze_strategy)
    rank = saved_args.get("lora_rank", lora_rank)
    
    print(f"  Model: {model_name}")
    print(f"  Freeze strategy: {strategy}")
    print(f"  LoRA rank: {rank}")
    
    # Create model
    model = Stage2ReconstructionModel(
        model_name=model_name,
        freeze_strategy=strategy,
        lora_rank=rank,
        label_embed_init=None,
        device=device
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Set to eval mode
    model.eval()
    
    print("Model loaded successfully!")
    return model


def create_dataloader(
    df: pd.DataFrame,
    split: str,
    batch_size: int,
    sentiment_labels: List[str],
    topic_labels: List[str]
) -> Tuple[torch.utils.data.DataLoader, pd.DataFrame]:
    """
    Create a DataLoader for inference. 
    
    Args:
        df: DataFrame with Stage 2 data
        split: Data split to use
        batch_size: Batch size
        sentiment_labels: List of sentiment label names
        topic_labels: List of topic label names
        
    Returns:
        DataLoader and the filtered DataFrame
    """
    from torch.utils.data import DataLoader
    
    # Filter by split if needed
    if split != "all" and "phase" in df.columns:
        df_split = df[df["phase"] == split].reset_index(drop=True)
    else:
        df_split = df.reset_index(drop=True)
    
    print(f"Creating dataloader for split '{split}': {len(df_split)} samples")
    
    # Create dataset
    dataset = Stage2ReconstructionDataset(
        df=df_split,
        sentiment_labels=sentiment_labels,
        topic_labels=topic_labels
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return dataloader, df_split


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for Stage 2 dataset."""
    return {
        'label_task1': torch.tensor([item['label_task1'] for item in batch], dtype=torch.long),
        'label_task2': torch.tensor([item['label_task2'] for item in batch], dtype=torch.long),
        "ei": torch.stack([item["ei"] for item in batch]),
        "Zi": torch.stack([item["Zi"] for item in batch]),
        "target_text": [item["target_text"] for item in batch],
    }


def encode_text_to_embedding(
    model: Stage2ReconstructionModel,
    texts: List[str],
    device: str,
    max_length: int = 96
) -> torch.Tensor:
    """
    Encode text strings into embedding vectors using the T5 encoder.
    
    Following GLIM's approach:  encode text -> get hidden states -> pool to vector
    Reference: https://github.com/justin-xzliu/GLIM/blob/main/model/glim.py
    
    Args:
        model: Stage2ReconstructionModel with T5 encoder
        texts: List of text strings to encode
        device: Device to use
        max_length: Maximum sequence length for tokenization
        
    Returns: 
        Text embedding vectors (batch_size, embed_dim)
    """
    # Tokenize texts
    encoding = model.tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask
    
    # Get the base model (handle LoRA wrapper if present)
    # Check freeze_strategy to determine model structure
    if model.freeze_strategy == "lora":
        # LoRA wrapped model: model.model is PeftModel
        # Access:  PeftModel -> base_model -> model (T5ForConditionalGeneration)
        base_model = model.model.base_model.model
    else:
        # No LoRA:  model.model is T5ForConditionalGeneration directly
        base_model = model.model

    # Get encoder
    encoder = base_model.get_encoder()
    
    # Encode text to hidden states
    with torch.no_grad():
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
    
    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    
    # Pool hidden states to get embedding vector
    # Following GLIM: use attention-weighted mean pooling
    # Expand attention mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    
    # Sum hidden states weighted by attention mask
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    
    # Normalize by the number of non-padded tokens
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    text_embeddings = sum_embeddings / sum_mask  # (batch, hidden_dim)
    
    return text_embeddings


def generate_predictions(
    model: Stage2ReconstructionModel,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_length: int = 50,
    num_beams: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    verbose: bool = False,
    num_samples_to_print: int = 5
) -> Tuple[List[Dict], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate text predictions using the model and collect embeddings.
    
    Args:
        model: Trained Stage 2 model
        dataloader: DataLoader for inference
        device: Device to use
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        do_sample: Use sampling instead of greedy/beam search
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        verbose: Print sample predictions
        num_samples_to_print: Number of samples to print
        
    Returns: 
        Tuple of: 
            - List of prediction dictionaries
            - EEG embeddings tensor (n, embed_dim)
            - Generated text embeddings tensor (n, embed_dim)
            - Target text embeddings tensor (n, embed_dim)
    """
    all_predictions = []
    all_eeg_embeddings = []
    all_gen_text_embeddings = []
    all_target_text_embeddings = []
    samples_printed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
            # Move batch to device
            label_task1 = batch["label_task1"].to(device)
            label_task2 = batch["label_task2"].to(device)
            ei = batch["ei"].to(device)
            Zi = batch["Zi"].to(device)
            target_texts = batch["target_text"]
            
            # Collect EEG embeddings (ei is the global EEG embedding vector)
            all_eeg_embeddings.append(ei.cpu())
            
            # Generate predictions
            predictions = generate_with_options(
                model=model,
                label_task1=label_task1,
                label_task2=label_task2,
                ei=ei,
                Zi=Zi,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Encode generated texts to embeddings
            gen_text_emb = encode_text_to_embedding(model, predictions, device)
            all_gen_text_embeddings.append(gen_text_emb.cpu())
            
            # Encode target texts to embeddings
            target_text_emb = encode_text_to_embedding(model, target_texts, device)
            all_target_text_embeddings.append(target_text_emb.cpu())
            
            # Store predictions
            for i, (pred, target) in enumerate(zip(predictions, target_texts)):
                pred_dict = {
                    "prediction": pred,
                    "target": target,
                    "sentiment_label_idx": label_task1[i].item(),
                    "topic_label_idx":  label_task2[i].item(),
                }
                all_predictions.append(pred_dict)
                
                # Print samples if verbose
                if verbose and samples_printed < num_samples_to_print: 
                    print(f"\n--- Sample {samples_printed + 1} ---")
                    print(f"Target: {target}")
                    print(f"Prediction: {pred}")
                    samples_printed += 1
    
    # Concatenate all embeddings
    eeg_embeddings = torch.cat(all_eeg_embeddings, dim=0)
    gen_text_embeddings = torch.cat(all_gen_text_embeddings, dim=0)
    target_text_embeddings = torch.cat(all_target_text_embeddings, dim=0)
    
    return all_predictions, eeg_embeddings, gen_text_embeddings, target_text_embeddings


def generate_with_options(
    model: Stage2ReconstructionModel,
    label_task1: torch.Tensor,
    label_task2: torch.Tensor,
    ei: torch.Tensor,
    Zi: torch.Tensor,
    max_length: int = 50,
    num_beams: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0
) -> List[str]:
    """
    Generate text with configurable generation options.
    
    This extends the model's generate method with more options. 

    -> generated replica of stage 2 model -> generate()
    More options provided

    """
    batch_size = label_task1.shape[0]
    device = label_task1.device
    
    # Build prompt
    prompt = model.build_prompt()
    prompts = [prompt] * batch_size
    
    # Tokenize prompt
    prompt_encoding = model.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    input_ids = prompt_encoding.input_ids
    
    # Get initial embeddings
    embeds = model.model.shared(input_ids)
    
    # Cast inputs to model dtype
    sent_embs = model.label_embed_task1(label_task1).to(embeds.dtype)
    topic_embs = model.label_embed_task2(label_task2).to(embeds.dtype)
    ei = ei.to(embeds.dtype)
    Zi = Zi.to(embeds.dtype)
    
    # Find token positions
    sent_mask = (input_ids == model.sent_val_id)
    topic_mask = (input_ids == model.topic_val_id)
    global_mask = (input_ids == model.eeg_global_id)
    seq_mask = (input_ids == model.eeg_seq_id)
    
    # Direct assignment
    batch_indices = torch.arange(batch_size, device=device)
    embeds[batch_indices, sent_mask.long().argmax(dim=1)] = sent_embs
    embeds[batch_indices, topic_mask.long().argmax(dim=1)] = topic_embs
    embeds[batch_indices, global_mask.long().argmax(dim=1)] = ei
    
    Zi_flat = Zi.view(-1, 1024)
    embeds[seq_mask] = Zi_flat
    
    # Generate with options
    gen_kwargs = {
        "inputs_embeds": embeds,
        "max_length": max_length,
        "num_beams": num_beams,
        "do_sample": do_sample,
    }
    
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = top_k
        gen_kwargs["top_p"] = top_p
    
    # !!! We need this to generate
    outputs = model.model.generate(**gen_kwargs)
    
    # Decode
    generated_texts = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts


def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute evaluation metrics for the predictions.
    
    Following the GLIM approach for computing BLEU, ROUGE, and WER metrics.
    Reference: https://github.com/justin-xzliu/GLIM/blob/main/model/glim.py
    
    Args:
        predictions: List of prediction dictionaries
            | -> "prediction"
            | -> "target"
            | -> "sentiment_label_idx"
            | -> "topic_label_idx"
    Returns:
        Dictionary of metrics including:
            - BLEU-1, BLEU-2, BLEU-3, BLEU-4
            - ROUGE-1 (fmeasure, precision, recall)
            - Word Error Rate (WER)
            - Per-sample metrics list
    """
    # Initialize lists for collecting metrics (following GLIM's cal_gen_metrics)
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    rouge1_fmeasure = []
    rouge1_precision = []
    rouge1_recall = []
    wer_scores = []
    
    # Per-sample metrics for detailed analysis
    per_sample_metrics = []
    
    print("\nComputing metrics...")
    for pred_dict in tqdm(predictions, desc="Computing metrics"):
        pred_text = pred_dict["prediction"]
        target_text = pred_dict["target"]
        
        # Handle empty predictions or targets
        if not pred_text or not pred_text.strip():
            pred_text = " "  # Use space to avoid empty string issues
        if not target_text or not target_text.strip():
            target_text = " "
        
        # Compute BLEU scores (n-gram 1-4)
        # Following GLIM:  bleu_score([pred], [targets], n_gram=n)
        # Note: targets can be a tuple/list for multiple references, here we use single reference
        try:
            b1 = bleu_score([pred_text], [[target_text]], n_gram=1)
            b2 = bleu_score([pred_text], [[target_text]], n_gram=2)
            b3 = bleu_score([pred_text], [[target_text]], n_gram=3)
            b4 = bleu_score([pred_text], [[target_text]], n_gram=4)
        except Exception as e:
            # Fallback for edge cases (very short texts, etc.)
            b1 = b2 = b3 = b4 = torch.tensor(0.0)
        
        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)
        
        # Compute ROUGE-1 scores
        # Following GLIM: rouge_score([pred], [targets], rouge_keys='rouge1')
        try:
            rouge1_dict = rouge_score([pred_text], [[target_text]], rouge_keys='rouge1')
            r1_fmeasure = rouge1_dict['rouge1_fmeasure']
            r1_precision = rouge1_dict['rouge1_precision']
            r1_recall = rouge1_dict['rouge1_recall']
        except Exception as e:
            r1_fmeasure = r1_precision = r1_recall = torch.tensor(0.0)
        
        rouge1_fmeasure.append(r1_fmeasure)
        rouge1_precision.append(r1_precision)
        rouge1_recall.append(r1_recall)
        
        # Compute Word Error Rate
        # Following GLIM: word_error_rate([pred], [target])
        try:
            wer = word_error_rate([pred_text], [target_text])
        except Exception as e:
            wer = torch.tensor(1.0)  # Maximum error for edge cases
        
        wer_scores.append(wer)
        
        # Store per-sample metrics
        per_sample_metrics.append({
            "prediction": pred_text,
            "target": target_text,
            "bleu1": b1.item() if torch.is_tensor(b1) else b1,
            "bleu2": b2.item() if torch.is_tensor(b2) else b2,
            "bleu3": b3.item() if torch.is_tensor(b3) else b3,
            "bleu4": b4.item() if torch.is_tensor(b4) else b4,
            "rouge1_fmeasure": r1_fmeasure.item() if torch.is_tensor(r1_fmeasure) else r1_fmeasure,
            "rouge1_precision": r1_precision.item() if torch.is_tensor(r1_precision) else r1_precision,
            "rouge1_recall": r1_recall.item() if torch.is_tensor(r1_recall) else r1_recall,
            "wer": wer.item() if torch.is_tensor(wer) else wer,
            "sentiment_label_idx": pred_dict["sentiment_label_idx"],
            "topic_label_idx":  pred_dict["topic_label_idx"],
        })
    
    # Compute mean metrics (following GLIM's approach of stacking and averaging)
    metrics_mean = {
        "bleu1": torch.stack(bleu1_scores).mean().item(),
        "bleu2": torch.stack(bleu2_scores).mean().item(),
        "bleu3": torch.stack(bleu3_scores).mean().item(),
        "bleu4": torch.stack(bleu4_scores).mean().item(),
        "rouge1_fmeasure": torch.stack(rouge1_fmeasure).mean().item(),
        "rouge1_precision": torch.stack(rouge1_precision).mean().item(),
        "rouge1_recall": torch.stack(rouge1_recall).mean().item(),
        "wer": torch.stack(wer_scores).mean().item(),
    }
    
    # Print summary metrics
    print("\n" + "=" * 60)
    print("Text Generation Metrics Summary")
    print("=" * 60)
    print(f"BLEU-1:            {metrics_mean['bleu1']:.4f}")
    print(f"BLEU-2:            {metrics_mean['bleu2']:.4f}")
    print(f"BLEU-3:            {metrics_mean['bleu3']:.4f}")
    print(f"BLEU-4:            {metrics_mean['bleu4']:.4f}")
    print("-" * 60)
    print(f"ROUGE-1 F-measure:  {metrics_mean['rouge1_fmeasure']:.4f}")
    print(f"ROUGE-1 Precision: {metrics_mean['rouge1_precision']:.4f}")
    print(f"ROUGE-1 Recall:    {metrics_mean['rouge1_recall']:.4f}")
    print("-" * 60)
    print(f"Word Error Rate:   {metrics_mean['wer']:.4f}")
    print("=" * 60)
    
    return {
        "mean":  metrics_mean,
        "per_sample":  per_sample_metrics,
    }


def compute_retrieval_metrics(
    eeg_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    device: str = "cpu"
) -> Dict:
    """
    Compute retrieval metrics using cosine similarity between EEG and text embeddings. 
    
    Following the GLIM approach for computing retrieval accuracy.
    Reference: https://github.com/justin-xzliu/GLIM/blob/main/model/glim.py
    
    Args:
        eeg_embeddings: EEG embedding vectors (n, embed_dim)
        text_embeddings: Text embedding vectors (n, embed_dim)
        device: Device to compute on
        
    Returns:
        Dictionary of retrieval metrics (top-1, top-5, top-10 accuracy)
    """
    eeg_embeddings = eeg_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)
    
    bsz = eeg_embeddings.shape[0]
    
    # Normalize embeddings (following GLIM's align_emb_vector)
    eeg_norm = eeg_embeddings / eeg_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    
    # Compute cosine similarity logits
    logits = eeg_norm @ text_norm.T  # (n, n)
    
    # Targets are identity (each EEG should match its corresponding text)
    targets = torch.arange(bsz, dtype=torch.int, device=device)
    
    # Compute softmax probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Compute top-k accuracy (following GLIM's cal_retrieval_metrics)
    acc_top1 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=1)
    acc_top5 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=min(5, bsz))
    acc_top10 = multiclass_accuracy(probs, targets, average='micro', num_classes=bsz, top_k=min(10, bsz))
    
    retrieval_metrics = {
        "retrieval_acc_top01": acc_top1.item(),
        "retrieval_acc_top05": acc_top5.item(),
        "retrieval_acc_top10": acc_top10.item(),
    }
    
    return retrieval_metrics


def save_predictions(
    predictions: List[Dict],
    df_split: pd.DataFrame,
    output_path: str,
    metrics: Dict
):
    """
    Save predictions to a CSV file.
    
    Args:
        predictions: List of prediction dictionaries
        df_split: Original DataFrame for additional columns
        output_path: Path to save the CSV
        metrics: Computed metrics
    """
    # Create output DataFrame from per-sample metrics if available
    if metrics is not None and "per_sample" in metrics: 
        pred_df = pd.DataFrame(metrics["per_sample"])
    else:
        pred_df = pd.DataFrame(predictions)
    
    # Add additional columns from original DataFrame if available
    # Provides extensibility
    if "text uid" in df_split.columns:
        pred_df["text_uid"] = df_split["text uid"].values[: len(predictions)]
    if "phase" in df_split.columns:
        pred_df["phase"] = df_split["phase"].values[: len(predictions)]
    
    # Save predictions
    pred_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Save metrics summary to a separate JSON file
    if metrics is not None and "mean" in metrics:
        import json
        metrics_path = output_path.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics["mean"], f, indent=2)
        print(f"Metrics summary saved to: {metrics_path}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set device
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("Warning:  CUDA not available, falling back to CPU")
        device = "cpu"
    
    print("=" * 60)
    print("Stage 2 Inference Configuration")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Num beams: {args.num_beams}")
    print(f"Do sample: {args.do_sample}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_pickle(args.data_path)
    print(f"Loaded DataFrame with {len(df)} samples")
    
    # Create dataloader
    dataloader, df_split = create_dataloader(
        df=df,
        split=args.split,
        batch_size=args.batch_size,
        sentiment_labels=args.sentiment_labels,
        topic_labels=args.topic_labels
    )
    
    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        text_model=args.text_model,
        freeze_strategy=args.freeze_strategy,
        lora_rank=args.lora_rank,
        device=device
    )
    
    # Generate predictions and collect embeddings
    print("\nGenerating predictions and collecting embeddings...")
    predictions, eeg_embeddings, gen_text_embeddings, target_text_embeddings = generate_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        max_length=args.max_length,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        verbose=args.verbose,
        num_samples_to_print=args.num_samples_to_print
    )
    
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"EEG embeddings shape: {eeg_embeddings.shape}")
    print(f"Generated text embeddings shape: {gen_text_embeddings.shape}")
    print(f"Target text embeddings shape: {target_text_embeddings.shape}")
    
    # Compute text generation metrics (BLEU, ROUGE, WER)
    metrics = compute_metrics(predictions)
    
    # Compute retrieval metrics
    # EEG -> Generated Text retrieval
    print("\n" + "=" * 60)
    print("Retrieval Metrics:  EEG -> Generated Text")
    print("=" * 60)
    retrieval_metrics_gen = compute_retrieval_metrics(
        eeg_embeddings=eeg_embeddings,
        text_embeddings=gen_text_embeddings,
        device=device
    )
    print(f"Top-1 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top01']:.4f}")
    print(f"Top-5 Accuracy:   {retrieval_metrics_gen['retrieval_acc_top05']:.4f}")
    print(f"Top-10 Accuracy:  {retrieval_metrics_gen['retrieval_acc_top10']:.4f}")
    
    # EEG -> Target Text retrieval
    print("\n" + "=" * 60)
    print("Retrieval Metrics: EEG -> Target Text")
    print("=" * 60)
    retrieval_metrics_target = compute_retrieval_metrics(
        eeg_embeddings=eeg_embeddings,
        text_embeddings=target_text_embeddings,
        device=device
    )
    print(f"Top-1 Accuracy:   {retrieval_metrics_target['retrieval_acc_top01']:.4f}")
    print(f"Top-5 Accuracy:   {retrieval_metrics_target['retrieval_acc_top05']:.4f}")
    print(f"Top-10 Accuracy:  {retrieval_metrics_target['retrieval_acc_top10']:.4f}")
    
    # Add retrieval metrics to the metrics dictionary
    metrics["mean"]["retrieval_eeg_gen_top01"] = retrieval_metrics_gen["retrieval_acc_top01"]
    metrics["mean"]["retrieval_eeg_gen_top05"] = retrieval_metrics_gen["retrieval_acc_top05"]
    metrics["mean"]["retrieval_eeg_gen_top10"] = retrieval_metrics_gen["retrieval_acc_top10"]
    metrics["mean"]["retrieval_eeg_target_top01"] = retrieval_metrics_target["retrieval_acc_top01"]
    metrics["mean"]["retrieval_eeg_target_top05"] = retrieval_metrics_target["retrieval_acc_top05"]
    metrics["mean"]["retrieval_eeg_target_top10"] = retrieval_metrics_target["retrieval_acc_top10"]
    
    # Save predictions
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./outputs/predictions_stage2_{timestamp}.csv"
    else:
        output_path = args.output_path
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    save_predictions(predictions, df_split, output_path, metrics)
    
    print("\nInference complete!")


if __name__ == "__main__": 
    main()