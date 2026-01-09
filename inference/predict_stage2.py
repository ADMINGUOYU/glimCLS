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
        "label_task1": torch.stack([item["label_task1"] for item in batch]),
        "label_task2": torch.stack([item["label_task2"] for item in batch]),
        "ei": torch.stack([item["ei"] for item in batch]),
        "Zi": torch.stack([item["Zi"] for item in batch]),
        "target_text": [item["target_text"] for item in batch],
    }


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
) -> List[Dict]:
    """
    Generate text predictions using the model.
    
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
        List of prediction dictionaries
    """
    all_predictions = []
    samples_printed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
            # Move batch to device
            label_task1 = batch["label_task1"].to(device)
            label_task2 = batch["label_task2"].to(device)
            ei = batch["ei"].to(device)
            Zi = batch["Zi"].to(device)
            target_texts = batch["target_text"]
            
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
    
    return all_predictions


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

    **TODO**
    
    Args:
        predictions: List of prediction dictionaries
            | -> "prediction"
            | -> "target"
            | -> "sentiment_label_idx"
            | -> "topic_label_idx"
    Returns:
        Dictionary of metrics
    """
    
    # TODO
    
    return { }


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
        metrics: Computed metrics **TODO
    """
    # Create output DataFrame
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
    
    # Save metrics
    # TODO


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
    print(f"Do sample:  {args.do_sample}")
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
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions(
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
    
    # Compute metrics
    # TODO
    metrics = None
    
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