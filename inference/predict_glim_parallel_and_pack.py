#!/usr/bin/env python
"""
Multi-Checkpoint Prediction Script for GLIM_PARALLEL

This script loads 4 separate checkpoints (best models for sentiment, topic, length, surprisal)
and generates predictions on the entire dataset. All predictions are saved to a pandas DataFrame.

TODO: to compact with a single ckpt

"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os. path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.glim_parallel import GLIM_PARALLEL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions using 4 specialized checkpoints for GLIM_PARALLEL"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the merged dataframe (e.g., zuco_merged_with_topic.df)"
    )
    
    # Checkpoint arguments
    parser. add_argument(
        "--sentiment_checkpoint",
        type=str,
        required=True,
        help="Path to the best checkpoint for sentiment classification"
    )
    parser.add_argument(
        "--topic_checkpoint",
        type=str,
        required=True,
        help="Path to the best checkpoint for topic classification"
    )
    parser.add_argument(
        "--length_checkpoint",
        type=str,
        required=True,
        help="Path to the best checkpoint for length prediction"
    )
    parser.add_argument(
        "--surprisal_checkpoint",
        type=str,
        required=True,
        help="Path to the best checkpoint for surprisal prediction"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the predictions DataFrame.  Defaults to predictions_<timestamp>.df"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index to use (default: 0)"
    )
    
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for prediction (default: 24)"
    )
    
    # Dataset split
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which data split to predict on (default: all)"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> GLIM_PARALLEL:
    """
    Load a GLIM_PARALLEL model from a checkpoint. 
    !!! model is set to evaluation mode
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded GLIM_PARALLEL model
    """
    print(f"Loading checkpoint:  {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get hyperparameters from checkpoint
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Create model with saved hyperparameters
    model = GLIM_PARALLEL(**hparams)
    
    # Load state dict (strict=False -> we don't have text_model weights -> let's play safe)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Setup model (loads text model)
    model.to(device)
    model.eval()
    
    return model


def prepare_batch_for_prediction(batch: dict, device: torch.device) -> dict:
    """
    Prepare a batch dictionary for prediction.
    Note: batch['prompt'] can be either in [(tuple), (tuple), ... ] or [['task' ... ], ['dataset' ... ], ['subject' ... ]]
    
    Args: 
        batch: Batch from dataloader
          |-> batch['eeg']    in shape (batch_size, seq_len, channels)
          |-> batch['prompt'] None for not using prompt
          |-> batch['mask']   None for not using mask
        device:  Device to move tensors to
        
    Returns:
        Prepared batch dictionary
    """

    # get batch size
    batch_size = batch['eeg'].shape[0]

    prepared = {
        'eeg': batch['eeg'].to(device),
        'mask': batch['mask'].to(device) if batch['mask'] is not None else None
    }
    
    if batch['prompt'] is None:
        # if none, we set to none (predict function in glim_parallel will automatically set this to <UNK>.)
        prepared['prompt'] = None
    elif isinstance(batch['prompt'][0], tuple):
        # If we have prompt as tuples, make it lists
        assert len(batch['prompt']) == batch_size, "[ERROR] batch['prompt'] should contain exactly batch_size tuples"
        # Already in correct format:  list of tuples
        prompts = batch['prompt']
        prepared['prompt'] = [
            [p[0] for p in prompts],  # task prompts
            [p[1] for p in prompts],  # dataset prompts
            [p[2] for p in prompts],  # subject prompts
        ]
    else:
        assert len(batch['prompt']) == 3, "[ERROR] batch['prompt'] should contain exactly 3 lists"
        assert len(batch['prompt'][0]) == batch_size, "[ERROR] batch['prompt'][0] should contain exactly batch_size prompts"
        # else, assign the same thing
        prepared['prompt'] = batch['prompt']
    
    return prepared


def predict_with_model(
    model: GLIM_PARALLEL,
    dataloader,
    device: torch.device,
    task: str
) -> dict:
    """
    Run predictions using a specific model.
    
    Args:
        model: The GLIM_PARALLEL model
        dataloader: DataLoader to iterate over
        device: Target device
        task: Task name ('sentiment', 'topic', 'length', 'surprisal')
        
    Returns:
        Dictionary of predictions (indexed by 'text uid')
    """
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {task}"):

            prepared_batch = prepare_batch_for_prediction(batch, device)
            outputs = model.predict_step(prepared_batch, 0)     # 0 is batch_idx -> not used param
            
            if task == 'sentiment':
                preds = {
                    'pred_idx': outputs['sentiment_pred'].cpu().numpy(),
                    'pred_label': outputs['sentiment_label'],
                    'pred_prob': outputs['sentiment_prob'].cpu().numpy(),
                }
            elif task == 'topic': 
                preds = {
                    'pred_idx': outputs['topic_pred'].cpu().numpy(),
                    'pred_label': outputs['topic_label'],
                    'pred_prob': outputs['topic_prob'].cpu().numpy(),
                }
            elif task == 'length':
                preds = {
                    'pred_value': outputs['length_pred'].cpu().numpy(),
                }
            elif task == 'surprisal':
                preds = {
                    'pred_value': outputs['surprisal_pred'].cpu().numpy(),
                }
            
            # Add text_uid for merging
            preds['text_uid'] = batch['text uid'] if isinstance(batch['text uid'], list) else batch['text uid'].tolist()
            
            # Adds embeddings
            preds['ei'] = outputs['eeg_emb'].cpu().numpy()
            preds['Zi'] = outputs['Zi'].cpu().numpy()

            all_predictions.append(preds)
    
    # Combine all predictions
    combined = {}
    for key in all_predictions[0].keys():
        if isinstance(all_predictions[0][key], np.ndarray):
            combined[key] = np.concatenate([p[key] for p in all_predictions])
        else: 
            combined[key] = []
            for p in all_predictions: 
                combined[key].extend(p[key])
    
    return combined


def create_prediction_dataloader(df: pd.DataFrame, split: str, batch_size: int):
    """
    Create a simple dataloader for predictions (without the complex sampling).
    
    Args:
        df: DataFrame with data
        split: Data split to use ('train', 'val', 'test', or 'all')
        batch_size: Batch size
        
    Returns: 
        DataLoader and indices mapping
        !!! df_split: selected rows based on 'split' argument passed
    """
    from torch.utils.data import DataLoader
    
    if split != 'all':
        df_split = df[df['phase'] == split].reset_index(drop = True)
    else:
        df_split = df.reset_index(drop = True)
    
    # Create dataset
    dataset = PredictionDataset(df_split)
    
    # Simple sequential dataloader
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
        collate_fn = prediction_collate_fn
    )
    
    return dataloader, df_split


class PredictionDataset(torch.utils.data.Dataset):
    """Simple dataset for prediction that includes all necessary fields."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.eeg = df['eeg'].tolist()
        self.mask = df['mask'].tolist()
        
        # Prompts
        raw_t_keys = df['task'].tolist()
        t_prompts = ['<NR>' if t_key != 'task3' else '<TSR>' for t_key in raw_t_keys]    # note that we only have these two types of prompts [following datamodule.py]
        d_prompts = df['dataset'].tolist()
        s_prompts = df['subject'].tolist()
        self.prompts = list(zip(t_prompts, d_prompts, s_prompts))   # now in tuple format
        
        self.text_uids = df['text uid'].tolist()
        
        # Ground truth labels (if available)
        self.sentiment_labels = df['sentiment label'].tolist() if 'sentiment label' in df.columns else [None] * len(df)
        self.topic_labels = df['topic_label'].tolist() if 'topic_label' in df.columns else [None] * len(df)
        self.length_values = df['length'].tolist() if 'length' in df.columns else [None] * len(df)
        self.surprisal_values = df['surprisal'].tolist() if 'surprisal' in df.columns else [None] * len(df)
        
        # Additional metadata
        self.phases = df['phase'].tolist() if 'phase' in df.columns else [None] * len(df)
        self.input_texts = df['input text'].tolist() if 'input text' in df.columns else [None] * len(df)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'eeg': torch.from_numpy(self. eeg[idx]),
            'mask': torch.from_numpy(self.mask[idx]),
            'prompt': self.prompts[idx],
            'text uid': self.text_uids[idx],
            'sentiment label': self.sentiment_labels[idx],
            'topic_label': self.topic_labels[idx],
            'length': self.length_values[idx],
            'surprisal': self.surprisal_values[idx],
            'phase': self.phases[idx],
            'input text': self.input_texts[idx],
        }


def prediction_collate_fn(batch):
    """Collate function for prediction batches."""
    # -> join the items within batch of items
    return {
        'eeg': torch.stack([item['eeg'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'prompt': [item['prompt'] for item in batch],
        'text uid': [item['text uid'] for item in batch],
        'sentiment label': [item['sentiment label'] for item in batch],
        'topic_label': [item['topic_label'] for item in batch],
        'length': [item['length'] for item in batch],
        'surprisal': [item['surprisal'] for item in batch],
        'phase': [item['phase'] for item in batch],
        'input text': [item['input text'] for item in batch],
    }


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate checkpoint paths
    for name, path in [
        ("sentiment", args.sentiment_checkpoint),
        ("topic", args.topic_checkpoint),
        ("length", args.length_checkpoint),
        ("surprisal", args.surprisal_checkpoint),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} checkpoint not found:  {path}")
    
    # Load data
    print(f"Loading data from:  {args.data_path}")
    df = pd.read_pickle(args.data_path)
    print(f"Total samples: {len(df)}")
    
    # Create dataloader
    dataloader, df_split = create_prediction_dataloader(df, args.split, args.batch_size)
    print(f"Samples in '{args.split}' split: {len(df_split)}")
    
    # Initialize results DataFrame with metadata
    # copy Ground Truth / phase / text_uid ... information
    results = pd.DataFrame({
        'text uid': df_split['text uid'].tolist(),
        'phase': df_split['phase'].tolist() if 'phase' in df_split. columns else [None] * len(df_split),
        'input text': df_split['input text'].tolist() if 'input text' in df_split.columns else [None] * len(df_split),
        'sentiment label': df_split['sentiment label'].tolist() if 'sentiment label' in df_split.columns else [None] * len(df_split),
        'topic_label': df_split['topic_label'].tolist() if 'topic_label' in df_split.columns else [None] * len(df_split),
        'length': df_split['length'].tolist() if 'length' in df_split. columns else [None] * len(df_split),
        'surprisal': df_split['surprisal'].tolist() if 'surprisal' in df_split.columns else [None] * len(df_split),
    })
    
    # Load and predict with each checkpoint
    checkpoint_tasks = [
        ('sentiment', args.sentiment_checkpoint),
        ('topic', args.topic_checkpoint),
        ('length', args.length_checkpoint),
        ('surprisal', args.surprisal_checkpoint),
    ]
    
    for task, checkpoint_path in checkpoint_tasks: 
        print(f"\n{'='*60}")
        print(f"Processing {task.upper()} predictions")
        print(f"{'='*60}")
        
        # Load model
        model = load_model_from_checkpoint(checkpoint_path, device)
        
        # Run predictions
        predictions = predict_with_model(model, dataloader, device, task)

        # Store ei and Zi
        if 'ei' not in results.columns:
            results['ei'] = predictions['ei']
        if 'Zi' not in results.columns:
            results['Zi'] = predictions['Zi']

        # Add predictions to results
        if task in ['sentiment', 'topic']: 
            results[f'pred_{task}_label'] = predictions['pred_label']
            results[f'pred_{task}_idx'] = predictions['pred_idx']
            # Store probabilities as list of arrays
            results[f'pred_{task}_prob'] = [prob.tolist() for prob in predictions['pred_prob']]
        else:
            results[f'pred_{task}'] = predictions['pred_value']
        
        # Clear model from memory
        del model
        torch.cuda.empty_cache()
    
    # Set output path
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = f"predictions_{timestamp}.df"
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Saving predictions to: {args.output_path}")
    results.to_pickle(args.output_path)
    
    # Also save as CSV for easy viewing
    csv_path = args.output_path.replace('.df', '.csv')
    # For CSV, convert probability lists to strings
    results_csv = results.copy()
    for col in ['pred_sentiment_prob', 'pred_topic_prob']: 
        if col in results_csv.columns:
            results_csv[col] = results_csv[col].apply(str)
    results_csv.to_csv(csv_path, index=False)
    print(f"Also saved as CSV: {csv_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total predictions: {len(results)}")
    
    # Sentiment accuracy (if ground truth available)
    if 'sentiment label' in results.columns and results['sentiment label'].notna().any():
        correct = (results['sentiment label'] == results['pred_sentiment_label']).sum()
        total = results['sentiment label'].notna().sum()
        print(f"Sentiment Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    
    # Topic accuracy (if ground truth available)
    if 'topic_label' in results.columns and results['topic_label'].notna().any():
        correct = (results['topic_label'] == results['pred_topic_label']).sum()
        total = results['topic_label'].notna().sum()
        print(f"Topic Accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    
    # Length MAE (if ground truth available)
    if 'length' in results.columns and results['length'].notna().any():
        valid_mask = results['length'].notna()
        mae = np.abs(results. loc[valid_mask, 'length']. values - results.loc[valid_mask, 'pred_length'].values).mean()
        print(f"Length MAE:  {mae:.4f}")
    
    # Surprisal MAE (if ground truth available)
    if 'surprisal' in results.columns and results['surprisal'].notna().any():
        valid_mask = results['surprisal'].notna()
        mae = np.abs(results. loc[valid_mask, 'surprisal'].values - results.loc[valid_mask, 'pred_surprisal'].values).mean()
        print(f"Surprisal MAE: {mae:.4f}")
    
    print(f"\nDone! Results saved to {args.output_path}")


if __name__ == "__main__":
    main()