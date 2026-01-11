import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional


# Label mappings for classification tasks
# TODO: integrated to training script? HERE HARD CODED
SENTIMENT_LABELS = ['non_neutral', 'neutral']
TOPIC_LABELS = ['Biographies and Factual Knowledge', 'Movie Reviews and Sentiment']


class Stage2ReconstructionDataset(Dataset):
    """
    Dataset for Stage 2 text reconstruction training.
    Loads EEG features and labels from a pandas DataFrame 
    (output from predict_glim_parallel_and_pack.py).
    -> to be checked
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sentiment_labels: Optional[List[str]] = None,
        topic_labels: Optional[List[str]] = None
    ):
        """
        Args: 
            df: DataFrame containing the data with columns:
                - 'ei': (1024,) global EEG embedding
                - 'Zi': (96, 1024) sequence EEG embeddings ??? Not sure, 96?
                - 'sentiment label' or 'pred_sentiment_label':  sentiment label string
                - 'topic_label' or 'pred_topic_label': topic label string
                - 'input text': target text for reconstruction
                - 'pred_length' or 'length': length of the sentence
                - 'pred_surprisal' or 'surprisal': surprisal of the sentence
            sentiment_labels: List of sentiment label names for mapping to indices
            topic_labels: List of topic label names for mapping to indices
        """
        self.df = df.reset_index(drop=True)
        
        # Set up label mappings
        self.sentiment_labels = sentiment_labels or SENTIMENT_LABELS
        self.topic_labels = topic_labels or TOPIC_LABELS
        
        self.sentiment_to_idx = {label: idx for idx, label in enumerate(self.sentiment_labels)}
        self.topic_to_idx = {label: idx for idx, label in enumerate(self.topic_labels)}
        
        # Determine which columns to use for labels
        # Prefer predicted labels if available (from Stage 1), otherwise use ground truth
        self.sentiment_col = 'pred_sentiment_label' if 'pred_sentiment_label' in df.columns else 'sentiment label'
        self.topic_col = 'pred_topic_label' if 'pred_topic_label' in df.columns else 'topic_label'
        self.length_col = 'pred_length' if 'pred_length' in df.columns else 'length'
        self.surprisal_col = 'pred_surprisal' if 'pred_surprisal' in df.columns else 'surprisal'
        
        # Extract data
        self.ei = df['ei'].tolist()
        self.Zi = df['Zi'].tolist()
        self.sentiment_labels_raw = df[self.sentiment_col].tolist()
        self.topic_labels_raw = df[self.topic_col].tolist()
        self.target_texts = df['input text'].tolist()
        self.length = df[self.length_col].tolist()
        self.surprisal = df[self.surprisal_col].tolist()
        
        print(f"Stage2Dataset initialized with {len(self.df)} samples")
        print(f"  Using sentiment column: {self.sentiment_col}")
        print(f"  Using topic column: {self.topic_col}")
        print(f"  Using length column:  {self.length_col}")
        print(f"  Using surprisal column:  {self.surprisal_col}")
        print(f"  Sentiment labels: {self.sentiment_labels}")
        print(f"  Topic labels:  {self.topic_labels}")

    def __len__(self) -> int:
        return len(self.df)

    def _convert_sentiment_to_idx(self, label: str) -> int:
        """Convert sentiment label string to index."""
        if label in self.sentiment_to_idx:
            return self.sentiment_to_idx[label]
        # Handle potential label variations
        label_lower = label.lower()
        for key, idx in self.sentiment_to_idx.items():
            if key.lower() == label_lower:
                return idx
        # Default to first class if unknown
        print(f"Warning:  Unknown sentiment label '{label}', defaulting to 0")
        return 0

    def _convert_topic_to_idx(self, label: str) -> int:
        """Convert topic label string to index."""
        if label in self.topic_to_idx:
            return self.topic_to_idx[label]
        # Handle potential label variations
        label_lower = label.lower()
        for key, idx in self.topic_to_idx.items():
            if key.lower() == label_lower: 
                return idx
        # Default to first class if unknown
        print(f"Warning: Unknown topic label '{label}', defaulting to 0")
        return 0

    def __getitem__(self, idx:  int) -> Dict: 
        # Get EEG features
        ei = self.ei[idx]
        Zi = self.Zi[idx]
        
        # Convert to tensors if numpy arrays
        if hasattr(ei, 'shape'):
            ei = torch.from_numpy(ei) if not isinstance(ei, torch.Tensor) else ei
        else:
            ei = torch.tensor(ei)
            
        if hasattr(Zi, 'shape'):
            Zi = torch.from_numpy(Zi) if not isinstance(Zi, torch.Tensor) else Zi
        else: 
            Zi = torch.tensor(Zi)
        
        # Ensure correct dtype
        ei = ei.float()
        Zi = Zi.float()
        
        # Get labels as indices
        label_task1 = self._convert_sentiment_to_idx(self.sentiment_labels_raw[idx])
        label_task2 = self._convert_topic_to_idx(self.topic_labels_raw[idx])
        
        # Get target text (using 'input text' for now)
        target_text = self.target_texts[idx]

        # Get length and surprisal
        length = self.length[idx]
        surprisal = self.surprisal[idx]
        
        return {
            'label_task1': label_task1,
            'label_task2': label_task2,
            'length': length,
            'surprisal': surprisal,
            'ei': ei,
            'Zi': Zi,
            'target_text': target_text
        }


def stage2_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for Stage2ReconstructionDataset.
    
    Args:
        batch: List of sample dictionaries
        
    Returns: 
        Batched dictionary with tensors
    """
    return {
        'label_task1': torch.tensor([item['label_task1'] for item in batch], dtype=torch.long),
        'label_task2': torch.tensor([item['label_task2'] for item in batch], dtype=torch.long),
        'length': torch.tensor([item['length'] for item in batch], dtype=torch.float),
        'surprisal': torch.tensor([item['surprisal'] for item in batch], dtype=torch.float),
        'ei': torch.stack([item['ei'] for item in batch]),
        'Zi': torch.stack([item['Zi'] for item in batch]),
        'target_text': [item['target_text'] for item in batch]
    }


def create_stage2_dataloaders(
    data_path: str,
    batch_size: int = 8,
    sentiment_labels: Optional[List[str]] = None,
    topic_labels: Optional[List[str]] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from a pandas DataFrame.
    
    Args:
        data_path: Path to the pandas DataFrame pickle file
        batch_size: Batch size for dataloaders
        sentiment_labels: List of sentiment label names
        topic_labels: List of topic label names
        num_workers:  Number of workers for data loading
        
    Returns: 
        train_loader, val_loader, test_loader
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_pickle(data_path)
    print(f"Total samples: {len(df)}")
    
    # Check if phase column exists for splitting
    if 'phase' not in df.columns:
        # Create splits based on 80/10/10
        # probably not used
        n = len(df)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        # Shuffle indices
        indices = df.index.tolist()
        import random
        random.shuffle(indices)
        
        train_indices = indices[: train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        df_train = df.loc[train_indices]
        df_val = df.loc[val_indices]
        df_test = df.loc[test_indices]
    else:
        # Use existing phase column
        df_train = df[df['phase'] == 'train']
        df_val = df[df['phase'] == 'val']
        df_test = df[df['phase'] == 'test']
    
    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")
    
    # Create datasets
    train_dataset = Stage2ReconstructionDataset(df_train, sentiment_labels, topic_labels)
    val_dataset = Stage2ReconstructionDataset(df_val, sentiment_labels, topic_labels)
    test_dataset = Stage2ReconstructionDataset(df_test, sentiment_labels, topic_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage2_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=stage2_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=stage2_collate_fn
    )
    
    return train_loader, val_loader, test_loader