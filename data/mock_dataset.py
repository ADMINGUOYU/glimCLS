import torch
import random
from torch.utils.data import Dataset
from typing import Dict


class MockReconstructionDataset(Dataset):
    """
    Mock dataset for Stage 2 text reconstruction training.
    Generates random EEG features, labels, and target text.
    """

    def __init__(self, size: int = 1000, seed: int = 42):
        """
        Args:
            size: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.size = size
        random.seed(seed)
        torch.manual_seed(seed)

        # Generate vocabulary
        self.vocab = [f"word_{i:04d}" for i in range(1000)]

        # Pre-generate all samples
        self.samples = []
        for i in range(size):
            self.samples.append(self._generate_sample(i))

    def _generate_sample(self, idx: int) -> Dict:
        """Generate a single random sample."""
        # Random labels (binary)
        label_task1 = random.randint(0, 1)
        label_task2 = random.randint(0, 1)

        # Random EEG features
        ei = torch.randn(1024)
        ei = ei / ei.norm()  # Normalize to unit norm

        Zi = torch.randn(96, 1024)
        Zi = Zi / Zi.norm(dim=-1, keepdim=True)  # Normalize along last dimension

        # Random target text (10-20 words)
        num_words = random.randint(10, 20)
        words = random.sample(self.vocab, num_words)
        target_text = " ".join(words)

        return {
            'label_task1': label_task1,
            'label_task2': label_task2,
            'ei': ei,
            'Zi': Zi,
            'target_text': target_text
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def create_dataloaders(data_size: int = 1000, batch_size: int = 8, seed: int = 42):
    """
    Create train/val/test dataloaders with 80/10/10 split.

    Args:
        data_size: Total number of samples
        batch_size: Batch size for dataloaders
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, Subset

    # Create full dataset
    dataset = MockReconstructionDataset(size=data_size, seed=seed)

    # Split indices
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    test_size = data_size - train_size - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, data_size))

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
