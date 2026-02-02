"""
Weighted dataset for quality-based training.

Allows sampling based on route quality (repeats) to give more importance
to popular, well-tested routes.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler
from .dataset import ClimbPathDataset


class WeightedClimbPathDataset(ClimbPathDataset):
    """Dataset with quality-based sampling weights."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_seq_len: int = 128,
        use_quality_weights: bool = True,
    ):
        """
        Initialize weighted dataset.
        
        Args:
            csv_path: Path to CSV file with climb data
            tokenizer: ClimbPathTokenizer instance
            max_seq_len: Maximum sequence length
            use_quality_weights: Whether to compute quality weights
        """
        super().__init__(csv_path, tokenizer, max_seq_len)
        
        self.use_quality_weights = use_quality_weights
        
        if use_quality_weights:
            # Compute quality weights from repeats
            # Use log scale to avoid extreme weights
            repeats = self.df['repeats'].values
            self.weights = np.log1p(repeats)
            self.weights = self.weights / self.weights.max()
            
            print(f"Quality weights computed:")
            print(f"  Min weight: {self.weights.min():.3f}")
            print(f"  Max weight: {self.weights.max():.3f}")
            print(f"  Mean weight: {self.weights.mean():.3f}")
        else:
            self.weights = np.ones(len(self.df))
    
    def get_sampler(self) -> WeightedRandomSampler:
        """
        Get a weighted random sampler for DataLoader.
        
        Returns:
            WeightedRandomSampler instance
        """
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.weights),
            replacement=True,
        )


def create_weighted_dataloader(
    csv_path: str,
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 128,
    num_workers: int = 4,
    use_quality_weights: bool = True,
):
    """
    Create a DataLoader with quality-based sampling.
    
    Args:
        csv_path: Path to CSV file
        tokenizer: ClimbPathTokenizer instance
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of DataLoader workers
        use_quality_weights: Whether to use quality-based sampling
        
    Returns:
        DataLoader instance
    """
    from .dataset import collate_fn
    
    dataset = WeightedClimbPathDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        use_quality_weights=use_quality_weights,
    )
    
    sampler = dataset.get_sampler() if use_quality_weights else None
    shuffle = not use_quality_weights  # Don't shuffle if using sampler
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader
