"""
Dataset class for climb path training.

Loads preprocessed climb data and converts to token sequences for training.
"""

import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset
from typing import Optional, List, Tuple


class ClimbPathDataset(Dataset):
    """Dataset for training climb path generation model."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_seq_len: int = 128,
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with climb data
            tokenizer: ClimbPathTokenizer instance
            max_seq_len: Maximum sequence length (for padding)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Parse hold coordinates if stored as strings
        if isinstance(self.df['full_path'].iloc[0], str):
            self.df['full_path'] = self.df['full_path'].apply(ast.literal_eval)
        
        print(f"Loaded {len(self.df)} climb paths from {csv_path}")
        print(f"Grade distribution:")
        print(self.df['grade'].value_counts().sort_index())
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Returns:
            (input_ids, attention_mask) tuple
        """
        row = self.df.iloc[idx]
        
        grade = row['grade']
        holds = row['full_path']
        
        # Encode to tokens
        tokens = self.tokenizer.encode(grade, holds)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        seq_len = len(tokens)
        attention_mask = np.ones(seq_len, dtype=np.int32)
        
        # Pad if necessary
        if seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - seq_len
            tokens = np.pad(tokens, (0, padding_len), constant_values=self.tokenizer.PAD_TOKEN)
            attention_mask = np.pad(attention_mask, (0, padding_len), constant_values=0)
        elif seq_len > self.max_seq_len:
            # Truncate if too long (keep BOS, grade, and as many holds as fit, then EOS)
            tokens = np.concatenate([
                tokens[:self.max_seq_len - 1],
                [self.tokenizer.EOS_TOKEN]
            ])
            attention_mask = np.ones(self.max_seq_len, dtype=np.int32)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (input_ids, attention_mask) tuples
        
    Returns:
        Dictionary with batched tensors
    """
    input_ids, attention_masks = zip(*batch)
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
    }


class ClimbPathDataModule:
    """
    Data module for managing train/val/test datasets.
    
    Provides convenient interface for loading all splits.
    """
    
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        tokenizer,
        batch_size: int = 32,
        max_seq_len: int = 128,
        num_workers: int = 4,
    ):
        """
        Initialize data module.
        
        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            test_csv: Path to test CSV
            tokenizer: ClimbPathTokenizer instance
            batch_size: Batch size for DataLoader
            max_seq_len: Maximum sequence length
            num_workers: Number of DataLoader workers
        """
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Load all datasets."""
        print("Setting up datasets...")
        
        self.train_dataset = ClimbPathDataset(
            self.train_csv,
            self.tokenizer,
            self.max_seq_len,
        )
        
        self.val_dataset = ClimbPathDataset(
            self.val_csv,
            self.tokenizer,
            self.max_seq_len,
        )
        
        self.test_dataset = ClimbPathDataset(
            self.test_csv,
            self.tokenizer,
            self.max_seq_len,
        )
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Val:   {len(self.val_dataset)}")
        print(f"  Test:  {len(self.test_dataset)}")
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training DataLoader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation DataLoader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get test DataLoader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
