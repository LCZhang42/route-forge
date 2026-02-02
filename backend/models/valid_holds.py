"""
Valid hold positions for MoonBoard 2016.

This module contains the set of valid hold coordinates that actually exist
on the MoonBoard 2016. Generated paths should only use these coordinates.
"""

import pandas as pd
import ast
from pathlib import Path
from typing import Set, Tuple, List


def load_valid_holds_from_dataset(csv_path: str) -> Set[Tuple[int, int]]:
    """
    Extract all unique hold positions from the training dataset.
    
    Args:
        csv_path: Path to CSV file with climbing paths
        
    Returns:
        Set of (x, y) tuples representing valid hold positions
    """
    df = pd.read_csv(csv_path)
    valid_holds = set()
    
    for path_str in df['full_path']:
        path = ast.literal_eval(path_str)
        for hold in path:
            valid_holds.add(tuple(hold))
    
    return valid_holds


def get_valid_holds() -> Set[Tuple[int, int]]:
    """
    Get the set of valid hold positions for MoonBoard 2016.
    
    Returns:
        Set of (x, y) tuples representing valid hold positions
    """
    # Try to load from training data
    data_dir = Path(__file__).parent.parent.parent / 'data'
    train_csv = data_dir / 'moonboard_train_quality.csv'
    
    if train_csv.exists():
        return load_valid_holds_from_dataset(str(train_csv))
    
    # Fallback: return empty set (should not happen in production)
    return set()


def filter_valid_holds(path: List[Tuple[int, int]], valid_holds: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Filter a path to only include valid holds.
    
    Args:
        path: List of (x, y) hold coordinates
        valid_holds: Set of valid hold positions
        
    Returns:
        Filtered list containing only valid holds
    """
    return [hold for hold in path if tuple(hold) in valid_holds]


def is_valid_hold(x: int, y: int, valid_holds: Set[Tuple[int, int]]) -> bool:
    """
    Check if a hold position is valid.
    
    Args:
        x: Column index (0-10)
        y: Row index (1-18)
        valid_holds: Set of valid hold positions
        
    Returns:
        True if the hold exists on the MoonBoard
    """
    return (x, y) in valid_holds


# Cache the valid holds set
_VALID_HOLDS_CACHE = None

def get_cached_valid_holds() -> Set[Tuple[int, int]]:
    """Get cached valid holds (loads once, reuses thereafter)."""
    global _VALID_HOLDS_CACHE
    if _VALID_HOLDS_CACHE is None:
        _VALID_HOLDS_CACHE = get_valid_holds()
    return _VALID_HOLDS_CACHE


if __name__ == '__main__':
    # Test the valid holds extraction
    valid_holds = get_valid_holds()
    print(f"Total valid holds on MoonBoard 2016: {len(valid_holds)}")
    print(f"Grid positions if all filled: 11 × 18 = {11 * 18}")
    print(f"Missing positions: {11 * 18 - len(valid_holds)}")
    print(f"\nSample valid holds (first 20):")
    for i, hold in enumerate(sorted(valid_holds)[:20]):
        col = chr(ord('A') + hold[0])
        print(f"  {col}{hold[1]} -> {hold}")
