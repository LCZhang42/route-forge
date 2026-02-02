"""
Check if generated paths are memorized from training data.

Compares generated paths against train/val/test sets to detect exact copies.
"""

import torch
import pandas as pd
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Set

sys.path.append(str(Path(__file__).parent.parent))

from models.tokenizer import ClimbPathTokenizer
from models.climb_transformer import ClimbPathTransformerWithGeneration
from models.logits_processor import (
    ClimbPathLogitsProcessor,
    MinHoldsLogitsProcessor,
    MaxHoldsLogitsProcessor,
)


def load_dataset_paths(csv_path: str) -> Set[Tuple[str, Tuple[Tuple[int, int], ...]]]:
    """
    Load all paths from a dataset.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Set of (grade, holds_tuple) tuples for fast lookup
    """
    df = pd.read_csv(csv_path)
    
    paths = set()
    for _, row in df.iterrows():
        grade = row['grade']
        holds = eval(row['full_path']) if isinstance(row['full_path'], str) else row['full_path']
        # Convert to tuple for hashing
        holds_tuple = tuple(tuple(h) for h in holds)
        paths.add((grade, holds_tuple))
    
    return paths


def generate_paths(
    model,
    tokenizer,
    grade: str,
    num_samples: int,
    device: str = 'cpu',
    temperature: float = 1.0,
) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Generate multiple paths for a grade."""
    logits_processors = [
        ClimbPathLogitsProcessor(tokenizer),
        MinHoldsLogitsProcessor(tokenizer, min_holds=3),
        MaxHoldsLogitsProcessor(tokenizer, max_holds=30),
    ]
    
    generated_paths = []
    
    for _ in range(num_samples):
        grade_token = tokenizer.encode_grade(grade)
        
        with torch.no_grad():
            tokens = model.generate_with_processors(
                grade_token=grade_token,
                tokenizer=tokenizer,
                logits_processors=logits_processors,
                max_length=128,
                temperature=temperature,
                device=device,
            )
        
        decoded_grade, holds = tokenizer.decode(tokens.cpu().numpy())
        generated_paths.append((decoded_grade, holds))
    
    return generated_paths


def check_memorization(
    generated_paths: List[Tuple[str, List[Tuple[int, int]]]],
    train_paths: Set[Tuple[str, Tuple[Tuple[int, int], ...]]],
    val_paths: Set[Tuple[str, Tuple[Tuple[int, int], ...]]],
    test_paths: Set[Tuple[str, Tuple[Tuple[int, int], ...]]],
) -> dict:
    """
    Check if generated paths exist in training/val/test sets.
    
    Returns:
        Dictionary with memorization statistics
    """
    stats = {
        'total_generated': len(generated_paths),
        'exact_train_matches': 0,
        'exact_val_matches': 0,
        'exact_test_matches': 0,
        'unique_paths': 0,
        'duplicates_within_generated': 0,
    }
    
    seen_generated = set()
    exact_matches = {'train': [], 'val': [], 'test': []}
    
    for i, (grade, holds) in enumerate(generated_paths):
        holds_tuple = tuple(tuple(h) for h in holds)
        path_key = (grade, holds_tuple)
        
        # Check if duplicate within generated set
        if path_key in seen_generated:
            stats['duplicates_within_generated'] += 1
        else:
            seen_generated.add(path_key)
            stats['unique_paths'] += 1
        
        # Check against datasets
        if path_key in train_paths:
            stats['exact_train_matches'] += 1
            exact_matches['train'].append(i)
        
        if path_key in val_paths:
            stats['exact_val_matches'] += 1
            exact_matches['val'].append(i)
        
        if path_key in test_paths:
            stats['exact_test_matches'] += 1
            exact_matches['test'].append(i)
    
    stats['exact_matches_indices'] = exact_matches
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Check for memorization in generated paths')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='checkpoints/climb_path_cpu/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='checkpoints/climb_path_cpu/config.json',
                        help='Path to training config')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='data/moonboard_train_quality.csv',
                        help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default='data/moonboard_val_quality.csv',
                        help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default='data/moonboard_test_quality.csv',
                        help='Path to test CSV')
    
    # Generation arguments
    parser.add_argument('--grade', type=str, default='7A',
                        help='Grade to generate')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of paths to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MEMORIZATION CHECK FOR CLIMB PATH MODEL")
    print("=" * 70)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_paths = load_dataset_paths(args.train_csv)
    val_paths = load_dataset_paths(args.val_csv)
    test_paths = load_dataset_paths(args.test_csv)
    
    print(f"   Train set: {len(train_paths)} unique paths")
    print(f"   Val set:   {len(val_paths)} unique paths")
    print(f"   Test set:  {len(test_paths)} unique paths")
    
    # Load model
    print("\n2. Loading model...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    tokenizer = ClimbPathTokenizer()
    model = ClimbPathTransformerWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 128),
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Generate paths
    print(f"\n3. Generating {args.num_samples} paths for grade {args.grade}...")
    generated_paths = generate_paths(
        model=model,
        tokenizer=tokenizer,
        grade=args.grade,
        num_samples=args.num_samples,
        device=args.device,
        temperature=args.temperature,
    )
    
    # Check memorization
    print("\n4. Checking for memorization...")
    stats = check_memorization(generated_paths, train_paths, val_paths, test_paths)
    
    # Print results
    print("\n" + "=" * 70)
    print("MEMORIZATION ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nGeneration Statistics:")
    print(f"  Total generated:              {stats['total_generated']}")
    print(f"  Unique paths:                 {stats['unique_paths']}")
    print(f"  Duplicates within generated:  {stats['duplicates_within_generated']}")
    
    print(f"\nExact Matches with Datasets:")
    print(f"  Training set matches:   {stats['exact_train_matches']} ({stats['exact_train_matches']/stats['total_generated']*100:.2f}%)")
    print(f"  Validation set matches: {stats['exact_val_matches']} ({stats['exact_val_matches']/stats['total_generated']*100:.2f}%)")
    print(f"  Test set matches:       {stats['exact_test_matches']} ({stats['exact_test_matches']/stats['total_generated']*100:.2f}%)")
    
    # Interpretation
    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    train_match_rate = stats['exact_train_matches'] / stats['total_generated'] * 100
    
    if train_match_rate == 0:
        print("✓ EXCELLENT: No exact copies from training data!")
        print("  The model is generating novel paths.")
    elif train_match_rate < 5:
        print("✓ GOOD: Very low memorization rate (<5%)")
        print("  Model is mostly generating novel paths with rare exact matches.")
    elif train_match_rate < 20:
        print("⚠ MODERATE: Some memorization detected (5-20%)")
        print("  Model may be overfitting. Consider:")
        print("  - Increasing dropout")
        print("  - Reducing model size")
        print("  - Adding more training data diversity")
    else:
        print("✗ HIGH MEMORIZATION: Model is copying training data (>20%)")
        print("  This indicates severe overfitting. Actions needed:")
        print("  - Reduce model capacity")
        print("  - Increase regularization")
        print("  - Add data augmentation")
    
    if stats['duplicates_within_generated'] > stats['total_generated'] * 0.1:
        print(f"\n⚠ WARNING: High duplicate rate in generated samples ({stats['duplicates_within_generated']/stats['total_generated']*100:.1f}%)")
        print("  Consider increasing temperature for more diversity.")
    
    # Save results
    output_path = Path(args.checkpoint).parent / 'memorization_check.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
