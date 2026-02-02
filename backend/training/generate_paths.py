"""
Inference script for generating climb paths using trained model.

Generates climb paths for specified grades with constraint enforcement.
"""

import torch
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from models.tokenizer import ClimbPathTokenizer
from models.climb_transformer import ClimbPathTransformerWithGeneration
from models.logits_processor import (
    ClimbPathLogitsProcessor,
    MinHoldsLogitsProcessor,
    MaxHoldsLogitsProcessor,
    ValidHoldsLogitsProcessor,
    NoRepeatHoldsLogitsProcessor,
)
from models.reachability_processor import (
    ReachabilityLogitsProcessor,
    AdaptiveReachabilityProcessor,
    ProgressiveReachabilityProcessor,
)


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[ClimbPathTransformerWithGeneration, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        (model, config) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    config_path = Path(checkpoint_path).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize tokenizer
    tokenizer = ClimbPathTokenizer()
    
    # Initialize model
    model = ClimbPathTransformerWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 128),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model, config


def generate_climb_path(
    model: ClimbPathTransformerWithGeneration,
    tokenizer: ClimbPathTokenizer,
    grade: str,
    device: str = 'cpu',
    temperature: float = 1.0,
    min_holds: int = 3,
    max_holds: int = 30,
    use_constraints: bool = True,
    valid_holds: set = None,
    use_reachability: bool = True,
    max_reach: float = 5.0,
    reachability_mode: str = 'standard',
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Generate a single climb path.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        grade: Grade to generate (e.g., '7A')
        device: Device to run on
        temperature: Sampling temperature
        min_holds: Minimum number of holds
        max_holds: Maximum number of holds
        use_constraints: Whether to use logits processors
        valid_holds: Set of (x, y) tuples for valid hold positions (optional)
        use_reachability: Whether to enforce reachability constraints
        max_reach: Maximum reachable distance (default 5.0 from data analysis)
        reachability_mode: 'standard', 'adaptive', or 'progressive'
        
    Returns:
        (grade, holds) tuple
    """
    # Encode grade
    grade_token = tokenizer.encode_grade(grade)
    
    # Setup logits processors
    logits_processors = None
    if use_constraints:
        logits_processors = [
            ClimbPathLogitsProcessor(tokenizer),
            MinHoldsLogitsProcessor(tokenizer, min_holds=min_holds),
            MaxHoldsLogitsProcessor(tokenizer, max_holds=max_holds),
            NoRepeatHoldsLogitsProcessor(tokenizer, lookback=2),  # Prevent consecutive duplicates
        ]
        
        # Add reachability constraint
        if use_reachability:
            if reachability_mode == 'adaptive':
                logits_processors.append(
                    AdaptiveReachabilityProcessor(tokenizer, initial_reach=6.0, final_reach=4.5)
                )
            elif reachability_mode == 'progressive':
                logits_processors.append(
                    ProgressiveReachabilityProcessor(tokenizer, max_reach=max_reach, upward_bonus=0.5)
                )
            else:  # standard
                logits_processors.append(
                    ReachabilityLogitsProcessor(tokenizer, max_reach=max_reach)
                )
        
        # Add valid holds constraint if provided
        if valid_holds is not None:
            logits_processors.append(ValidHoldsLogitsProcessor(tokenizer, valid_holds))
    
    # Generate
    with torch.no_grad():
        tokens = model.generate_with_processors(
            grade_token=grade_token,
            tokenizer=tokenizer,
            logits_processors=logits_processors,
            max_length=128,
            temperature=temperature,
            device=device,
        )
    
    # Decode
    generated_grade, holds = tokenizer.decode(tokens.cpu().numpy())
    
    return generated_grade, holds


def visualize_path(holds: List[Tuple[int, int]], grade: str):
    """
    Print ASCII visualization of climb path on MoonBoard grid.
    
    Args:
        holds: List of (x, y) hold coordinates
        grade: Grade string
    """
    # Create grid (11 x 18)
    grid = [['.' for _ in range(11)] for _ in range(18)]
    
    # Mark holds
    for i, (x, y) in enumerate(holds):
        if 0 <= x <= 10 and 1 <= y <= 17:
            # Use numbers for first 9 holds, then letters
            if i < 9:
                marker = str(i + 1)
            elif i < 35:
                marker = chr(ord('A') + i - 9)
            else:
                marker = '*'
            grid[17 - (y - 1)][x] = marker  # Flip y for display
    
    # Print grid
    print(f"\nClimb Path - Grade: {grade}")
    print(f"Number of holds: {len(holds)}")
    print("\n   " + "".join([str(i) for i in range(11)]))
    print("  +" + "-" * 11 + "+")
    
    for i, row in enumerate(grid):
        y_coord = 17 - i
        print(f"{y_coord:2}|" + "".join(row) + "|")
    
    print("  +" + "-" * 11 + "+")
    print("   " + "".join([str(i) for i in range(11)]))
    
    # Print hold sequence
    print("\nHold sequence:")
    for i, (x, y) in enumerate(holds):
        print(f"  {i+1}. [{x}, {y}]")


def main():
    parser = argparse.ArgumentParser(description='Generate climb paths')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='checkpoints/climb_path/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    
    # Generation arguments
    parser.add_argument('--grade', type=str, default='7A',
                        help='Grade to generate (e.g., 7A, 6B+, 8A)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of paths to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--min_holds', type=int, default=3,
                        help='Minimum number of holds')
    parser.add_argument('--max_holds', type=int, default=30,
                        help='Maximum number of holds')
    parser.add_argument('--no_constraints', action='store_true',
                        help='Disable constraint enforcement')
    parser.add_argument('--valid_holds_only', action='store_true',
                        help='Only generate holds that exist on MoonBoard 2016')
    parser.add_argument('--train_csv', type=str, default='data/moonboard_train_quality.csv',
                        help='Path to training CSV for extracting valid holds')
    
    # Reachability arguments
    parser.add_argument('--no_reachability', action='store_true',
                        help='Disable reachability constraints')
    parser.add_argument('--max_reach', type=float, default=5.0,
                        help='Maximum reachable distance (default 5.0 from data analysis)')
    parser.add_argument('--reachability_mode', type=str, default='standard',
                        choices=['standard', 'adaptive', 'progressive'],
                        help='Reachability constraint mode')
    
    # Output arguments
    parser.add_argument('--save_json', type=str, default=None,
                        help='Save generated paths to JSON file')
    parser.add_argument('--visualize', action='store_true',
                        help='Print ASCII visualization of paths')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = ClimbPathTokenizer()
    
    # Load valid holds if requested
    valid_holds = None
    if args.valid_holds_only:
        print("\nLoading valid hold positions from dataset...")
        sys.path.append(str(Path(__file__).parent.parent))
        from models.valid_holds import load_valid_holds_from_dataset
        
        train_csv_path = Path(args.train_csv)
        if train_csv_path.exists():
            valid_holds = load_valid_holds_from_dataset(str(train_csv_path))
            print(f"Loaded {len(valid_holds)} valid holds from MoonBoard 2016")
            print(f"(Out of {11 * 18} possible grid positions)")
        else:
            print(f"Warning: Training CSV not found at {train_csv_path}")
            print("Proceeding without valid holds constraint")
    
    # Validate grade
    if args.grade not in tokenizer.GRADES:
        print(f"Error: Invalid grade '{args.grade}'")
        print(f"Valid grades: {', '.join(tokenizer.GRADES)}")
        return
    
    # Generate paths
    print(f"\nGenerating {args.num_samples} climb paths for grade {args.grade}...")
    print(f"Temperature: {args.temperature}")
    print(f"Constraints: {'Enabled' if not args.no_constraints else 'Disabled'}")
    print(f"Valid holds only: {'Yes' if args.valid_holds_only else 'No'}")
    print(f"Reachability: {'Enabled' if not args.no_reachability else 'Disabled'}")
    if not args.no_reachability:
        print(f"  Mode: {args.reachability_mode}")
        print(f"  Max reach: {args.max_reach}")
    print(f"Hold range: {args.min_holds}-{args.max_holds}")
    print("=" * 60)
    
    generated_paths = []
    
    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        
        grade, holds = generate_climb_path(
            model=model,
            tokenizer=tokenizer,
            grade=args.grade,
            device=args.device,
            temperature=args.temperature,
            min_holds=args.min_holds,
            max_holds=args.max_holds,
            use_constraints=not args.no_constraints,
            valid_holds=valid_holds,
            use_reachability=not args.no_reachability,
            max_reach=args.max_reach,
            reachability_mode=args.reachability_mode,
        )
        
        generated_paths.append({
            'grade': grade,
            'holds': holds,
            'num_holds': len(holds),
        })
        
        if args.visualize:
            visualize_path(holds, grade)
        else:
            print(f"Grade: {grade}")
            print(f"Holds: {holds}")
            print(f"Number of holds: {len(holds)}")
    
    # Save to JSON if requested
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'config': {
                'grade': args.grade,
                'temperature': args.temperature,
                'min_holds': args.min_holds,
                'max_holds': args.max_holds,
                'constraints_enabled': not args.no_constraints,
            },
            'paths': generated_paths,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved {args.num_samples} paths to {args.save_json}")
    
    print("\n" + "=" * 60)
    print("Generation complete!")


if __name__ == '__main__':
    main()
