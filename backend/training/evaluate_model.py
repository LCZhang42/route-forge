"""
Evaluate trained model on test data.

Loads a trained checkpoint and computes test loss and perplexity.
"""

import torch
import torch.nn as nn
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from typing import Tuple

sys.path.append(str(Path(__file__).parent.parent))

from models.tokenizer import ClimbPathTokenizer
from models.climb_transformer import ClimbPathTransformerWithGeneration
from models.dataset import ClimbPathDataModule


def compute_vertical_progression_loss(input_ids: torch.Tensor, tokenizer: ClimbPathTokenizer) -> torch.Tensor:
    """
    Compute loss to encourage vertical progression in climbing paths.
    Penalizes flat paths and rewards Y-coordinate variety.
    
    Args:
        input_ids: Token sequence [batch_size, seq_len]
        tokenizer: ClimbPathTokenizer instance
        
    Returns:
        Vertical progression loss
    """
    batch_size, seq_len = input_ids.shape
    
    # Extract Y-coordinate tokens (odd positions after grade token)
    y_positions = list(range(3, seq_len, 2))
    
    if len(y_positions) < 2:
        return torch.tensor(0.0, device=input_ids.device)
    
    # Get Y tokens and convert to actual Y coordinates
    y_tokens = input_ids[:, y_positions]
    
    # Filter out padding, EOS, and invalid tokens
    valid_mask = (y_tokens >= tokenizer.Y_COORD_START) & (y_tokens < tokenizer.EOS_TOKEN)
    
    total_loss = 0.0
    num_valid_sequences = 0
    
    for b in range(batch_size):
        valid_y = y_tokens[b][valid_mask[b]]
        
        if len(valid_y) < 2:
            continue
        
        # Convert tokens to actual Y coordinates (1-18)
        y_coords = valid_y - tokenizer.Y_COORD_START + 1
        
        # Compute vertical progression metrics
        y_diffs = y_coords[1:] - y_coords[:-1]
        
        # Loss 1: Penalize negative progression (going down too much)
        downward_penalty = torch.clamp(-y_diffs - 2, min=0).float().mean()
        
        # Loss 2: Encourage overall upward progression
        total_vertical_gain = y_coords[-1] - y_coords[0]
        min_expected_gain = 8.0
        upward_loss = torch.clamp(min_expected_gain - total_vertical_gain, min=0).float()
        
        # Loss 3: Penalize too many consecutive flat moves
        flat_moves = (torch.abs(y_diffs) < 1).float()
        flat_penalty = flat_moves.mean()
        
        # Combine losses
        sequence_loss = downward_penalty + 0.5 * upward_loss + 0.3 * flat_penalty
        total_loss += sequence_loss
        num_valid_sequences += 1
    
    if num_valid_sequences == 0:
        return torch.tensor(0.0, device=input_ids.device)
    
    return total_loss / num_valid_sequences


def compute_position_constraint_loss(input_ids: torch.Tensor, tokenizer: ClimbPathTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss to encourage valid start and end positions.
    Start positions should be low (Y <= 6), end positions should be high (Y >= 12).
    
    Args:
        input_ids: Token sequence [batch_size, seq_len]
        tokenizer: ClimbPathTokenizer instance
        
    Returns:
        (start_loss, end_loss)
    """
    batch_size, seq_len = input_ids.shape
    
    start_loss = 0.0
    end_loss = 0.0
    num_valid_sequences = 0
    
    for b in range(batch_size):
        # Find Y-coordinate tokens
        y_positions = list(range(3, seq_len, 2))
        if len(y_positions) < 1:
            continue
        
        y_tokens = input_ids[b, y_positions]
        valid_mask = (y_tokens >= tokenizer.Y_COORD_START) & (y_tokens < tokenizer.EOS_TOKEN)
        valid_y = y_tokens[valid_mask]
        
        if len(valid_y) < 1:
            continue
        
        # Convert to Y coordinates
        y_coords = valid_y - tokenizer.Y_COORD_START + 1
        
        # Start position constraint: first hold should have Y <= 6
        start_y = y_coords[0].float()
        max_start_y = 6.0
        start_penalty = torch.clamp(start_y - max_start_y, min=0)
        start_loss += start_penalty
        
        # End position constraint: last hold should have Y >= 12
        end_y = y_coords[-1].float()
        min_end_y = 12.0
        end_penalty = torch.clamp(min_end_y - end_y, min=0)
        end_loss += end_penalty
        
        num_valid_sequences += 1
    
    if num_valid_sequences == 0:
        return torch.tensor(0.0, device=input_ids.device), torch.tensor(0.0, device=input_ids.device)
    
    return start_loss / num_valid_sequences, end_loss / num_valid_sequences


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    tokenizer: ClimbPathTokenizer,
    data_module: ClimbPathDataModule,
    device: str = 'cpu',
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        data_module: Data module with test data
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Loss weights (matching training)
    vertical_loss_weight = 0.5
    start_constraint_weight = 0.3
    end_constraint_weight = 0.3
    
    total_loss = 0
    total_ce_loss = 0
    total_vertical_loss = 0
    total_start_loss = 0
    total_end_loss = 0
    total_tokens = 0
    num_batches = 0
    
    test_loader = data_module.test_dataloader()
    pbar = tqdm(test_loader, desc='Evaluating on test set')
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits = model(
            src=input_ids,
            src_key_padding_mask=(attention_mask == 0),
        )
        
        # Compute cross-entropy loss (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Flatten for loss computation
        ce_loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        # Compute vertical progression loss
        vertical_loss = compute_vertical_progression_loss(input_ids, tokenizer)
        
        # Compute position constraint losses
        start_loss, end_loss = compute_position_constraint_loss(input_ids, tokenizer)
        
        # Combined loss (matching training)
        loss = ce_loss + \
               vertical_loss_weight * vertical_loss + \
               start_constraint_weight * start_loss + \
               end_constraint_weight * end_loss
        
        # Count actual tokens (excluding padding)
        num_tokens = shift_mask.sum().item()
        
        total_loss += loss.item() * num_tokens
        total_ce_loss += ce_loss.item() * num_tokens
        total_vertical_loss += vertical_loss.item() * num_tokens
        total_start_loss += start_loss.item() * num_tokens
        total_end_loss += end_loss.item() * num_tokens
        total_tokens += num_tokens
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'vert': f'{vertical_loss.item():.4f}'
        })
    
    # Compute average metrics
    avg_loss = total_loss / total_tokens
    avg_ce_loss = total_ce_loss / total_tokens
    avg_vertical_loss = total_vertical_loss / total_tokens
    avg_start_loss = total_start_loss / total_tokens
    avg_end_loss = total_end_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()
    
    metrics = {
        'test_loss': avg_loss,
        'test_ce_loss': avg_ce_loss,
        'test_vertical_loss': avg_vertical_loss,
        'test_start_loss': avg_start_loss,
        'test_end_loss': avg_end_loss,
        'test_perplexity': perplexity,
        'num_batches': num_batches,
        'num_tokens': total_tokens,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained climb path model')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, default='checkpoints/climb_path/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='checkpoints/climb_path/config.json',
                        help='Path to training config')
    
    # Data arguments
    parser.add_argument('--test_csv', type=str, default='data/moonboard_test_quality.csv',
                        help='Path to test CSV')
    parser.add_argument('--train_csv', type=str, default='data/moonboard_train_quality.csv',
                        help='Path to training CSV (needed for data module)')
    parser.add_argument('--val_csv', type=str, default='data/moonboard_val_quality.csv',
                        help='Path to validation CSV (needed for data module)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    
    args = parser.parse_args()
    
    # Load config if available
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Override with config values
        d_model = config.get('d_model', 256)
        nhead = config.get('nhead', 8)
        num_layers = config.get('num_layers', 6)
        dim_feedforward = config.get('dim_feedforward', 1024)
        dropout = config.get('dropout', 0.1)
        max_seq_len = config.get('max_seq_len', 128)
    else:
        print("Config file not found, using default values")
        d_model = 256
        nhead = 8
        num_layers = 6
        dim_feedforward = 1024
        dropout = 0.1
        max_seq_len = 128
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = ClimbPathTokenizer()
    
    # Initialize data module
    print("Loading test data...")
    data_module = ClimbPathDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        num_workers=args.num_workers,
    )
    data_module.setup()
    
    # Initialize model
    print("Initializing model...")
    model = ClimbPathTransformerWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss at checkpoint: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Evaluate
    print(f"\nEvaluating on {len(data_module.test_dataset)} test examples...")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    metrics = evaluate_model(model, tokenizer, data_module, args.device)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Total Loss:           {metrics['test_loss']:.4f}")
    print(f"Cross-Entropy Loss:   {metrics['test_ce_loss']:.4f}")
    print(f"Vertical Loss:        {metrics['test_vertical_loss']:.4f}")
    print(f"Start Position Loss:  {metrics['test_start_loss']:.4f}")
    print(f"End Position Loss:    {metrics['test_end_loss']:.4f}")
    print(f"Perplexity:           {metrics['test_perplexity']:.4f}")
    print(f"Num Batches:          {metrics['num_batches']}")
    print(f"Num Tokens:           {metrics['num_tokens']:,}")
    print("=" * 60)
    
    # Save results
    results_path = checkpoint_path.parent / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
