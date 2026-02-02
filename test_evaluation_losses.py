"""
Test script to verify evaluation script loss functions work correctly.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'backend'))

from models.tokenizer import ClimbPathTokenizer
from training.evaluate_model import compute_vertical_progression_loss, compute_position_constraint_loss

def test_evaluation_losses():
    """Test that evaluation loss functions work correctly."""
    print("="*60)
    print("Testing Evaluation Script Loss Functions")
    print("="*60)
    
    tokenizer = ClimbPathTokenizer()
    
    print("\n1. Testing with realistic vertical path")
    # Realistic path: Grade 7A, Path: [5,4] -> [3,8] -> [1,10] -> [6,12] -> [8,15] -> [6,17]
    realistic_path = torch.tensor([[
        tokenizer.BOS_TOKEN,
        tokenizer.encode_grade('7A'),
        tokenizer.encode_x_coord(5), tokenizer.encode_y_coord(4),
        tokenizer.encode_x_coord(3), tokenizer.encode_y_coord(8),
        tokenizer.encode_x_coord(1), tokenizer.encode_y_coord(10),
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(12),
        tokenizer.encode_x_coord(8), tokenizer.encode_y_coord(15),
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(17),
        tokenizer.EOS_TOKEN,
    ]])
    
    vertical_loss = compute_vertical_progression_loss(realistic_path, tokenizer)
    start_loss, end_loss = compute_position_constraint_loss(realistic_path, tokenizer)
    
    print(f"   Vertical Loss: {vertical_loss.item():.4f} (should be ~0)")
    print(f"   Start Loss: {start_loss.item():.4f} (should be ~0)")
    print(f"   End Loss: {end_loss.item():.4f} (should be ~0)")
    
    print("\n2. Testing with flat horizontal path")
    # Flat path: all at Y=10
    flat_path = torch.tensor([[
        tokenizer.BOS_TOKEN,
        tokenizer.encode_grade('7A'),
        tokenizer.encode_x_coord(0), tokenizer.encode_y_coord(10),
        tokenizer.encode_x_coord(2), tokenizer.encode_y_coord(10),
        tokenizer.encode_x_coord(4), tokenizer.encode_y_coord(10),
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(10),
        tokenizer.encode_x_coord(8), tokenizer.encode_y_coord(10),
        tokenizer.EOS_TOKEN,
    ]])
    
    vertical_loss_flat = compute_vertical_progression_loss(flat_path, tokenizer)
    start_loss_flat, end_loss_flat = compute_position_constraint_loss(flat_path, tokenizer)
    
    print(f"   Vertical Loss: {vertical_loss_flat.item():.4f} (should be HIGH)")
    print(f"   Start Loss: {start_loss_flat.item():.4f} (should be >0)")
    print(f"   End Loss: {end_loss_flat.item():.4f} (should be >0)")
    
    print("\n3. Testing batch processing")
    # Batch of 2 paths
    batch_paths = torch.tensor([
        [
            tokenizer.BOS_TOKEN,
            tokenizer.encode_grade('7A'),
            tokenizer.encode_x_coord(5), tokenizer.encode_y_coord(4),
            tokenizer.encode_x_coord(3), tokenizer.encode_y_coord(8),
            tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(17),
            tokenizer.EOS_TOKEN,
            tokenizer.PAD_TOKEN, tokenizer.PAD_TOKEN,
        ],
        [
            tokenizer.BOS_TOKEN,
            tokenizer.encode_grade('6C'),
            tokenizer.encode_x_coord(0), tokenizer.encode_y_coord(10),
            tokenizer.encode_x_coord(2), tokenizer.encode_y_coord(10),
            tokenizer.encode_x_coord(4), tokenizer.encode_y_coord(10),
            tokenizer.EOS_TOKEN,
            tokenizer.PAD_TOKEN, tokenizer.PAD_TOKEN,
        ]
    ])
    
    vertical_loss_batch = compute_vertical_progression_loss(batch_paths, tokenizer)
    start_loss_batch, end_loss_batch = compute_position_constraint_loss(batch_paths, tokenizer)
    
    print(f"   Batch Vertical Loss: {vertical_loss_batch.item():.4f}")
    print(f"   Batch Start Loss: {start_loss_batch.item():.4f}")
    print(f"   Batch End Loss: {end_loss_batch.item():.4f}")
    print(f"   [OK] Batch processing works")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"[OK] Vertical progression loss function working")
    print(f"[OK] Position constraint loss functions working")
    print(f"[OK] Batch processing working")
    
    # Verify expectations
    assert vertical_loss.item() < vertical_loss_flat.item(), "Realistic path should have lower vertical loss"
    assert start_loss.item() == 0.0, "Good start should have zero loss"
    assert end_loss.item() == 0.0, "Good end should have zero loss"
    
    print("\n[SUCCESS] All evaluation loss tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_evaluation_losses()
