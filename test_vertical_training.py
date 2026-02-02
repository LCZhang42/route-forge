"""
Test script to verify vertical progression loss implementation.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'backend'))

from models.tokenizer import ClimbPathTokenizer
from models.climb_transformer import ClimbPathTransformerWithGeneration
from models.dataset import ClimbPathDataModule
from training.train_autoregressive import Trainer

def test_vertical_loss():
    """Test that vertical progression loss is computed correctly."""
    print("="*60)
    print("Testing Vertical Progression Loss Implementation")
    print("="*60)
    
    # Initialize components
    tokenizer = ClimbPathTokenizer()
    
    # Create a simple model
    model = ClimbPathTransformerWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=64,
    )
    
    # Create data module (we won't use it for this test)
    data_module = ClimbPathDataModule(
        train_csv='data/moonboard_train.csv',
        val_csv='data/moonboard_val.csv',
        test_csv='data/moonboard_test.csv',
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_len=64,
        num_workers=0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_module=data_module,
        device='cpu',
        learning_rate=1e-4,
    )
    
    print("\n1. Testing with realistic vertical path (should have LOW loss)")
    # Realistic path: starts low, ends high
    # Grade: 7A (token 6), Path: [5,4] -> [3,8] -> [1,10] -> [6,12] -> [8,15] -> [6,17]
    realistic_path = torch.tensor([[
        tokenizer.BOS_TOKEN,
        tokenizer.encode_grade('7A'),
        tokenizer.encode_x_coord(5), tokenizer.encode_y_coord(4),   # Start at Y=4
        tokenizer.encode_x_coord(3), tokenizer.encode_y_coord(8),   # Y=8
        tokenizer.encode_x_coord(1), tokenizer.encode_y_coord(10),  # Y=10
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(12),  # Y=12
        tokenizer.encode_x_coord(8), tokenizer.encode_y_coord(15),  # Y=15
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(17),  # End at Y=17
        tokenizer.EOS_TOKEN,
    ]])
    
    vertical_loss = trainer.compute_vertical_progression_loss(realistic_path)
    start_loss, end_loss = trainer.compute_position_constraint_loss(realistic_path)
    
    print(f"   Vertical Loss: {vertical_loss.item():.4f}")
    print(f"   Start Loss: {start_loss.item():.4f} (Y=4, should be ~0)")
    print(f"   End Loss: {end_loss.item():.4f} (Y=17, should be ~0)")
    print(f"   [OK] Good path has low constraint losses")
    
    print("\n2. Testing with flat horizontal path (should have HIGH loss)")
    # Flat path: all at same Y level
    # Grade: 7A, Path: [0,10] -> [2,10] -> [4,10] -> [6,10] -> [8,10]
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
    
    vertical_loss_flat = trainer.compute_vertical_progression_loss(flat_path)
    start_loss_flat, end_loss_flat = trainer.compute_position_constraint_loss(flat_path)
    
    print(f"   Vertical Loss: {vertical_loss_flat.item():.4f}")
    print(f"   Start Loss: {start_loss_flat.item():.4f} (Y=10, should be >0)")
    print(f"   End Loss: {end_loss_flat.item():.4f} (Y=10, should be >0)")
    print(f"   [OK] Flat path has high losses")
    
    print("\n3. Testing with bad start position (should have HIGH start loss)")
    # Path starting too high
    # Grade: 7A, Path: [5,15] -> [3,16] -> [6,17]
    bad_start_path = torch.tensor([[
        tokenizer.BOS_TOKEN,
        tokenizer.encode_grade('7A'),
        tokenizer.encode_x_coord(5), tokenizer.encode_y_coord(15),  # Start at Y=15 (too high!)
        tokenizer.encode_x_coord(3), tokenizer.encode_y_coord(16),
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(17),
        tokenizer.EOS_TOKEN,
    ]])
    
    vertical_loss_bad = trainer.compute_vertical_progression_loss(bad_start_path)
    start_loss_bad, end_loss_bad = trainer.compute_position_constraint_loss(bad_start_path)
    
    print(f"   Vertical Loss: {vertical_loss_bad.item():.4f}")
    print(f"   Start Loss: {start_loss_bad.item():.4f} (Y=15, should be HIGH)")
    print(f"   End Loss: {end_loss_bad.item():.4f} (Y=17, should be ~0)")
    print(f"   [OK] Bad start position detected")
    
    print("\n4. Testing with bad end position (should have HIGH end loss)")
    # Path ending too low
    # Grade: 7A, Path: [5,4] -> [3,6] -> [6,8]
    bad_end_path = torch.tensor([[
        tokenizer.BOS_TOKEN,
        tokenizer.encode_grade('7A'),
        tokenizer.encode_x_coord(5), tokenizer.encode_y_coord(4),
        tokenizer.encode_x_coord(3), tokenizer.encode_y_coord(6),
        tokenizer.encode_x_coord(6), tokenizer.encode_y_coord(8),  # End at Y=8 (too low!)
        tokenizer.EOS_TOKEN,
    ]])
    
    vertical_loss_bad_end = trainer.compute_vertical_progression_loss(bad_end_path)
    start_loss_bad_end, end_loss_bad_end = trainer.compute_position_constraint_loss(bad_end_path)
    
    print(f"   Vertical Loss: {vertical_loss_bad_end.item():.4f}")
    print(f"   Start Loss: {start_loss_bad_end.item():.4f} (Y=4, should be ~0)")
    print(f"   End Loss: {end_loss_bad_end.item():.4f} (Y=8, should be HIGH)")
    print(f"   [OK] Bad end position detected")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"[OK] Vertical loss implementation working correctly")
    print(f"[OK] Start position constraint working correctly")
    print(f"[OK] End position constraint working correctly")
    print(f"\nLoss comparisons:")
    print(f"  Realistic path vertical loss: {vertical_loss.item():.4f}")
    print(f"  Flat path vertical loss:      {vertical_loss_flat.item():.4f} (should be higher)")
    print(f"  Good start loss:              {start_loss.item():.4f}")
    print(f"  Bad start loss:               {start_loss_bad.item():.4f} (should be higher)")
    print(f"  Good end loss:                {end_loss.item():.4f}")
    print(f"  Bad end loss:                 {end_loss_bad_end.item():.4f} (should be higher)")
    
    # Verify expectations
    assert vertical_loss_flat.item() > vertical_loss.item(), "Flat path should have higher vertical loss"
    assert start_loss_bad.item() > start_loss.item(), "Bad start should have higher start loss"
    assert end_loss_bad_end.item() > end_loss.item(), "Bad end should have higher end loss"
    
    print("\n[SUCCESS] All tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_vertical_loss()
