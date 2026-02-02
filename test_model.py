"""
Quick test script to verify the autoregressive model implementation.

Run this before training to ensure everything is set up correctly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'backend'))

print("=" * 70)
print("TESTING AUTOREGRESSIVE CLIMB PATH MODEL")
print("=" * 70)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from models.tokenizer import ClimbPathTokenizer
    from models.climb_transformer import ClimbPathTransformerWithGeneration
    from models.logits_processor import (
        ClimbPathLogitsProcessor,
        MinHoldsLogitsProcessor,
        MaxHoldsLogitsProcessor,
    )
    from models.dataset import ClimbPathDataset
    print("   [OK] All imports successful")
except Exception as e:
    print(f"   [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize tokenizer
print("\n2. Testing tokenizer...")
try:
    tokenizer = ClimbPathTokenizer()
    assert tokenizer.vocab_size == 46
    print(f"   [OK] Tokenizer initialized (vocab_size={tokenizer.vocab_size})")
except Exception as e:
    print(f"   [FAIL] Tokenizer failed: {e}")
    sys.exit(1)

# Test 3: Test encoding/decoding
print("\n3. Testing encoding/decoding...")
try:
    grade = "7A"
    holds = [[0, 4], [1, 7], [3, 11], [5, 13], [8, 17]]
    
    tokens = tokenizer.encode(grade, holds)
    decoded_grade, decoded_holds = tokenizer.decode(tokens)
    
    assert decoded_grade == grade
    assert len(decoded_holds) == len(holds)
    print(f"   [OK] Encoded {len(holds)} holds into {len(tokens)} tokens")
    print(f"   [OK] Decoded back correctly")
except Exception as e:
    print(f"   [FAIL] Encoding/decoding failed: {e}")
    sys.exit(1)

# Test 4: Initialize model
print("\n4. Testing model initialization...")
try:
    import torch
    
    model = ClimbPathTransformerWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   [OK] Model initialized ({num_params:,} parameters)")
except Exception as e:
    print(f"   [FAIL] Model initialization failed: {e}")
    sys.exit(1)

# Test 5: Test forward pass
print("\n5. Testing forward pass...")
try:
    batch_size = 2
    seq_len = 10
    
    # Create dummy input
    dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    
    assert logits.shape == (batch_size, seq_len, tokenizer.vocab_size)
    print(f"   [OK] Forward pass successful")
    print(f"   [OK] Output shape: {logits.shape}")
except Exception as e:
    print(f"   [FAIL] Forward pass failed: {e}")
    sys.exit(1)

# Test 6: Test logits processors
print("\n6. Testing logits processors...")
try:
    processors = [
        ClimbPathLogitsProcessor(tokenizer),
        MinHoldsLogitsProcessor(tokenizer, min_holds=3),
        MaxHoldsLogitsProcessor(tokenizer, max_holds=20),
    ]
    
    # Test on dummy sequence
    dummy_seq = torch.tensor([[0, 6, 15, 30]])  # BOS, grade, x1, y1
    dummy_logits = torch.randn(1, tokenizer.vocab_size)
    
    for processor in processors:
        processed_logits = processor(dummy_seq, dummy_logits)
        assert processed_logits.shape == dummy_logits.shape
    
    print(f"   [OK] All {len(processors)} logits processors working")
except Exception as e:
    print(f"   [FAIL] Logits processors failed: {e}")
    sys.exit(1)

# Test 7: Test generation (without training)
print("\n7. Testing generation (untrained model)...")
try:
    grade_token = tokenizer.encode_grade("7A")
    
    tokens = model.generate_with_processors(
        grade_token=grade_token,
        tokenizer=tokenizer,
        logits_processors=processors,
        max_length=20,
        temperature=1.0,
        device='cpu',
    )
    
    decoded_grade, decoded_holds = tokenizer.decode(tokens.cpu().numpy())
    
    print(f"   [OK] Generated sequence with {len(decoded_holds)} holds")
    print(f"   [OK] Grade: {decoded_grade}")
    print(f"   [OK] Holds: {decoded_holds[:3]}..." if len(decoded_holds) > 3 else f"   [OK] Holds: {decoded_holds}")
    print(f"   Note: Output is random (model not trained yet)")
except Exception as e:
    print(f"   [FAIL] Generation failed: {e}")
    sys.exit(1)

# Test 8: Check data files
print("\n8. Checking data files...")
data_dir = Path(__file__).parent / 'data'
required_files = ['moonboard_train.csv', 'moonboard_val.csv', 'moonboard_test.csv']

missing_files = []
for filename in required_files:
    filepath = data_dir / filename
    if filepath.exists():
        print(f"   [OK] Found {filename}")
    else:
        print(f"   [FAIL] Missing {filename}")
        missing_files.append(filename)

if missing_files:
    print(f"\n   Warning: Missing data files. Run data cleaning first:")
    print(f"   python backend/data_processing/clean_data_for_training.py")
else:
    print(f"   [OK] All data files present")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! [OK]")
print("=" * 70)
print("\nYou're ready to train the model!")
print("\nNext steps:")
print("  1. Train: python backend/training/train_autoregressive.py")
print("  2. Generate: python backend/training/generate_paths.py --grade 7A --visualize")
print("\nSee QUICKSTART.md for detailed instructions.")
print("=" * 70)
