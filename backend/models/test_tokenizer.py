"""
Test script for tokenizer functionality.

Run this to verify the tokenizer works correctly before training.
"""

from tokenizer import ClimbPathTokenizer


def test_tokenizer():
    """Test basic tokenizer functionality."""
    print("=" * 60)
    print("Testing ClimbPathTokenizer")
    print("=" * 60)
    
    tokenizer = ClimbPathTokenizer()
    
    # Test 1: Vocabulary size
    print(f"\n1. Vocabulary size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 45, "Vocab size should be 45"
    print("   ✓ Correct")
    
    # Test 2: Grade encoding/decoding
    print("\n2. Testing grade encoding/decoding...")
    for grade in tokenizer.GRADES:
        token = tokenizer.encode_grade(grade)
        decoded = tokenizer.decode_grade(token)
        assert decoded == grade, f"Grade mismatch: {grade} != {decoded}"
        print(f"   {grade:>4} → token {token:2d} → {decoded:>4} ✓")
    
    # Test 3: Coordinate encoding/decoding
    print("\n3. Testing coordinate encoding/decoding...")
    test_coords = [(0, 1), (5, 10), (10, 17)]
    for x, y in test_coords:
        x_token = tokenizer.encode_x_coord(x)
        y_token = tokenizer.encode_y_coord(y)
        x_decoded = tokenizer.decode_x_coord(x_token)
        y_decoded = tokenizer.decode_y_coord(y_token)
        assert (x_decoded, y_decoded) == (x, y), f"Coord mismatch: ({x}, {y}) != ({x_decoded}, {y_decoded})"
        print(f"   ({x:2d}, {y:2d}) → tokens ({x_token}, {y_token}) → ({x_decoded:2d}, {y_decoded:2d}) ✓")
    
    # Test 4: Full path encoding/decoding
    print("\n4. Testing full path encoding/decoding...")
    grade = "7A"
    holds = [[0, 4], [1, 7], [3, 11], [5, 13], [8, 17]]
    
    print(f"   Input: grade={grade}, holds={holds}")
    
    tokens = tokenizer.encode(grade, holds)
    print(f"   Tokens: {tokens.tolist()}")
    print(f"   Length: {len(tokens)}")
    
    decoded_grade, decoded_holds = tokenizer.decode(tokens)
    print(f"   Decoded: grade={decoded_grade}, holds={decoded_holds}")
    
    assert decoded_grade == grade, f"Grade mismatch: {grade} != {decoded_grade}"
    assert len(decoded_holds) == len(holds), f"Length mismatch: {len(holds)} != {len(decoded_holds)}"
    for i, (orig, dec) in enumerate(zip(holds, decoded_holds)):
        assert tuple(orig) == dec, f"Hold {i} mismatch: {orig} != {dec}"
    
    print("   ✓ Perfect match!")
    
    # Test 5: Token type detection
    print("\n5. Testing token type detection...")
    sequence_types = [
        (0, 'bos'),
        (1, 'grade'),
        (2, 'x'),
        (3, 'y'),
        (4, 'x'),
        (5, 'y'),
    ]
    for pos, expected_type in sequence_types:
        token_type = tokenizer.get_token_type(pos)
        assert token_type == expected_type, f"Type mismatch at pos {pos}: {expected_type} != {token_type}"
        print(f"   Position {pos}: {token_type} ✓")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    
    # Minimum path (1 hold)
    min_holds = [[5, 9]]
    tokens = tokenizer.encode("6B", min_holds)
    decoded_grade, decoded_holds = tokenizer.decode(tokens)
    assert len(decoded_holds) == 1
    print(f"   Min path (1 hold): ✓")
    
    # Maximum coordinates
    max_holds = [[10, 17]]
    tokens = tokenizer.encode("8B+", max_holds)
    decoded_grade, decoded_holds = tokenizer.decode(tokens)
    assert decoded_holds[0] == (10, 17)
    print(f"   Max coordinates (10, 17): ✓")
    
    # Long path
    long_holds = [[i % 11, (i % 17) + 1] for i in range(20)]
    tokens = tokenizer.encode("7B", long_holds)
    decoded_grade, decoded_holds = tokenizer.decode(tokens)
    assert len(decoded_holds) == 20
    print(f"   Long path (20 holds): ✓")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_tokenizer()
