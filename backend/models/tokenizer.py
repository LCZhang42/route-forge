"""
Tokenizer for MoonBoard climb path generation.

Vocabulary structure:
- Token 0: PAD (padding token)
- Token 1: BOS (beginning of sequence)
- Tokens 2-15: Grade tokens (6B to 8B+)
- Tokens 16-26: X coordinates (0-10)
- Tokens 27-44: Y coordinates (1-18, offset by 1)
- Token 45: EOS (end of sequence)

Total vocabulary size: 46 tokens
"""

import numpy as np
from typing import List, Tuple, Optional


class ClimbPathTokenizer:
    """Tokenizer for converting climb paths to/from token sequences."""
    
    # Grade vocabulary (14 grades)
    GRADES = ['6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']
    
    # Special tokens
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    GRADE_START = 2
    X_COORD_START = 16  # After 14 grades (2-15)
    Y_COORD_START = 27  # After 11 x coordinates (16-26)
    EOS_TOKEN = 45
    
    # Grid dimensions
    MAX_X = 10
    MAX_Y = 17
    MIN_Y = 1
    
    def __init__(self):
        """Initialize tokenizer with grade mappings."""
        self.grade_to_id = {grade: i + self.GRADE_START for i, grade in enumerate(self.GRADES)}
        self.id_to_grade = {v: k for k, v in self.grade_to_id.items()}
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return 46
    
    def encode_grade(self, grade: str) -> int:
        """Encode grade string to token ID."""
        if grade not in self.grade_to_id:
            raise ValueError(f"Invalid grade: {grade}. Must be one of {self.GRADES}")
        return self.grade_to_id[grade]
    
    def decode_grade(self, token_id: int) -> str:
        """Decode grade token ID to string."""
        if token_id not in self.id_to_grade:
            raise ValueError(f"Invalid grade token ID: {token_id}")
        return self.id_to_grade[token_id]
    
    def encode_x_coord(self, x: int) -> int:
        """Encode x coordinate to token ID."""
        if not (0 <= x <= self.MAX_X):
            raise ValueError(f"Invalid x coordinate: {x}. Must be in [0, {self.MAX_X}]")
        return self.X_COORD_START + x
    
    def decode_x_coord(self, token_id: int) -> int:
        """Decode x coordinate token ID."""
        if not (self.X_COORD_START <= token_id < self.Y_COORD_START):
            raise ValueError(f"Invalid x coordinate token ID: {token_id}")
        return token_id - self.X_COORD_START
    
    def encode_y_coord(self, y: int) -> int:
        """Encode y coordinate to token ID."""
        if not (self.MIN_Y <= y <= self.MAX_Y):
            raise ValueError(f"Invalid y coordinate: {y}. Must be in [{self.MIN_Y}, {self.MAX_Y}]")
        return self.Y_COORD_START + (y - self.MIN_Y)
    
    def decode_y_coord(self, token_id: int) -> int:
        """Decode y coordinate token ID."""
        if not (self.Y_COORD_START <= token_id < self.EOS_TOKEN):
            raise ValueError(f"Invalid y coordinate token ID: {token_id}")
        return (token_id - self.Y_COORD_START) + self.MIN_Y
    
    def encode_hold(self, x: int, y: int) -> List[int]:
        """Encode a single hold [x, y] to two tokens."""
        return [self.encode_x_coord(x), self.encode_y_coord(y)]
    
    def decode_hold(self, x_token: int, y_token: int) -> Tuple[int, int]:
        """Decode two tokens to a hold [x, y]."""
        return (self.decode_x_coord(x_token), self.decode_y_coord(y_token))
    
    def encode(self, grade: str, holds: List[List[int]]) -> np.ndarray:
        """
        Encode a complete climb path to token sequence.
        
        Args:
            grade: Grade string (e.g., '7A')
            holds: List of [x, y] coordinates
            
        Returns:
            Token sequence: [BOS, grade_token, x1, y1, x2, y2, ..., EOS]
        """
        tokens = [self.BOS_TOKEN, self.encode_grade(grade)]
        
        for hold in holds:
            if len(hold) != 2:
                raise ValueError(f"Hold must be [x, y], got {hold}")
            tokens.extend(self.encode_hold(hold[0], hold[1]))
        
        tokens.append(self.EOS_TOKEN)
        
        return np.array(tokens, dtype=np.int32)
    
    def encode_with_endpoints(self, grade: str, holds: List[List[int]]) -> np.ndarray:
        """
        Encode climb path with explicit start and end holds as conditioning.
        
        The model will learn to generate the path between start and end holds.
        
        Args:
            grade: Grade string (e.g., '7A')
            holds: List of [x, y] coordinates (full path including start and end)
            
        Returns:
            Token sequence: [BOS, grade, start_x, start_y, end_x, end_y, x2, y2, ..., xn-1, yn-1, EOS]
            where x2,y2,...,xn-1,yn-1 are the intermediate holds (excluding start and end)
        """
        if len(holds) < 2:
            raise ValueError(f"Path must have at least start and end holds, got {len(holds)} holds")
        
        # Start with BOS and grade
        tokens = [self.BOS_TOKEN, self.encode_grade(grade)]
        
        # Add start hold (first hold)
        start_hold = holds[0]
        tokens.extend(self.encode_hold(start_hold[0], start_hold[1]))
        
        # Add end hold (last hold)
        end_hold = holds[-1]
        tokens.extend(self.encode_hold(end_hold[0], end_hold[1]))
        
        # Add intermediate holds (everything between start and end)
        for hold in holds[1:-1]:
            if len(hold) != 2:
                raise ValueError(f"Hold must be [x, y], got {hold}")
            tokens.extend(self.encode_hold(hold[0], hold[1]))
        
        tokens.append(self.EOS_TOKEN)
        
        return np.array(tokens, dtype=np.int32)
    
    def decode(self, tokens: np.ndarray) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Decode token sequence to grade and holds.
        
        Args:
            tokens: Token sequence
            
        Returns:
            (grade, holds) where holds is list of (x, y) tuples
        """
        tokens = tokens.tolist() if isinstance(tokens, np.ndarray) else tokens
        
        if tokens[0] != self.BOS_TOKEN:
            raise ValueError("Sequence must start with BOS token")
        
        # Decode grade
        grade_token = tokens[1]
        grade = self.decode_grade(grade_token)
        
        # Decode holds (skip BOS and grade, stop at EOS)
        holds = []
        i = 2
        while i < len(tokens) - 1:  # -1 to skip EOS
            if i + 1 >= len(tokens):
                break
            x_token = tokens[i]
            y_token = tokens[i + 1]
            
            # Stop if we hit EOS
            if x_token == self.EOS_TOKEN or y_token == self.EOS_TOKEN:
                break
            
            holds.append(self.decode_hold(x_token, y_token))
            i += 2
        
        return grade, holds
    
    def decode_with_endpoints(self, tokens: np.ndarray) -> Tuple[str, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """
        Decode endpoint-conditioned token sequence.
        
        Args:
            tokens: Token sequence [BOS, grade, start_x, start_y, end_x, end_y, x2, y2, ..., EOS]
            
        Returns:
            (grade, start_hold, end_hold, intermediate_holds)
        """
        # Convert to list if it's a numpy array or torch tensor
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        elif hasattr(tokens, 'tolist'):  # PyTorch tensor
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            tokens = list(tokens)
        
        if tokens[0] != self.BOS_TOKEN:
            raise ValueError("Sequence must start with BOS token")
        
        if len(tokens) < 7:  # BOS, grade, start_x, start_y, end_x, end_y, EOS
            raise ValueError(f"Sequence too short for endpoint conditioning: {len(tokens)} tokens")
        
        # Decode grade
        grade_token = tokens[1]
        grade = self.decode_grade(grade_token)
        
        # Decode start hold (positions 2-3)
        start_hold = self.decode_hold(tokens[2], tokens[3])
        
        # Decode end hold (positions 4-5)
        end_hold = self.decode_hold(tokens[4], tokens[5])
        
        # Decode intermediate holds (from position 6 onwards)
        intermediate_holds = []
        i = 6
        while i < len(tokens) - 1:  # -1 to skip EOS
            if i + 1 >= len(tokens):
                break
            x_token = tokens[i]
            y_token = tokens[i + 1]
            
            # Stop if we hit EOS
            if x_token == self.EOS_TOKEN or y_token == self.EOS_TOKEN:
                break
            
            intermediate_holds.append(self.decode_hold(x_token, y_token))
            i += 2
        
        return grade, start_hold, end_hold, intermediate_holds
    
    def get_token_type(self, position: int) -> str:
        """
        Get the expected token type at a given position in the sequence.
        
        Args:
            position: Position in sequence (0-indexed)
            
        Returns:
            'bos', 'grade', 'x', 'y', or 'eos'
        """
        if position == 0:
            return 'bos'
        elif position == 1:
            return 'grade'
        elif position % 2 == 0:  # Even positions after grade are x coordinates
            return 'x'
        else:  # Odd positions after grade are y coordinates
            return 'y'
    
    def is_valid_token_at_position(self, token_id: int, position: int) -> bool:
        """Check if a token is valid at a given position."""
        token_type = self.get_token_type(position)
        
        if token_type == 'bos':
            return token_id == self.BOS_TOKEN
        elif token_type == 'grade':
            return self.GRADE_START <= token_id < self.X_COORD_START
        elif token_type == 'x':
            return self.X_COORD_START <= token_id < self.Y_COORD_START or token_id == self.EOS_TOKEN
        elif token_type == 'y':
            return self.Y_COORD_START <= token_id < self.EOS_TOKEN or token_id == self.EOS_TOKEN
        
        return False
