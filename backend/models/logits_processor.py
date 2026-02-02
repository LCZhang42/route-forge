"""
Custom logits processor for enforcing valid token generation in climb path sequences.

Ensures that at each position, only valid tokens can be generated:
- Position 0: BOS token only
- Position 1: Grade tokens only
- Even positions (2, 4, 6, ...): X coordinates or EOS
- Odd positions (3, 5, 7, ...): Y coordinates or EOS
"""

import torch
from transformers.generation.logits_process import LogitsProcessor


class ClimbPathLogitsProcessor(LogitsProcessor):
    """
    Logits processor that masks invalid tokens based on sequence position.
    
    This ensures structural validity of generated climb paths by only allowing
    appropriate token types at each position.
    """
    
    def __init__(self, tokenizer, mask_value: float = -float('inf')):
        """
        Initialize the logits processor.
        
        Args:
            tokenizer: ClimbPathTokenizer instance
            mask_value: Value to set for masked (invalid) tokens
        """
        self.tokenizer = tokenizer
        self.mask_value = mask_value
        
        # Precompute token ranges for efficiency
        self.bos_token = tokenizer.BOS_TOKEN
        self.grade_start = tokenizer.GRADE_START
        self.x_coord_start = tokenizer.X_COORD_START
        self.y_coord_start = tokenizer.Y_COORD_START
        self.eos_token = tokenizer.EOS_TOKEN
        self.vocab_size = tokenizer.vocab_size
    
    def get_valid_token_range(self, position: int) -> tuple:
        """
        Get the valid token range for a given position.
        
        Args:
            position: Current position in sequence
            
        Returns:
            (start_idx, end_idx) tuple for valid token range (inclusive)
        """
        if position == 0:
            # BOS token only
            return (self.bos_token, self.bos_token)
        elif position == 1:
            # Grade tokens only
            return (self.grade_start, self.x_coord_start - 1)
        elif position % 2 == 0:
            # Even positions: X coordinates (can also end with EOS)
            return (self.x_coord_start, self.y_coord_start - 1)
        else:
            # Odd positions: Y coordinates (can also end with EOS)
            return (self.y_coord_start, self.eos_token - 1)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply logits masking based on current sequence position.
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
            
        Returns:
            Modified scores with invalid tokens masked
        """
        batch_size = scores.shape[0]
        current_position = input_ids.shape[1]
        
        # Get valid token range for this position
        start_idx, end_idx = self.get_valid_token_range(current_position)
        
        # Create mask: True for tokens to mask (invalid)
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=scores.device)
        mask[start_idx:end_idx + 1] = False  # Valid tokens
        
        # For positions where we can generate holds, also allow EOS
        if current_position > 1 and current_position % 2 == 0:
            mask[self.eos_token] = False
        
        # Expand mask for batch
        mask = mask.unsqueeze(0).expand(batch_size, -1)
        
        # Apply mask
        scores = scores.masked_fill(mask, self.mask_value)
        
        return scores


class MinHoldsLogitsProcessor(LogitsProcessor):
    """
    Prevents EOS generation until minimum number of holds are generated.
    
    This ensures generated paths have at least a minimum length.
    """
    
    def __init__(self, tokenizer, min_holds: int = 3):
        """
        Initialize the processor.
        
        Args:
            tokenizer: ClimbPathTokenizer instance
            min_holds: Minimum number of holds required
        """
        self.tokenizer = tokenizer
        self.min_holds = min_holds
        self.eos_token = tokenizer.EOS_TOKEN
        # Minimum sequence length = BOS + grade + (min_holds * 2) + EOS
        self.min_length = 2 + (min_holds * 2)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask EOS token if sequence is too short.
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
            
        Returns:
            Modified scores with EOS masked if needed
        """
        current_length = input_ids.shape[1]
        
        # If we haven't reached minimum length, mask EOS
        if current_length < self.min_length:
            scores[:, self.eos_token] = -float('inf')
        
        return scores


class MaxHoldsLogitsProcessor(LogitsProcessor):
    """
    Forces EOS generation after maximum number of holds.
    
    This prevents sequences from becoming too long.
    """
    
    def __init__(self, tokenizer, max_holds: int = 30):
        """
        Initialize the processor.
        
        Args:
            tokenizer: ClimbPathTokenizer instance
            max_holds: Maximum number of holds allowed
        """
        self.tokenizer = tokenizer
        self.max_holds = max_holds
        self.eos_token = tokenizer.EOS_TOKEN
        # Maximum sequence length = BOS + grade + (max_holds * 2)
        self.max_length = 2 + (max_holds * 2)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Force EOS if sequence is at maximum length.
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
            
        Returns:
            Modified scores with only EOS allowed if at max length
        """
        current_length = input_ids.shape[1]
        
        # If we've reached maximum length, force EOS
        if current_length >= self.max_length:
            # Mask everything except EOS
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[:, self.eos_token] = False
            scores = scores.masked_fill(mask, -float('inf'))
        
        return scores


class NoRepeatHoldsLogitsProcessor(LogitsProcessor):
    """
    Prevents generating holds that were recently used.
    
    Tracks the last N holds and prevents repeating any of them.
    This significantly improves path quality by forcing variety.
    """
    
    def __init__(self, tokenizer, lookback: int = 3):
        """
        Initialize the processor.
        
        Args:
            tokenizer: ClimbPathTokenizer instance
            lookback: Number of recent holds to check (default: 3)
        """
        self.tokenizer = tokenizer
        self.x_coord_start = tokenizer.X_COORD_START
        self.y_coord_start = tokenizer.Y_COORD_START
        self.eos_token = tokenizer.EOS_TOKEN
        self.lookback = lookback
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask recently used holds to prevent repetition.
        
        Sequence structure: [BOS, GRADE, X1, Y1, X2, Y2, X3, Y3, ..., EOS]
        Position indices:    0    1     2   3   4   5   6   7
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
            
        Returns:
            Modified scores with recent holds masked
        """
        batch_size = scores.shape[0]
        current_position = input_ids.shape[1]
        
        # Only apply for hold coordinates (positions >= 2)
        if current_position < 2:
            return scores
        
        # Process each batch item
        for batch_idx in range(batch_size):
            # Extract all complete holds so far
            # Holds start at position 2 (after BOS and GRADE)
            holds_start = 2
            num_complete_holds = (current_position - holds_start) // 2
            
            if num_complete_holds == 0:
                continue
            
            # Get recent holds (up to lookback)
            recent_holds = []
            for i in range(max(0, num_complete_holds - self.lookback), num_complete_holds):
                x_pos = holds_start + i * 2
                y_pos = holds_start + i * 2 + 1
                if y_pos < current_position:
                    x_token = input_ids[batch_idx, x_pos].item()
                    y_token = input_ids[batch_idx, y_pos].item()
                    recent_holds.append((x_token, y_token))
            
            # Position 2, 4, 6, ... are X coordinates
            # Position 3, 5, 7, ... are Y coordinates
            
            if current_position % 2 == 0:
                # We're about to generate an X coordinate
                # Mask all X coordinates from recent holds
                for x_token, _ in recent_holds:
                    if x_token < scores.shape[1]:
                        scores[batch_idx, x_token] = -float('inf')
            
            else:
                # We're about to generate a Y coordinate
                # The X coordinate was just generated at position -1
                current_x_token = input_ids[batch_idx, -1].item()
                
                # Mask Y coordinates that would create a repeated hold
                for x_token, y_token in recent_holds:
                    if x_token == current_x_token and y_token < scores.shape[1]:
                        scores[batch_idx, y_token] = -float('inf')
        
        return scores


class ValidHoldsLogitsProcessor(LogitsProcessor):
    """
    Masks logits for invalid hold positions during generation.
    
    Only allows generation of holds that actually exist on the MoonBoard 2016.
    This prevents the model from generating coordinates where no physical hold exists.
    """
    
    def __init__(self, tokenizer, valid_holds: set):
        """
        Initialize the processor.
        
        Args:
            tokenizer: ClimbPathTokenizer instance
            valid_holds: Set of (x, y) tuples representing valid hold positions
        """
        self.tokenizer = tokenizer
        self.valid_holds = valid_holds
        self.x_coord_start = tokenizer.X_COORD_START
        self.y_coord_start = tokenizer.Y_COORD_START
        self.eos_token = tokenizer.EOS_TOKEN
        
        # Precompute valid X and Y coordinates
        self.valid_x_coords = set()
        self.valid_y_coords_by_x = {}  # Map x -> set of valid y coords
        
        for x, y in valid_holds:
            self.valid_x_coords.add(x)
            if x not in self.valid_y_coords_by_x:
                self.valid_y_coords_by_x[x] = set()
            self.valid_y_coords_by_x[x].add(y)
        
        # Convert to token indices
        self.valid_x_tokens = {self.x_coord_start + x for x in self.valid_x_coords}
        self.valid_y_tokens_by_x = {
            x: {self.y_coord_start + y - 1 for y in y_set}  # y is 1-indexed
            for x, y_set in self.valid_y_coords_by_x.items()
        }
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Mask invalid hold positions based on sequence context.
        
        Args:
            input_ids: Current input sequence [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
            
        Returns:
            Modified scores with invalid holds masked
        """
        batch_size = scores.shape[0]
        current_position = input_ids.shape[1]
        
        # Only apply masking for hold coordinates (positions > 1)
        if current_position <= 1:
            return scores
        
        # Even positions (2, 4, 6, ...): X coordinates
        if current_position % 2 == 0:
            # Mask all X coordinates except valid ones
            mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
            
            # Allow valid X coordinates
            for x_token in self.valid_x_tokens:
                if x_token < scores.shape[1]:
                    mask[x_token] = False
            
            # Also allow EOS (can end path at any point)
            if self.eos_token < scores.shape[1]:
                mask[self.eos_token] = False
            
            # Expand mask for batch
            mask = mask.unsqueeze(0).expand(batch_size, -1)
            scores = scores.masked_fill(mask, -float('inf'))
        
        # Odd positions (3, 5, 7, ...): Y coordinates
        else:
            # Get the X coordinate from previous token
            for batch_idx in range(batch_size):
                prev_token = input_ids[batch_idx, -1].item()
                
                # Convert token back to X coordinate
                if prev_token >= self.x_coord_start:
                    x_coord = prev_token - self.x_coord_start
                    
                    # Get valid Y coordinates for this X
                    if x_coord in self.valid_y_tokens_by_x:
                        valid_y_tokens = self.valid_y_tokens_by_x[x_coord]
                        
                        # Mask all Y coordinates except valid ones for this X
                        mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                        
                        for y_token in valid_y_tokens:
                            if y_token < scores.shape[1]:
                                mask[y_token] = False
                        
                        # Also allow EOS
                        if self.eos_token < scores.shape[1]:
                            mask[self.eos_token] = False
                        
                        scores[batch_idx] = scores[batch_idx].masked_fill(mask, -float('inf'))
        
        return scores
