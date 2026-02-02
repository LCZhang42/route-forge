"""
Logits processor for reachability constraints during path generation.
"""

import torch
from typing import List, Tuple
from .reachability_constraints import (
    get_last_n_unique,
    apply_reachability_constraint,
    get_directional_reachability_mask,
)


class ReachabilityLogitsProcessor:
    """
    Logits processor that enforces physical reachability constraints.
    
    Uses sliding window to approximate 4-limb body position and masks
    holds that are too far to reach.
    """
    
    def __init__(
        self,
        tokenizer,
        max_reach: float = 5.0,
        window_size: int = 4,
        use_directional: bool = False,
        max_horizontal: float = 5.0,
        max_vertical: float = 4.0,
    ):
        """
        Args:
            tokenizer: ClimbPathTokenizer instance
            max_reach: Maximum reachable distance (Euclidean)
            window_size: Number of recent holds to consider as body position
            use_directional: Use separate horizontal/vertical constraints
            max_horizontal: Max horizontal reach (if use_directional=True)
            max_vertical: Max vertical reach (if use_directional=True)
        """
        self.tokenizer = tokenizer
        self.max_reach = max_reach
        self.window_size = window_size
        self.use_directional = use_directional
        self.max_horizontal = max_horizontal
        self.max_vertical = max_vertical
        
        # Pre-compute all possible holds
        self.all_holds = [
            (x, y) for x in range(11) for y in range(1, 18)
        ]
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply reachability constraint to logits.
        
        Args:
            input_ids: Generated tokens so far [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
            
        Returns:
            Modified logits with unreachable holds masked
        """
        batch_size = input_ids.shape[0]
        device = scores.device
        
        # Process each sequence in batch
        for i in range(batch_size):
            # Determine what type of token we're generating
            current_pos = input_ids[i].shape[0]
            token_type = self.tokenizer.get_token_type(current_pos)
            
            # Only apply reachability constraints to hold coordinates
            if token_type not in ['x', 'y']:
                continue
            
            # Decode current path
            tokens = input_ids[i].cpu().numpy()
            _, holds = self.tokenizer.decode(tokens)
            
            # Skip if path is too short (need at least some holds for context)
            if len(holds) < 2:
                continue
            
            # Get current body position (last N unique holds)
            current_position = get_last_n_unique(holds, self.window_size)
            
            if not current_position:
                continue
            
            # Create reachability mask
            if self.use_directional:
                # Use separate horizontal/vertical constraints
                mask = get_directional_reachability_mask(
                    current_position,
                    self.all_holds,
                    max_horizontal=self.max_horizontal,
                    max_vertical=self.max_vertical,
                    device=device
                )
            else:
                # Use circular reach constraint
                from .reachability_constraints import create_reachability_mask
                mask = create_reachability_mask(
                    current_position,
                    self.all_holds,
                    max_reach=self.max_reach,
                    device=device
                )
            
            # Map hold mask to token mask based on current position
            # Start with -inf for all tokens (will be masked out)
            token_mask = torch.full((scores.shape[1],), -1e9, device=device)
            
            # Allow EOS token always
            token_mask[self.tokenizer.EOS_TOKEN] = 0.0
            
            for hold_idx, hold in enumerate(self.all_holds):
                if mask[hold_idx] > 0:  # If hold is reachable
                    hold_tokens = self.tokenizer.encode_hold(hold[0], hold[1])
                    # Only unmask the token type we're currently generating
                    if token_type == 'x':
                        token_mask[hold_tokens[0]] = 0.0
                    else:  # token_type == 'y'
                        token_mask[hold_tokens[1]] = 0.0
            
            # Apply mask (add to logits)
            scores[i] = scores[i] + token_mask
        
        return scores


class AdaptiveReachabilityProcessor:
    """
    Adaptive reachability processor that adjusts constraints based on path progress.
    
    Allows larger reaches early in the path (setting up) and tighter constraints
    later (finishing moves).
    """
    
    def __init__(
        self,
        tokenizer,
        initial_reach: float = 6.0,
        final_reach: float = 4.5,
        window_size: int = 4,
    ):
        """
        Args:
            tokenizer: ClimbPathTokenizer instance
            initial_reach: Max reach at start of path
            final_reach: Max reach near end of path
            window_size: Number of recent holds for body position
        """
        self.tokenizer = tokenizer
        self.initial_reach = initial_reach
        self.final_reach = final_reach
        self.window_size = window_size
        
        self.all_holds = [
            (x, y) for x in range(11) for y in range(1, 18)
        ]
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply adaptive reachability constraint"""
        batch_size = input_ids.shape[0]
        device = scores.device
        
        for i in range(batch_size):
            # Determine what type of token we're generating
            current_pos = input_ids[i].shape[0]
            token_type = self.tokenizer.get_token_type(current_pos)
            
            # Only apply reachability constraints to hold coordinates
            if token_type not in ['x', 'y']:
                continue
            
            tokens = input_ids[i].cpu().numpy()
            _, holds = self.tokenizer.decode(tokens)
            
            if len(holds) < 2:
                continue
            
            # Calculate adaptive reach based on path progress
            # Estimate progress by max Y-coordinate reached
            if holds:
                max_y = max(h[1] for h in holds)
                progress = max_y / 17.0  # Normalize to [0, 1]
                
                # Interpolate reach constraint
                current_reach = (
                    self.initial_reach * (1 - progress) + 
                    self.final_reach * progress
                )
            else:
                current_reach = self.initial_reach
            
            # Get current position and create mask
            current_position = get_last_n_unique(holds, self.window_size)
            
            if not current_position:
                continue
            
            from .reachability_constraints import create_reachability_mask
            mask = create_reachability_mask(
                current_position,
                self.all_holds,
                max_reach=current_reach,
                device=device
            )
            
            # Map hold mask to token mask based on current position
            token_mask = torch.full((scores.shape[1],), -1e9, device=device)
            token_mask[self.tokenizer.EOS_TOKEN] = 0.0
            
            for hold_idx, hold in enumerate(self.all_holds):
                if mask[hold_idx] > 0:
                    hold_tokens = self.tokenizer.encode_hold(hold[0], hold[1])
                    if token_type == 'x':
                        token_mask[hold_tokens[0]] = 0.0
                    else:
                        token_mask[hold_tokens[1]] = 0.0
            
            scores[i] = scores[i] + token_mask
        
        return scores


class ProgressiveReachabilityProcessor:
    """
    Reachability processor that prefers upward movement.
    
    Gives bonus reach for holds above current position while still allowing
    sideways and downward moves when necessary.
    """
    
    def __init__(
        self,
        tokenizer,
        max_reach: float = 5.0,
        upward_bonus: float = 0.5,
        window_size: int = 4,
    ):
        """
        Args:
            tokenizer: ClimbPathTokenizer instance
            max_reach: Base maximum reach
            upward_bonus: Extra reach allowed for upward moves
            window_size: Number of recent holds for body position
        """
        self.tokenizer = tokenizer
        self.max_reach = max_reach
        self.upward_bonus = upward_bonus
        self.window_size = window_size
        
        self.all_holds = [
            (x, y) for x in range(11) for y in range(1, 18)
        ]
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply progressive reachability constraint"""
        batch_size = input_ids.shape[0]
        device = scores.device
        
        for i in range(batch_size):
            # Determine what type of token we're generating
            current_pos = input_ids[i].shape[0]
            token_type = self.tokenizer.get_token_type(current_pos)
            
            # Only apply reachability constraints to hold coordinates
            if token_type not in ['x', 'y']:
                continue
            
            tokens = input_ids[i].cpu().numpy()
            _, holds = self.tokenizer.decode(tokens)
            
            if len(holds) < 2:
                continue
            
            current_position = get_last_n_unique(holds, self.window_size)
            
            if not current_position:
                continue
            
            from .reachability_constraints import get_progressive_reachability_mask
            mask = get_progressive_reachability_mask(
                current_position,
                self.all_holds,
                max_reach=self.max_reach,
                upward_bonus=self.upward_bonus,
                device=device
            )
            
            # Map to token mask based on current position
            # Use soft weighting for progressive processor
            token_mask = torch.zeros(scores.shape[1], device=device)
            
            for hold_idx, hold in enumerate(self.all_holds):
                hold_tokens = self.tokenizer.encode_hold(hold[0], hold[1])
                if token_type == 'x':
                    token_mask[hold_tokens[0]] = mask[hold_idx]
                else:
                    token_mask[hold_tokens[1]] = mask[hold_idx]
            
            # Apply soft mask (weights instead of hard cutoff)
            scores[i] = scores[i] * token_mask
        
        return scores
