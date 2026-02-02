"""
Reachability loss for training climbing path generation models.

Penalizes sequences where holds are too far apart to reach.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ReachabilityLoss(nn.Module):
    """
    Loss function that penalizes unreachable hold transitions.
    
    Uses sliding window to approximate body position and penalizes
    holds that are too far from any recent hold.
    """
    
    def __init__(
        self,
        tokenizer,
        max_reach: float = 5.0,
        window_size: int = 4,
        penalty_scale: float = 1.0,
    ):
        """
        Args:
            tokenizer: ClimbPathTokenizer instance
            max_reach: Maximum reachable distance (from data analysis)
            window_size: Number of recent holds to consider as body position
            penalty_scale: Scaling factor for penalty (higher = stronger penalty)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_reach = max_reach
        self.window_size = window_size
        self.penalty_scale = penalty_scale
    
    def euclidean_distance(self, x1: torch.Tensor, y1: torch.Tensor, 
                          x2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Calculate Euclidean distance between two holds"""
        return torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute reachability loss for a batch of sequences.
        
        Args:
            input_ids: Token sequence [batch_size, seq_len]
                      Format: [BOS, grade, x1, y1, x2, y2, ..., EOS]
        
        Returns:
            Reachability loss (scalar)
        """
        batch_size, seq_len = input_ids.shape
        
        # Extract X and Y coordinate tokens
        # Positions: 0=BOS, 1=grade, 2=x1, 3=y1, 4=x2, 5=y2, ...
        x_positions = list(range(2, seq_len, 2))  # Even positions after grade
        y_positions = list(range(3, seq_len, 2))  # Odd positions after grade
        
        if len(x_positions) < self.window_size + 1:
            # Not enough holds to compute reachability
            return torch.tensor(0.0, device=input_ids.device)
        
        total_penalty = 0.0
        num_valid_sequences = 0
        
        for b in range(batch_size):
            # Get X and Y tokens
            x_tokens = input_ids[b, x_positions]
            y_tokens = input_ids[b, y_positions]
            
            # Filter out padding and invalid tokens
            # Valid X tokens: [16, 26] (X coords 0-10)
            # Valid Y tokens: [27, 44] (Y coords 1-18)
            valid_mask = (
                (x_tokens >= self.tokenizer.X_COORD_START) & 
                (x_tokens < self.tokenizer.Y_COORD_START) &
                (y_tokens >= self.tokenizer.Y_COORD_START) & 
                (y_tokens < self.tokenizer.EOS_TOKEN)
            )
            
            valid_x = x_tokens[valid_mask]
            valid_y = y_tokens[valid_mask]
            
            if len(valid_x) < self.window_size + 1:
                continue
            
            # Convert tokens to actual coordinates
            x_coords = valid_x - self.tokenizer.X_COORD_START  # 0-10
            y_coords = valid_y - self.tokenizer.Y_COORD_START + 1  # 1-18
            
            # For each hold after the window, check if it's reachable
            sequence_penalty = 0.0
            num_checks = 0
            
            for i in range(self.window_size, len(x_coords)):
                # Current body position = last window_size holds
                window_x = x_coords[i - self.window_size:i]
                window_y = y_coords[i - self.window_size:i]
                
                # Next hold
                next_x = x_coords[i]
                next_y = y_coords[i]
                
                # Calculate minimum distance from next hold to any hold in window
                distances = self.euclidean_distance(
                    next_x.float(), next_y.float(),
                    window_x.float(), window_y.float()
                )
                min_distance = torch.min(distances)
                
                # Penalty if distance exceeds max_reach
                # Use smooth penalty (ReLU) instead of hard threshold
                violation = torch.clamp(min_distance - self.max_reach, min=0.0)
                sequence_penalty += violation
                num_checks += 1
            
            if num_checks > 0:
                total_penalty += sequence_penalty / num_checks
                num_valid_sequences += 1
        
        if num_valid_sequences == 0:
            return torch.tensor(0.0, device=input_ids.device)
        
        # Average penalty across batch, scaled
        return self.penalty_scale * (total_penalty / num_valid_sequences)


class SoftReachabilityLoss(nn.Module):
    """
    Soft reachability loss with exponential penalty.
    
    Instead of linear penalty, uses exponential to more heavily penalize
    large violations while being lenient on small ones.
    """
    
    def __init__(
        self,
        tokenizer,
        max_reach: float = 5.0,
        window_size: int = 4,
        penalty_scale: float = 0.5,
        sharpness: float = 0.5,
    ):
        """
        Args:
            tokenizer: ClimbPathTokenizer instance
            max_reach: Maximum reachable distance
            window_size: Number of recent holds for body position
            penalty_scale: Scaling factor for penalty
            sharpness: Controls how sharply penalty increases (higher = sharper)
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_reach = max_reach
        self.window_size = window_size
        self.penalty_scale = penalty_scale
        self.sharpness = sharpness
    
    def euclidean_distance(self, x1: torch.Tensor, y1: torch.Tensor,
                          x2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Calculate Euclidean distance"""
        return torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute soft reachability loss"""
        batch_size, seq_len = input_ids.shape
        
        x_positions = list(range(2, seq_len, 2))
        y_positions = list(range(3, seq_len, 2))
        
        if len(x_positions) < self.window_size + 1:
            return torch.tensor(0.0, device=input_ids.device)
        
        total_penalty = 0.0
        num_valid_sequences = 0
        
        for b in range(batch_size):
            x_tokens = input_ids[b, x_positions]
            y_tokens = input_ids[b, y_positions]
            
            valid_mask = (
                (x_tokens >= self.tokenizer.X_COORD_START) &
                (x_tokens < self.tokenizer.Y_COORD_START) &
                (y_tokens >= self.tokenizer.Y_COORD_START) &
                (y_tokens < self.tokenizer.EOS_TOKEN)
            )
            
            valid_x = x_tokens[valid_mask]
            valid_y = y_tokens[valid_mask]
            
            if len(valid_x) < self.window_size + 1:
                continue
            
            x_coords = valid_x - self.tokenizer.X_COORD_START
            y_coords = valid_y - self.tokenizer.Y_COORD_START + 1
            
            sequence_penalty = 0.0
            num_checks = 0
            
            for i in range(self.window_size, len(x_coords)):
                window_x = x_coords[i - self.window_size:i]
                window_y = y_coords[i - self.window_size:i]
                next_x = x_coords[i]
                next_y = y_coords[i]
                
                distances = self.euclidean_distance(
                    next_x.float(), next_y.float(),
                    window_x.float(), window_y.float()
                )
                min_distance = torch.min(distances)
                
                # Exponential penalty: exp(sharpness * (distance - max_reach)) - 1
                # This is 0 when distance = max_reach, grows exponentially beyond
                violation = min_distance - self.max_reach
                if violation > 0:
                    penalty = torch.exp(self.sharpness * violation) - 1.0
                    sequence_penalty += penalty
                
                num_checks += 1
            
            if num_checks > 0:
                total_penalty += sequence_penalty / num_checks
                num_valid_sequences += 1
        
        if num_valid_sequences == 0:
            return torch.tensor(0.0, device=input_ids.device)
        
        return self.penalty_scale * (total_penalty / num_valid_sequences)


class AdaptiveReachabilityLoss(nn.Module):
    """
    Adaptive reachability loss that adjusts based on path progress.
    
    Allows larger reaches early in the path (setting up) and enforces
    tighter constraints later (finishing moves).
    """
    
    def __init__(
        self,
        tokenizer,
        initial_reach: float = 6.0,
        final_reach: float = 4.5,
        window_size: int = 4,
        penalty_scale: float = 0.5,
    ):
        """
        Args:
            tokenizer: ClimbPathTokenizer instance
            initial_reach: Max reach at start of path
            final_reach: Max reach near end of path
            window_size: Number of recent holds for body position
            penalty_scale: Scaling factor for penalty
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.initial_reach = initial_reach
        self.final_reach = final_reach
        self.window_size = window_size
        self.penalty_scale = penalty_scale
    
    def euclidean_distance(self, x1: torch.Tensor, y1: torch.Tensor,
                          x2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Calculate Euclidean distance"""
        return torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute adaptive reachability loss"""
        batch_size, seq_len = input_ids.shape
        
        x_positions = list(range(2, seq_len, 2))
        y_positions = list(range(3, seq_len, 2))
        
        if len(x_positions) < self.window_size + 1:
            return torch.tensor(0.0, device=input_ids.device)
        
        total_penalty = 0.0
        num_valid_sequences = 0
        
        for b in range(batch_size):
            x_tokens = input_ids[b, x_positions]
            y_tokens = input_ids[b, y_positions]
            
            valid_mask = (
                (x_tokens >= self.tokenizer.X_COORD_START) &
                (x_tokens < self.tokenizer.Y_COORD_START) &
                (y_tokens >= self.tokenizer.Y_COORD_START) &
                (y_tokens < self.tokenizer.EOS_TOKEN)
            )
            
            valid_x = x_tokens[valid_mask]
            valid_y = y_tokens[valid_mask]
            
            if len(valid_x) < self.window_size + 1:
                continue
            
            x_coords = valid_x - self.tokenizer.X_COORD_START
            y_coords = valid_y - self.tokenizer.Y_COORD_START + 1
            
            sequence_penalty = 0.0
            num_checks = 0
            
            for i in range(self.window_size, len(x_coords)):
                # Calculate progress (0 to 1 based on Y-coordinate)
                max_y = torch.max(y_coords[:i])
                progress = (max_y.float() - 1.0) / 17.0  # Normalize to [0, 1]
                
                # Interpolate max_reach based on progress
                current_max_reach = (
                    self.initial_reach * (1.0 - progress) +
                    self.final_reach * progress
                )
                
                window_x = x_coords[i - self.window_size:i]
                window_y = y_coords[i - self.window_size:i]
                next_x = x_coords[i]
                next_y = y_coords[i]
                
                distances = self.euclidean_distance(
                    next_x.float(), next_y.float(),
                    window_x.float(), window_y.float()
                )
                min_distance = torch.min(distances)
                
                # Penalty if exceeds adaptive threshold
                violation = torch.clamp(min_distance - current_max_reach, min=0.0)
                sequence_penalty += violation
                num_checks += 1
            
            if num_checks > 0:
                total_penalty += sequence_penalty / num_checks
                num_valid_sequences += 1
        
        if num_valid_sequences == 0:
            return torch.tensor(0.0, device=input_ids.device)
        
        return self.penalty_scale * (total_penalty / num_valid_sequences)
