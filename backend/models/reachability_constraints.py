"""
Reachability constraints for climbing path generation.

Implements sliding window state tracking to approximate 4-limb body position
and constrains next hold selection based on physical reachability.
"""

import torch
import numpy as np
from typing import List, Tuple, Set


def euclidean_distance(hold1: Tuple[int, int], hold2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two holds"""
    return ((hold1[0] - hold2[0])**2 + (hold1[1] - hold2[1])**2)**0.5


def get_last_n_unique(holds: List[Tuple[int, int]], n: int = 4) -> List[Tuple[int, int]]:
    """
    Get last N unique holds from path.
    This approximates the current body position (4 limbs on 4 holds).
    
    Args:
        holds: List of holds in path
        n: Number of unique holds to return (default 4 for 4 limbs)
        
    Returns:
        List of last N unique holds
    """
    seen = []
    for hold in reversed(holds):
        if hold not in seen:
            seen.append(hold)
        if len(seen) >= n:
            break
    return list(reversed(seen))


def get_reachable_holds(
    current_position: List[Tuple[int, int]],
    all_holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    prefer_upward: bool = True
) -> Set[Tuple[int, int]]:
    """
    Get holds reachable from current body position.
    
    A hold is reachable if it's within max_reach distance from ANY of the
    holds in current_position (approximating that any limb can reach it).
    
    Args:
        current_position: List of holds representing current body position
                         (typically last 4 unique holds)
        all_holds: All available holds on the board
        max_reach: Maximum reachable distance (default 5.0 from data analysis)
        prefer_upward: If True, slightly favor holds above current position
        
    Returns:
        Set of reachable holds
    """
    if not current_position:
        return set(all_holds)
    
    reachable = set()
    max_y = max(h[1] for h in current_position)
    
    for hold in all_holds:
        # Check if hold is within reach of ANY current hold
        for current_hold in current_position:
            dist = euclidean_distance(hold, current_hold)
            
            if dist <= max_reach:
                reachable.add(hold)
                break
    
    return reachable


def create_reachability_mask(
    current_position: List[Tuple[int, int]],
    all_holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create binary mask for reachable holds.
    
    Args:
        current_position: Current body position (last N holds)
        all_holds: All possible holds (in same order as model output)
        max_reach: Maximum reachable distance
        device: Device for tensor
        
    Returns:
        Binary mask tensor (1 = reachable, 0 = unreachable)
    """
    reachable = get_reachable_holds(current_position, all_holds, max_reach)
    
    mask = torch.zeros(len(all_holds), device=device)
    for i, hold in enumerate(all_holds):
        if hold in reachable:
            mask[i] = 1.0
    
    return mask


def apply_reachability_constraint(
    logits: torch.Tensor,
    current_position: List[Tuple[int, int]],
    all_holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    mask_value: float = -1e9
) -> torch.Tensor:
    """
    Apply reachability constraint to logits by masking unreachable holds.
    
    Args:
        logits: Model output logits for next hold prediction
        current_position: Current body position (last N holds)
        all_holds: All possible holds (must match logits order)
        max_reach: Maximum reachable distance
        mask_value: Value to set for unreachable holds (large negative)
        
    Returns:
        Masked logits with unreachable holds set to mask_value
    """
    mask = create_reachability_mask(
        current_position, 
        all_holds, 
        max_reach, 
        device=logits.device
    )
    
    # Set unreachable holds to very low probability
    masked_logits = logits.clone()
    masked_logits[mask == 0] = mask_value
    
    return masked_logits


def get_directional_reachability_mask(
    current_position: List[Tuple[int, int]],
    all_holds: List[Tuple[int, int]],
    max_horizontal: float = 5.0,
    max_vertical: float = 4.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create reachability mask with separate horizontal and vertical constraints.
    
    This is more realistic than circular reach - climbers can reach further
    horizontally than vertically in many positions.
    
    Args:
        current_position: Current body position
        all_holds: All possible holds
        max_horizontal: Maximum horizontal reach
        max_vertical: Maximum vertical reach
        device: Device for tensor
        
    Returns:
        Binary mask tensor
    """
    if not current_position:
        return torch.ones(len(all_holds), device=device)
    
    mask = torch.zeros(len(all_holds), device=device)
    
    for i, hold in enumerate(all_holds):
        for current_hold in current_position:
            x_dist = abs(hold[0] - current_hold[0])
            y_dist = abs(hold[1] - current_hold[1])
            
            if x_dist <= max_horizontal and y_dist <= max_vertical:
                mask[i] = 1.0
                break
    
    return mask


def get_progressive_reachability_mask(
    current_position: List[Tuple[int, int]],
    all_holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    upward_bonus: float = 0.5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create reachability mask that prefers upward movement.
    
    Instead of binary mask, returns soft weights that favor holds above
    current position while still allowing downward/sideways moves.
    
    Args:
        current_position: Current body position
        all_holds: All possible holds
        max_reach: Maximum reachable distance
        upward_bonus: Extra reach allowed for upward moves
        device: Device for tensor
        
    Returns:
        Soft mask tensor (values between 0 and 1)
    """
    if not current_position:
        return torch.ones(len(all_holds), device=device)
    
    mask = torch.zeros(len(all_holds), device=device)
    max_y = max(h[1] for h in current_position)
    
    for i, hold in enumerate(all_holds):
        min_dist = float('inf')
        
        for current_hold in current_position:
            dist = euclidean_distance(hold, current_hold)
            
            # Give bonus reach for upward moves
            if hold[1] > current_hold[1]:
                effective_reach = max_reach + upward_bonus
            else:
                effective_reach = max_reach
            
            if dist <= effective_reach:
                min_dist = min(min_dist, dist)
        
        if min_dist <= max_reach + upward_bonus:
            # Soft mask: closer holds get higher weight
            mask[i] = 1.0 - (min_dist / (max_reach + upward_bonus))
    
    return mask


def validate_path_reachability(
    path: List[Tuple[int, int]],
    max_reach: float = 5.0,
    window_size: int = 4
) -> Tuple[bool, List[int]]:
    """
    Validate that a path satisfies reachability constraints.
    
    Args:
        path: List of holds in sequence
        max_reach: Maximum allowed reach distance
        window_size: Size of sliding window for body position
        
    Returns:
        (is_valid, list of invalid indices)
    """
    invalid_indices = []
    
    for i in range(window_size, len(path)):
        current_position = get_last_n_unique(path[:i], window_size)
        next_hold = path[i]
        
        # Check if next hold is reachable from any hold in current position
        is_reachable = False
        for current_hold in current_position:
            if euclidean_distance(next_hold, current_hold) <= max_reach:
                is_reachable = True
                break
        
        if not is_reachable:
            invalid_indices.append(i)
    
    return len(invalid_indices) == 0, invalid_indices


def analyze_path_reachability(path: List[Tuple[int, int]], window_size: int = 4):
    """
    Analyze reachability statistics for a path.
    
    Args:
        path: List of holds in sequence
        window_size: Size of sliding window
        
    Returns:
        Dictionary with reachability statistics
    """
    if len(path) < window_size + 1:
        return {
            'mean_reach': 0,
            'max_reach': 0,
            'min_reach': 0,
            'unreachable_count': 0
        }
    
    reaches = []
    
    for i in range(window_size, len(path)):
        current_position = get_last_n_unique(path[:i], window_size)
        next_hold = path[i]
        
        # Find minimum distance to any hold in current position
        min_dist = min(
            euclidean_distance(next_hold, h) for h in current_position
        )
        reaches.append(min_dist)
    
    return {
        'mean_reach': np.mean(reaches),
        'max_reach': np.max(reaches),
        'min_reach': np.min(reaches),
        'median_reach': np.median(reaches),
        'unreachable_count': sum(1 for r in reaches if r > 5.0)
    }


if __name__ == '__main__':
    # Test reachability constraints
    print("Testing reachability constraints...")
    
    # Example path
    path = [[5, 4], [6, 7], [4, 9], [7, 12], [5, 14], [6, 17]]
    
    # Analyze reachability
    stats = analyze_path_reachability(path, window_size=4)
    print(f"\nPath reachability statistics:")
    print(f"  Mean reach: {stats['mean_reach']:.2f}")
    print(f"  Max reach: {stats['max_reach']:.2f}")
    print(f"  Median reach: {stats['median_reach']:.2f}")
    print(f"  Unreachable moves: {stats['unreachable_count']}")
    
    # Validate path
    is_valid, invalid = validate_path_reachability(path, max_reach=5.0)
    print(f"\nPath valid: {is_valid}")
    if not is_valid:
        print(f"Invalid indices: {invalid}")
    
    # Test reachability from a position
    current_pos = [[5, 4], [6, 7], [4, 9], [7, 12]]
    all_holds = [[x, y] for x in range(11) for y in range(1, 18)]
    
    reachable = get_reachable_holds(current_pos, all_holds, max_reach=5.0)
    print(f"\nFrom position {current_pos}:")
    print(f"  Reachable holds: {len(reachable)} / {len(all_holds)}")
    print(f"  Reachability: {len(reachable)/len(all_holds)*100:.1f}%")
