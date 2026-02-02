"""
Smart path reordering that minimizes reachability violations.

Uses a more sophisticated algorithm than simple greedy selection.
"""

import pandas as pd
import ast
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set, Optional
import itertools


def euclidean_distance(hold1: Tuple[int, int], hold2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two holds"""
    return ((hold1[0] - hold2[0])**2 + (hold1[1] - hold2[1])**2)**0.5


def count_violations(path: List[Tuple[int, int]], max_reach: float = 5.0, window_size: int = 4) -> int:
    """Count number of reachability violations in a path"""
    violations = 0
    for i in range(window_size, len(path)):
        window = path[i-window_size:i]
        next_hold = path[i]
        min_dist = min(euclidean_distance(next_hold, h) for h in window)
        if min_dist > max_reach:
            violations += 1
    return violations


def beam_search_reorder(
    holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    window_size: int = 4,
    beam_width: int = 3,
    max_depth: int = 5
) -> List[Tuple[int, int]]:
    """
    Use beam search to find better path ordering.
    
    Args:
        holds: List of holds to reorder
        max_reach: Maximum reachable distance
        window_size: Number of holds for body position
        beam_width: Number of candidates to keep at each step
        max_depth: How far ahead to look
        
    Returns:
        Reordered path
    """
    if len(holds) <= window_size:
        # For short paths, just sort by Y then X
        return sorted(holds, key=lambda h: (h[1], h[0]))
    
    holds_tuples = [tuple(h) for h in holds]
    
    # Start with lowest holds
    min_y = min(h[1] for h in holds_tuples)
    start_candidates = [h for h in holds_tuples if h[1] == min_y]
    start_candidates.sort(key=lambda h: h[0])
    
    # Initialize with start holds
    ordered = list(start_candidates[:min(2, len(start_candidates))])
    remaining = set(holds_tuples) - set(ordered)
    
    # Greedy with lookahead
    while remaining:
        current_position = ordered[-window_size:] if len(ordered) >= window_size else ordered
        
        # Find reachable holds
        reachable = []
        for hold in remaining:
            min_dist = min(euclidean_distance(hold, c) for c in current_position)
            if min_dist <= max_reach:
                # Score based on: upward progress, distance, future reachability
                max_y = max(c[1] for c in current_position)
                upward_score = (hold[1] - max_y) * 2.0  # Prefer upward
                distance_score = -min_dist  # Prefer closer
                
                # Lookahead: how many holds would be reachable after this move?
                future_position = (current_position + [hold])[-window_size:]
                future_reachable = sum(
                    1 for h in remaining if h != hold and
                    any(euclidean_distance(h, fp) <= max_reach for fp in future_position)
                )
                future_score = future_reachable * 0.5
                
                total_score = upward_score + distance_score + future_score
                reachable.append((hold, total_score))
        
        if reachable:
            # Pick best scoring hold
            reachable.sort(key=lambda x: x[1], reverse=True)
            next_hold = reachable[0][0]
        else:
            # No reachable holds - pick closest
            next_hold = min(remaining, key=lambda h: min(
                euclidean_distance(h, c) for c in current_position
            ))
        
        ordered.append(next_hold)
        remaining.remove(next_hold)
    
    return [list(h) for h in ordered]


def optimize_path_order(
    holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    window_size: int = 4
) -> List[Tuple[int, int]]:
    """
    Optimize path order to minimize violations.
    
    Strategy:
    1. Start with Y-sorted path
    2. Use beam search for better ordering
    3. Return whichever has fewer violations
    """
    holds_tuples = [tuple(h) for h in holds]
    
    # Option 1: Y-sorted (current approach)
    y_sorted = sorted(holds_tuples, key=lambda h: (h[1], h[0]))
    y_violations = count_violations(y_sorted, max_reach, window_size)
    
    # Option 2: Beam search
    beam_ordered = beam_search_reorder(holds, max_reach, window_size)
    beam_ordered_tuples = [tuple(h) for h in beam_ordered]
    beam_violations = count_violations(beam_ordered_tuples, max_reach, window_size)
    
    # Return better option
    if beam_violations <= y_violations:
        return beam_ordered
    else:
        return [list(h) for h in y_sorted]


def reorder_path_with_start_end(
    start_holds: List[Tuple[int, int]],
    mid_holds: List[Tuple[int, int]],
    end_holds: List[Tuple[int, int]],
    max_reach: float = 5.0
) -> List[Tuple[int, int]]:
    """
    Reorder path preserving start and end holds.
    """
    start_tuples = [tuple(h) for h in start_holds]
    mid_tuples = [tuple(h) for h in mid_holds]
    end_tuples = [tuple(h) for h in end_holds]
    
    # Start with start holds
    ordered = list(start_tuples)
    remaining = set(mid_tuples)
    
    # Reorder mid holds with lookahead
    while remaining:
        current_position = ordered[-4:] if len(ordered) >= 4 else ordered
        
        # Find best next hold
        best_hold = None
        best_score = float('-inf')
        
        for hold in remaining:
            min_dist = min(euclidean_distance(hold, c) for c in current_position)
            
            if min_dist <= max_reach:
                # Score this hold
                max_y = max(c[1] for c in current_position)
                upward = (hold[1] - max_y) * 2.0
                distance = -min_dist
                
                # Lookahead
                future_pos = (current_position + [hold])[-4:]
                future_reachable = sum(
                    1 for h in remaining if h != hold and
                    any(euclidean_distance(h, fp) <= max_reach for fp in future_pos)
                )
                
                score = upward + distance + future_reachable * 0.5
                
                if score > best_score:
                    best_score = score
                    best_hold = hold
        
        if best_hold is None:
            # Fall back to closest
            best_hold = min(remaining, key=lambda h: min(
                euclidean_distance(h, c) for c in current_position
            ))
        
        ordered.append(best_hold)
        remaining.remove(best_hold)
    
    # Add end holds
    ordered.extend(end_tuples)
    
    return [list(h) for h in ordered]


def reorder_csv_paths(input_csv: str, output_csv: str = None, max_reach: float = 5.0):
    """Reorder all paths in CSV file"""
    if output_csv is None:
        output_csv = input_csv
    
    print(f'\nProcessing: {input_csv}')
    df = pd.read_csv(input_csv)
    
    improved_count = 0
    unchanged_count = 0
    worsened_count = 0
    
    for idx, row in df.iterrows():
        try:
            full_path = ast.literal_eval(row['full_path'])
            original_violations = count_violations([tuple(h) for h in full_path], max_reach)
            
            # Reorder
            if 'start_holds' in df.columns:
                start = ast.literal_eval(row['start_holds'])
                mid = ast.literal_eval(row['mid_holds'])
                end = ast.literal_eval(row['end_holds'])
                reordered = reorder_path_with_start_end(start, mid, end, max_reach)
            else:
                reordered = optimize_path_order(full_path, max_reach)
            
            new_violations = count_violations([tuple(h) for h in reordered], max_reach)
            
            # Only update if improved or same
            if new_violations <= original_violations:
                df.at[idx, 'full_path'] = str(reordered)
                if new_violations < original_violations:
                    improved_count += 1
                else:
                    unchanged_count += 1
            else:
                worsened_count += 1
                
        except Exception as e:
            print(f'   Warning: Failed at index {idx}: {e}')
            continue
    
    df.to_csv(output_csv, index=False)
    print(f'   Improved: {improved_count} paths')
    print(f'   Unchanged: {unchanged_count} paths')
    print(f'   Kept original (would worsen): {worsened_count} paths')
    print(f'   Saved to: {output_csv}')


def analyze_dataset(csv_path: str, max_reach: float = 5.0):
    """Analyze reachability violations in dataset"""
    print(f'\nAnalyzing: {csv_path}')
    df = pd.read_csv(csv_path)
    
    total_violations = 0
    total_moves = 0
    paths_with_violations = 0
    
    for idx, row in df.iterrows():
        try:
            full_path = ast.literal_eval(row['full_path'])
            violations = count_violations([tuple(h) for h in full_path], max_reach)
            
            if violations > 0:
                paths_with_violations += 1
                total_violations += violations
            
            if len(full_path) > 4:
                total_moves += len(full_path) - 4
                
        except:
            continue
    
    print(f'   Total violations: {total_violations} / {total_moves} moves ({total_violations/total_moves*100:.2f}%)')
    print(f'   Paths with violations: {paths_with_violations} / {len(df)} ({paths_with_violations/len(df)*100:.2f}%)')


if __name__ == '__main__':
    print('=' * 70)
    print('SMART PATH REORDERING WITH REACHABILITY OPTIMIZATION')
    print('=' * 70)
    
    data_dir = Path('data')
    max_reach = 5.0
    
    # Analyze current state
    test_file = data_dir / 'moonboard_train_quality.csv'
    if test_file.exists():
        print('\nCURRENT STATE:')
        analyze_dataset(str(test_file), max_reach)
    
    print('\n' + '=' * 70)
    print('This will reorder paths to minimize reachability violations.')
    print('Only paths that improve will be changed.')
    print('=' * 70)
    
    files_to_process = [
        'moonboard_cleaned.csv',
        'moonboard_train.csv',
        'moonboard_val.csv',
        'moonboard_test.csv',
        'moonboard_train_quality.csv',
        'moonboard_val_quality.csv',
        'moonboard_test_quality.csv',
        'moonboard_train_benchmark.csv',
        'moonboard_val_benchmark.csv',
        'moonboard_test_benchmark.csv',
    ]
    
    print(f'\nProcessing {len([f for f in files_to_process if (data_dir / f).exists()])} files...\n')
    
    for filename in files_to_process:
        filepath = data_dir / filename
        if filepath.exists():
            reorder_csv_paths(str(filepath), max_reach=max_reach)
    
    # Analyze after
    if test_file.exists():
        print('\n' + '=' * 70)
        print('AFTER REORDERING:')
        analyze_dataset(str(test_file), max_reach)
    
    print('\n' + '=' * 70)
    print('COMPLETE!')
    print('=' * 70)
