"""
Reorder climbing paths based on reachability constraints.

Instead of just sorting by Y-coordinate, this creates a more realistic
climbing sequence by considering which holds are actually reachable from
the current body position.
"""

import pandas as pd
import ast
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set


def euclidean_distance(hold1: Tuple[int, int], hold2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two holds"""
    return ((hold1[0] - hold2[0])**2 + (hold1[1] - hold2[1])**2)**0.5


def find_reachable_holds(
    current_position: List[Tuple[int, int]],
    remaining_holds: Set[Tuple[int, int]],
    max_reach: float = 5.0
) -> List[Tuple[int, int]]:
    """
    Find holds reachable from current body position.
    
    Args:
        current_position: List of holds representing current body position
        remaining_holds: Set of holds not yet used
        max_reach: Maximum reachable distance
        
    Returns:
        List of reachable holds sorted by preference (upward, closer)
    """
    if not remaining_holds:
        return []
    
    reachable = []
    max_y = max(h[1] for h in current_position) if current_position else 0
    
    for hold in remaining_holds:
        # Check if reachable from any current hold
        for current_hold in current_position:
            dist = euclidean_distance(hold, current_hold)
            if dist <= max_reach:
                # Score: prefer upward movement and closer holds
                upward_bonus = 2.0 if hold[1] > max_y else 0.0
                score = upward_bonus - dist  # Higher score = better
                reachable.append((hold, score))
                break
    
    # Sort by score (descending)
    reachable.sort(key=lambda x: x[1], reverse=True)
    return [hold for hold, score in reachable]


def reorder_path_by_reachability(
    holds: List[Tuple[int, int]],
    max_reach: float = 5.0,
    window_size: int = 4
) -> List[Tuple[int, int]]:
    """
    Reorder a path to satisfy reachability constraints while maintaining
    the same set of holds.
    
    Algorithm:
    1. Start with holds at lowest Y-coordinates (start holds)
    2. Track last N holds as current body position
    3. From remaining holds, pick the best reachable hold (prefer upward)
    4. Repeat until all holds are placed
    
    Args:
        holds: List of holds to reorder
        max_reach: Maximum reachable distance
        window_size: Number of holds to track as body position
        
    Returns:
        Reordered list of holds
    """
    if len(holds) <= 2:
        return holds
    
    # Convert to tuples for set operations
    holds_tuples = [tuple(h) for h in holds]
    remaining = set(holds_tuples)
    ordered = []
    
    # Start with the lowest holds (typically start holds)
    min_y = min(h[1] for h in holds_tuples)
    start_holds = [h for h in holds_tuples if h[1] == min_y]
    
    # If multiple start holds, sort by X
    start_holds.sort(key=lambda h: h[0])
    
    # Add start holds to ordered path
    for hold in start_holds[:min(2, len(start_holds))]:  # Take up to 2 start holds
        ordered.append(hold)
        remaining.remove(hold)
    
    # Greedy algorithm: always pick best reachable hold
    while remaining:
        # Current body position = last N holds
        current_position = ordered[-window_size:] if len(ordered) >= window_size else ordered
        
        # Find reachable holds
        reachable = find_reachable_holds(current_position, remaining, max_reach)
        
        if reachable:
            # Pick the best reachable hold
            next_hold = reachable[0]
            ordered.append(next_hold)
            remaining.remove(next_hold)
        else:
            # No reachable holds - this shouldn't happen with proper max_reach
            # Fall back to closest hold by Euclidean distance
            closest = min(remaining, key=lambda h: min(
                euclidean_distance(h, c) for c in current_position
            ))
            ordered.append(closest)
            remaining.remove(closest)
    
    return [list(h) for h in ordered]


def reorder_path_with_start_end_preserved(
    start_holds: List[Tuple[int, int]],
    mid_holds: List[Tuple[int, int]],
    end_holds: List[Tuple[int, int]],
    max_reach: float = 5.0
) -> List[Tuple[int, int]]:
    """
    Reorder path while preserving start and end holds.
    
    Args:
        start_holds: Must be at the beginning
        mid_holds: Can be reordered
        end_holds: Must be at the end
        max_reach: Maximum reachable distance
        
    Returns:
        Reordered full path
    """
    # Convert to tuples
    start_tuples = [tuple(h) for h in start_holds]
    mid_tuples = [tuple(h) for h in mid_holds]
    end_tuples = [tuple(h) for h in end_holds]
    
    # Start with start holds
    ordered = list(start_tuples)
    remaining = set(mid_tuples)
    
    # Reorder mid holds
    while remaining:
        current_position = ordered[-4:] if len(ordered) >= 4 else ordered
        reachable = find_reachable_holds(current_position, remaining, max_reach)
        
        if reachable:
            next_hold = reachable[0]
            ordered.append(next_hold)
            remaining.remove(next_hold)
        else:
            # Fall back to closest
            closest = min(remaining, key=lambda h: min(
                euclidean_distance(h, c) for c in current_position
            ))
            ordered.append(closest)
            remaining.remove(closest)
    
    # Add end holds
    ordered.extend(end_tuples)
    
    return [list(h) for h in ordered]


def reorder_csv_paths(input_csv: str, output_csv: str = None, max_reach: float = 5.0):
    """
    Reorder all paths in a CSV file based on reachability.
    
    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path (overwrites input if None)
        max_reach: Maximum reachable distance
    """
    if output_csv is None:
        output_csv = input_csv
    
    print(f'\nProcessing: {input_csv}')
    df = pd.read_csv(input_csv)
    
    # Show example before
    if len(df) > 0:
        example_before = ast.literal_eval(df.iloc[0]['full_path'])
        print(f'   Example before: {example_before[:5]}...')
    
    reordered_count = 0
    unchanged_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse holds
            full_path = ast.literal_eval(row['full_path'])
            
            # Check if we have start/mid/end columns
            if 'start_holds' in df.columns and 'mid_holds' in df.columns and 'end_holds' in df.columns:
                start_holds = ast.literal_eval(row['start_holds'])
                mid_holds = ast.literal_eval(row['mid_holds'])
                end_holds = ast.literal_eval(row['end_holds'])
                
                # Reorder with start/end preserved
                reordered = reorder_path_with_start_end_preserved(
                    start_holds, mid_holds, end_holds, max_reach
                )
            else:
                # Reorder entire path
                reordered = reorder_path_by_reachability(full_path, max_reach)
            
            # Update dataframe
            df.at[idx, 'full_path'] = str(reordered)
            
            if reordered != full_path:
                reordered_count += 1
            else:
                unchanged_count += 1
                
        except Exception as e:
            print(f'   Warning: Failed to reorder path at index {idx}: {e}')
            continue
    
    # Show example after
    if len(df) > 0:
        example_after = ast.literal_eval(df.iloc[0]['full_path'])
        print(f'   Example after:  {example_after[:5]}...')
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f'   Reordered: {reordered_count} paths')
    print(f'   Unchanged: {unchanged_count} paths')
    print(f'   Saved to: {output_csv}')


def analyze_reordering_impact(csv_path: str, max_reach: float = 5.0):
    """
    Analyze how reordering affects path statistics.
    """
    print(f'\nAnalyzing reordering impact: {csv_path}')
    df = pd.read_csv(csv_path)
    
    original_violations = 0
    reordered_violations = 0
    total_moves = 0
    
    for idx, row in df.iterrows():
        try:
            full_path = ast.literal_eval(row['full_path'])
            
            # Check original path
            for i in range(4, len(full_path)):
                window = full_path[i-4:i]
                next_hold = full_path[i]
                
                min_dist = min(euclidean_distance(next_hold, h) for h in window)
                if min_dist > max_reach:
                    original_violations += 1
                total_moves += 1
            
            # Reorder and check
            if 'start_holds' in df.columns:
                start = ast.literal_eval(row['start_holds'])
                mid = ast.literal_eval(row['mid_holds'])
                end = ast.literal_eval(row['end_holds'])
                reordered = reorder_path_with_start_end_preserved(start, mid, end, max_reach)
            else:
                reordered = reorder_path_by_reachability(full_path, max_reach)
            
            # Check reordered path
            for i in range(4, len(reordered)):
                window = reordered[i-4:i]
                next_hold = reordered[i]
                
                min_dist = min(euclidean_distance(tuple(next_hold), tuple(h)) for h in window)
                if min_dist > max_reach:
                    reordered_violations += 1
                    
        except:
            continue
    
    print(f'\n   Original path violations: {original_violations} / {total_moves} ({original_violations/total_moves*100:.1f}%)')
    print(f'   Reordered path violations: {reordered_violations} / {total_moves} ({reordered_violations/total_moves*100:.1f}%)')
    print(f'   Improvement: {original_violations - reordered_violations} fewer violations')


if __name__ == '__main__':
    print('=' * 70)
    print('REORDER PATHS BY REACHABILITY CONSTRAINTS')
    print('=' * 70)
    
    data_dir = Path('data')
    max_reach = 5.0
    
    # List of files to reorder
    files_to_reorder = [
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
    
    # First, analyze impact on one file
    test_file = data_dir / 'moonboard_train_quality.csv'
    if test_file.exists():
        print('\n' + '=' * 70)
        print('ANALYZING IMPACT (before reordering)')
        print('=' * 70)
        analyze_reordering_impact(str(test_file), max_reach)
    
    # Ask for confirmation
    print('\n' + '=' * 70)
    print('READY TO REORDER ALL FILES')
    print('=' * 70)
    print(f'Max reach: {max_reach} units')
    print(f'Files to process: {len([f for f in files_to_reorder if (data_dir / f).exists()])}')
    
    response = input('\nProceed with reordering? (yes/no): ')
    
    if response.lower() == 'yes':
        print('\nReordering files...')
        
        reordered_count = 0
        skipped_count = 0
        
        for filename in files_to_reorder:
            filepath = data_dir / filename
            if filepath.exists():
                reorder_csv_paths(str(filepath), max_reach=max_reach)
                reordered_count += 1
            else:
                print(f'\nSkipping: {filename} (not found)')
                skipped_count += 1
        
        print('\n' + '=' * 70)
        print(f'COMPLETE! Reordered {reordered_count} files, skipped {skipped_count} files')
        print('=' * 70)
        
        # Analyze impact after reordering
        if test_file.exists():
            print('\n' + '=' * 70)
            print('ANALYZING IMPACT (after reordering)')
            print('=' * 70)
            analyze_reordering_impact(str(test_file), max_reach)
    else:
        print('\nReordering cancelled.')
