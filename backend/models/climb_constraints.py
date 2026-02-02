"""
Extract climbing constraints from training data.

Provides common start/end holds and hold count distributions by grade.
"""

import pandas as pd
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import random


def extract_grade_constraints(csv_path: str) -> Dict:
    """
    Extract start holds, end holds, and hold count distributions by grade.
    
    Args:
        csv_path: Path to training CSV
        
    Returns:
        Dictionary with constraints by grade
    """
    df = pd.read_csv(csv_path)
    
    constraints = {}
    
    for grade in df['grade'].unique():
        grade_df = df[df['grade'] == grade]
        
        start_holds = []
        end_holds = []
        hold_counts = []
        
        for path_str in grade_df['full_path']:
            path = ast.literal_eval(path_str)
            if len(path) >= 2:
                start_holds.append(tuple(path[0]))
                end_holds.append(tuple(path[-1]))
                hold_counts.append(len(path))
        
        # Get most common start/end holds (top 20)
        start_counter = Counter(start_holds)
        end_counter = Counter(end_holds)
        
        constraints[grade] = {
            'common_starts': [list(hold) for hold, _ in start_counter.most_common(20)],
            'common_ends': [list(hold) for hold, _ in end_counter.most_common(20)],
            'hold_count_min': min(hold_counts) if hold_counts else 3,
            'hold_count_max': max(hold_counts) if hold_counts else 30,
            'hold_count_mean': sum(hold_counts) / len(hold_counts) if hold_counts else 10,
            'hold_count_std': (sum((x - sum(hold_counts) / len(hold_counts)) ** 2 for x in hold_counts) / len(hold_counts)) ** 0.5 if hold_counts else 3,
        }
    
    return constraints


def get_random_hold_count(grade: str, constraints: Dict) -> int:
    """
    Get a random hold count based on grade distribution.
    
    Args:
        grade: Climbing grade
        constraints: Grade constraints dictionary
        
    Returns:
        Random hold count
    """
    if grade not in constraints:
        return random.randint(5, 15)
    
    grade_data = constraints[grade]
    mean = grade_data['hold_count_mean']
    std = grade_data['hold_count_std']
    
    # Sample from normal distribution and clip to valid range
    count = int(random.gauss(mean, std))
    count = max(grade_data['hold_count_min'], min(grade_data['hold_count_max'], count))
    
    return count


def get_common_start_hold(grade: str, constraints: Dict) -> Tuple[int, int]:
    """
    Get a random common start hold for the grade.
    
    Args:
        grade: Climbing grade
        constraints: Grade constraints dictionary
        
    Returns:
        (x, y) tuple for start hold
    """
    if grade not in constraints or not constraints[grade]['common_starts']:
        # Default start holds (bottom rows)
        return random.choice([[5, 4], [6, 4], [4, 4], [7, 4], [5, 3], [6, 3]])
    
    start_hold = random.choice(constraints[grade]['common_starts'])
    return tuple(start_hold)


def get_common_end_hold(grade: str, constraints: Dict) -> Tuple[int, int]:
    """
    Get a random common end hold for the grade.
    
    Args:
        grade: Climbing grade
        constraints: Grade constraints dictionary
        
    Returns:
        (x, y) tuple for end hold
    """
    if grade not in constraints or not constraints[grade]['common_ends']:
        # Default end holds (top rows)
        return random.choice([[5, 17], [6, 17], [4, 17], [7, 17], [5, 18], [6, 18]])
    
    end_hold = random.choice(constraints[grade]['common_ends'])
    return tuple(end_hold)


def save_constraints(constraints: Dict, output_path: str):
    """Save constraints to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(constraints, f, indent=2)


def load_constraints(input_path: str) -> Dict:
    """Load constraints from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Extract and save constraints
    train_csv = Path('data/moonboard_train_quality.csv')
    
    if train_csv.exists():
        print("Extracting grade constraints from training data...")
        constraints = extract_grade_constraints(str(train_csv))
        
        # Save to file
        output_path = Path('data/grade_constraints.json')
        save_constraints(constraints, str(output_path))
        
        print(f"\nExtracted constraints for {len(constraints)} grades")
        print(f"Saved to: {output_path}")
        
        # Print sample
        for grade in list(constraints.keys())[:3]:
            data = constraints[grade]
            print(f"\n{grade}:")
            print(f"  Common starts: {data['common_starts'][:5]}")
            print(f"  Common ends: {data['common_ends'][:5]}")
            print(f"  Hold count: {data['hold_count_min']}-{data['hold_count_max']} (mean: {data['hold_count_mean']:.1f})")
    else:
        print(f"Training data not found at {train_csv}")
