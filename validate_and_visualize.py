"""
Validate and visualize climbing paths, filtering out invalid holds.

This script checks generated paths against the actual MoonBoard 2016 hold positions
and creates visualizations with warnings for invalid holds.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.append(str(Path(__file__).parent))

from visualize_path import MoonBoardVisualizer
from backend.models.valid_holds import get_cached_valid_holds


def validate_path(holds: List[Tuple[int, int]], valid_holds: set) -> Dict:
    """
    Validate a climbing path against valid hold positions.
    
    Args:
        holds: List of (x, y) hold coordinates
        valid_holds: Set of valid hold positions
        
    Returns:
        Dictionary with validation results
    """
    valid = []
    invalid = []
    
    for hold in holds:
        hold_tuple = tuple(hold)
        if hold_tuple in valid_holds:
            valid.append(hold)
        else:
            invalid.append(hold)
    
    return {
        'valid_holds': valid,
        'invalid_holds': invalid,
        'num_valid': len(valid),
        'num_invalid': len(invalid),
        'is_fully_valid': len(invalid) == 0,
        'validity_percentage': (len(valid) / len(holds) * 100) if holds else 0
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate and visualize climbing paths with hold validation'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to JSON file with generated paths'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='validated_visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--background',
        type=str,
        default='data/moonboard2016Background.jpg',
        help='Path to moonboard background image'
    )
    parser.add_argument(
        '--filter_invalid',
        action='store_true',
        help='Remove invalid holds from visualization'
    )
    parser.add_argument(
        '--show_stats',
        action='store_true',
        help='Show validation statistics'
    )
    
    args = parser.parse_args()
    
    # Load valid holds
    print("Loading valid hold positions from dataset...")
    valid_holds = get_cached_valid_holds()
    print(f"Found {len(valid_holds)} valid holds on MoonBoard 2016")
    print(f"(Out of {11 * 18} possible grid positions)\n")
    
    # Load paths from JSON
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    paths = data.get('paths', [])
    
    if not paths:
        print("No paths found in input file")
        return
    
    # Initialize visualizer
    visualizer = MoonBoardVisualizer(args.background)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate and visualize each path
    total_holds = 0
    total_valid = 0
    total_invalid = 0
    
    for idx, path_data in enumerate(paths):
        holds = path_data['holds']
        grade = path_data['grade']
        
        # Validate path
        validation = validate_path(holds, valid_holds)
        
        total_holds += len(holds)
        total_valid += validation['num_valid']
        total_invalid += validation['num_invalid']
        
        # Determine which holds to visualize
        if args.filter_invalid:
            viz_holds = validation['valid_holds']
            suffix = "_filtered"
        else:
            viz_holds = holds
            suffix = ""
        
        # Generate visualization
        output_path = output_dir / f"path_{idx+1}_{grade}{suffix}.png"
        visualizer.draw_path(viz_holds, grade, str(output_path))
        
        # Print validation info
        status = "[OK]" if validation['is_fully_valid'] else "[WARNING]"
        print(f"{status} Path {idx+1} ({grade}): {validation['num_valid']}/{len(holds)} valid holds ({validation['validity_percentage']:.1f}%)")
        
        if validation['num_invalid'] > 0:
            print(f"     Invalid holds: {validation['invalid_holds']}")
        
        print(f"     Saved: {output_path}")
    
    # Print summary statistics
    if args.show_stats:
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total paths: {len(paths)}")
        print(f"Total holds: {total_holds}")
        print(f"Valid holds: {total_valid} ({total_valid/total_holds*100:.1f}%)")
        print(f"Invalid holds: {total_invalid} ({total_invalid/total_holds*100:.1f}%)")
        print(f"\nValid MoonBoard 2016 positions: {len(valid_holds)}/198")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
