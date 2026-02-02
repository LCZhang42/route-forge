"""
Analyze hold-to-hold distances in climbing paths to understand reachability constraints.
"""

import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def euclidean_distance(hold1, hold2):
    """Calculate Euclidean distance between two holds"""
    return ((hold1[0] - hold2[0])**2 + (hold1[1] - hold2[1])**2)**0.5

def manhattan_distance(hold1, hold2):
    """Calculate Manhattan distance (|dx| + |dy|)"""
    return abs(hold1[0] - hold2[0]) + abs(hold1[1] - hold2[1])

def analyze_distances(csv_path):
    """Analyze hold distances in climbing paths"""
    print(f'\nAnalyzing: {csv_path}')
    df = pd.read_csv(csv_path)
    
    euclidean_dists = []
    manhattan_dists = []
    x_dists = []
    y_dists = []
    
    # Analyze consecutive hold distances
    for path_str in df['full_path']:
        try:
            path = ast.literal_eval(path_str)
            for i in range(len(path) - 1):
                h1, h2 = path[i], path[i+1]
                
                euclidean_dists.append(euclidean_distance(h1, h2))
                manhattan_dists.append(manhattan_distance(h1, h2))
                x_dists.append(abs(h2[0] - h1[0]))
                y_dists.append(abs(h2[1] - h1[1]))
        except:
            continue
    
    return {
        'euclidean': euclidean_dists,
        'manhattan': manhattan_dists,
        'x_dist': x_dists,
        'y_dist': y_dists
    }

def print_statistics(distances, name):
    """Print distance statistics"""
    print(f'\n{name} Distance Statistics:')
    print(f'  Mean:   {np.mean(distances):.2f}')
    print(f'  Median: {np.median(distances):.2f}')
    print(f'  Std:    {np.std(distances):.2f}')
    print(f'  Min:    {np.min(distances):.2f}')
    print(f'  Max:    {np.max(distances):.2f}')
    print(f'  25th percentile: {np.percentile(distances, 25):.2f}')
    print(f'  75th percentile: {np.percentile(distances, 75):.2f}')
    print(f'  90th percentile: {np.percentile(distances, 90):.2f}')
    print(f'  95th percentile: {np.percentile(distances, 95):.2f}')
    print(f'  99th percentile: {np.percentile(distances, 99):.2f}')

def analyze_sliding_window_distances(csv_path, window_size=4):
    """
    Analyze distances assuming last N holds represent body position.
    For each new hold, find minimum distance to any of the last N holds.
    """
    print(f'\n\nSliding Window Analysis (window_size={window_size}):')
    df = pd.read_csv(csv_path)
    
    min_distances = []
    
    for path_str in df['full_path']:
        try:
            path = ast.literal_eval(path_str)
            
            # Start after we have enough holds for a window
            for i in range(window_size, len(path)):
                new_hold = path[i]
                window = path[i-window_size:i]
                
                # Find minimum distance from new hold to any hold in window
                min_dist = min(euclidean_distance(new_hold, w) for w in window)
                min_distances.append(min_dist)
        except:
            continue
    
    print_statistics(min_distances, 'Minimum Distance to Window')
    
    return min_distances

def plot_distributions(distances_dict, output_path):
    """Plot distance distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Euclidean distances
    axes[0, 0].hist(distances_dict['euclidean'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Euclidean Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Euclidean Distance Between Consecutive Holds')
    axes[0, 0].axvline(np.median(distances_dict['euclidean']), color='red', 
                       linestyle='--', label=f'Median: {np.median(distances_dict["euclidean"]):.2f}')
    axes[0, 0].axvline(np.percentile(distances_dict['euclidean'], 95), color='orange', 
                       linestyle='--', label=f'95th: {np.percentile(distances_dict["euclidean"], 95):.2f}')
    axes[0, 0].legend()
    
    # X distances
    axes[0, 1].hist(distances_dict['x_dist'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('X Distance (Horizontal)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Horizontal Distance Between Consecutive Holds')
    axes[0, 1].axvline(np.median(distances_dict['x_dist']), color='red', 
                       linestyle='--', label=f'Median: {np.median(distances_dict["x_dist"]):.2f}')
    axes[0, 1].legend()
    
    # Y distances
    axes[1, 0].hist(distances_dict['y_dist'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Y Distance (Vertical)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Vertical Distance Between Consecutive Holds')
    axes[1, 0].axvline(np.median(distances_dict['y_dist']), color='red', 
                       linestyle='--', label=f'Median: {np.median(distances_dict["y_dist"]):.2f}')
    axes[1, 0].legend()
    
    # Manhattan distances
    axes[1, 1].hist(distances_dict['manhattan'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Manhattan Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Manhattan Distance Between Consecutive Holds')
    axes[1, 1].axvline(np.median(distances_dict['manhattan']), color='red', 
                       linestyle='--', label=f'Median: {np.median(distances_dict["manhattan"]):.2f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved plot to: {output_path}')

print('=' * 70)
print('HOLD DISTANCE ANALYSIS')
print('=' * 70)

# Analyze training data
train_csv = Path('data/moonboard_train_quality.csv')

if train_csv.exists():
    distances = analyze_distances(str(train_csv))
    
    print('\n' + '=' * 70)
    print('CONSECUTIVE HOLD DISTANCES')
    print('=' * 70)
    
    print_statistics(distances['euclidean'], 'Euclidean')
    print_statistics(distances['x_dist'], 'X (Horizontal)')
    print_statistics(distances['y_dist'], 'Y (Vertical)')
    print_statistics(distances['manhattan'], 'Manhattan')
    
    # Sliding window analysis
    print('\n' + '=' * 70)
    print('BODY POSITION WINDOW ANALYSIS')
    print('=' * 70)
    
    window_dists = analyze_sliding_window_distances(str(train_csv), window_size=4)
    
    # Plot distributions
    output_plot = Path('training_plots/hold_distance_distributions.png')
    output_plot.parent.mkdir(exist_ok=True)
    plot_distributions(distances, str(output_plot))
    
    # Recommendations
    print('\n' + '=' * 70)
    print('RECOMMENDATIONS FOR REACHABILITY CONSTRAINTS')
    print('=' * 70)
    
    max_consecutive = np.percentile(distances['euclidean'], 95)
    max_window = np.percentile(window_dists, 95)
    
    print(f'\n1. Consecutive Hold Model (current approach):')
    print(f'   - Typical reach: {np.median(distances["euclidean"]):.2f} units')
    print(f'   - Max reach (95th percentile): {max_consecutive:.2f} units')
    print(f'   - Suggested max_reach parameter: {max_consecutive:.1f}')
    
    print(f'\n2. Sliding Window Model (4-limb approximation):')
    print(f'   - Typical reach from body: {np.median(window_dists):.2f} units')
    print(f'   - Max reach (95th percentile): {max_window:.2f} units')
    print(f'   - Suggested max_reach parameter: {max_window:.1f}')
    
    print(f'\n3. Directional constraints:')
    print(f'   - Typical horizontal move: {np.median(distances["x_dist"]):.2f} units')
    print(f'   - Max horizontal (95th): {np.percentile(distances["x_dist"], 95):.2f} units')
    print(f'   - Typical vertical move: {np.median(distances["y_dist"]):.2f} units')
    print(f'   - Max vertical (95th): {np.percentile(distances["y_dist"], 95):.2f} units')
    
    print(f'\n4. Key insights:')
    if np.median(distances['y_dist']) > np.median(distances['x_dist']):
        print(f'   - Climbers move more VERTICALLY than horizontally (as expected)')
    
    zero_y = sum(1 for d in distances['y_dist'] if d == 0)
    total = len(distances['y_dist'])
    print(f'   - {zero_y/total*100:.1f}% of moves have no vertical progress (traversing)')
    
    large_jumps = sum(1 for d in distances['euclidean'] if d > max_consecutive)
    print(f'   - {large_jumps/total*100:.1f}% of moves exceed 95th percentile (dynos/big moves)')
    
    print('\n' + '=' * 70)
    print('ANALYSIS COMPLETE')
    print('=' * 70)
    
else:
    print(f'Training data not found at {train_csv}')
