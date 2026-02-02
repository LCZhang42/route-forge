"""
Check training data for vertical progression patterns.
"""

import pandas as pd
import json
import numpy as np

print("="*70)
print("ANALYZING TRAINING DATA FOR VERTICAL PROGRESSION")
print("="*70)

# Load training data
df = pd.read_csv('data/moonboard_train_quality.csv')
print(f"\nTotal training samples: {len(df)}")

# Analyze vertical progression in training data
print("\n" + "="*70)
print("SAMPLE ANALYSIS (First 20 paths)")
print("="*70)

flat_count = 0
good_vertical_count = 0
bad_start_count = 0
bad_end_count = 0

for i in range(min(20, len(df))):
    sample = df.iloc[i]
    path = json.loads(sample['full_path'])
    y_coords = [h[1] for h in path]
    
    start_y = y_coords[0]
    end_y = y_coords[-1]
    vertical_gain = end_y - start_y
    y_range = max(y_coords) - min(y_coords)
    
    # Check for flat paths
    is_flat = y_range < 3
    has_good_vertical = vertical_gain >= 8
    bad_start = start_y > 6
    bad_end = end_y < 12
    
    if is_flat:
        flat_count += 1
    if has_good_vertical:
        good_vertical_count += 1
    if bad_start:
        bad_start_count += 1
    if bad_end:
        bad_end_count += 1
    
    status = []
    if is_flat:
        status.append("FLAT")
    if bad_start:
        status.append("HIGH_START")
    if bad_end:
        status.append("LOW_END")
    if has_good_vertical:
        status.append("GOOD_VERTICAL")
    
    status_str = ", ".join(status) if status else "OK"
    
    print(f"\n{i+1}. Grade: {sample['grade']}")
    print(f"   Path: {path[:2]} ... {path[-2:]}")
    print(f"   Y coords: {y_coords}")
    print(f"   Start Y: {start_y}, End Y: {end_y}, Gain: {vertical_gain:+d}, Range: {y_range}")
    print(f"   Status: {status_str}")

# Overall statistics
print("\n" + "="*70)
print("OVERALL STATISTICS (All training data)")
print("="*70)

all_vertical_gains = []
all_start_y = []
all_end_y = []
all_y_ranges = []
all_flat = 0
all_good_vertical = 0
all_bad_start = 0
all_bad_end = 0

for i in range(len(df)):
    sample = df.iloc[i]
    path = json.loads(sample['full_path'])
    y_coords = [h[1] for h in path]
    
    start_y = y_coords[0]
    end_y = y_coords[-1]
    vertical_gain = end_y - start_y
    y_range = max(y_coords) - min(y_coords)
    
    all_vertical_gains.append(vertical_gain)
    all_start_y.append(start_y)
    all_end_y.append(end_y)
    all_y_ranges.append(y_range)
    
    if y_range < 3:
        all_flat += 1
    if vertical_gain >= 8:
        all_good_vertical += 1
    if start_y > 6:
        all_bad_start += 1
    if end_y < 12:
        all_bad_end += 1

print(f"\nVertical Gain Statistics:")
print(f"  Mean: {np.mean(all_vertical_gains):.2f}")
print(f"  Median: {np.median(all_vertical_gains):.2f}")
print(f"  Min: {np.min(all_vertical_gains)}")
print(f"  Max: {np.max(all_vertical_gains)}")
print(f"  Paths with gain >= 8: {all_good_vertical} ({all_good_vertical/len(df)*100:.1f}%)")

print(f"\nStart Position (Y) Statistics:")
print(f"  Mean: {np.mean(all_start_y):.2f}")
print(f"  Median: {np.median(all_start_y):.2f}")
print(f"  Min: {np.min(all_start_y)}")
print(f"  Max: {np.max(all_start_y)}")
print(f"  Paths starting at Y > 6: {all_bad_start} ({all_bad_start/len(df)*100:.1f}%)")

print(f"\nEnd Position (Y) Statistics:")
print(f"  Mean: {np.mean(all_end_y):.2f}")
print(f"  Median: {np.median(all_end_y):.2f}")
print(f"  Min: {np.min(all_end_y)}")
print(f"  Max: {np.max(all_end_y)}")
print(f"  Paths ending at Y < 12: {all_bad_end} ({all_bad_end/len(df)*100:.1f}%)")

print(f"\nY Range Statistics:")
print(f"  Mean: {np.mean(all_y_ranges):.2f}")
print(f"  Median: {np.median(all_y_ranges):.2f}")
print(f"  Flat paths (range < 3): {all_flat} ({all_flat/len(df)*100:.1f}%)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if all_flat > len(df) * 0.1:
    print(f"[WARNING] {all_flat/len(df)*100:.1f}% of training data is FLAT!")
    print("          This could explain why the model generates flat paths.")
else:
    print(f"[OK] Only {all_flat/len(df)*100:.1f}% of training data is flat.")
    print("     Training data has good vertical progression.")

if all_good_vertical > len(df) * 0.8:
    print(f"[OK] {all_good_vertical/len(df)*100:.1f}% of paths have good vertical gain (>= 8 units)")
else:
    print(f"[WARNING] Only {all_good_vertical/len(df)*100:.1f}% have good vertical gain")

print("="*70)
