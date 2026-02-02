# Reachability Constraints Implementation

## Problem Identified

You correctly identified two critical issues with the sequential climbing path generation:

1. **X-coordinate distance not considered** - Only Y-coordinate sorting was used, but holds too far apart horizontally are unreachable
2. **Fundamental modeling issue** - Climbing is not truly sequential; it's a 4-limb state transition problem where the climber has 4 limbs on 4 holds simultaneously

## Data Analysis Results

Analysis of `moonboard_train_quality.csv` revealed:

### Distance Statistics
- **Typical reach**: 3.16 units (Euclidean)
- **Max reasonable reach**: 5.66 units (95th percentile)
- **Sliding window (4-limb approximation)**: 5.0 units max reach
- **Horizontal moves**: 2.0 typical, 5.0 max (95th)
- **Vertical moves**: 2.0 typical, 4.0 max (95th)

### Key Insights
- 9.7% of moves are horizontal traverses (no vertical progress)
- Only 4.7% are "big moves" exceeding typical reach
- The data confirms climbing IS constrained by physical reachability

## Solution: Hybrid Approach

Implemented a **sliding window state tracking** approach that:

1. **Approximates 4-limb body position** using last 4 unique holds
2. **Enforces reachability constraints** during generation
3. **Works with existing sequential model** and data
4. **Masks unreachable holds** in logits before sampling

### Why This Approach?

- ✅ Works with existing data (no need to reconstruct 4-limb states)
- ✅ Realistic constraints based on actual climbing data
- ✅ Easy to implement and tune
- ✅ Can be enhanced incrementally

## Implementation Details

### Core Components

1. **`backend/models/reachability_constraints.py`**
   - Distance calculations (Euclidean, Manhattan)
   - Sliding window state tracking
   - Reachability mask generation
   - Path validation utilities

2. **`backend/models/reachability_processor.py`**
   - `ReachabilityLogitsProcessor` - Standard circular reach constraint
   - `AdaptiveReachabilityProcessor` - Adjusts reach based on path progress
   - `ProgressiveReachabilityProcessor` - Prefers upward movement

3. **`backend/training/generate_paths.py`** (updated)
   - Integrated reachability processors
   - New command-line arguments for reachability control

### Three Reachability Modes

#### 1. Standard Mode (default)
```python
ReachabilityLogitsProcessor(tokenizer, max_reach=5.0)
```
- Circular reach constraint from any of last 4 holds
- Fixed max_reach distance
- Simple and predictable

#### 2. Adaptive Mode
```python
AdaptiveReachabilityProcessor(tokenizer, initial_reach=6.0, final_reach=4.5)
```
- Larger reach at start (setting up)
- Tighter constraints near finish (precise moves)
- Interpolates based on vertical progress

#### 3. Progressive Mode
```python
ProgressiveReachabilityProcessor(tokenizer, max_reach=5.0, upward_bonus=0.5)
```
- Gives bonus reach for upward moves
- Still allows sideways/downward when needed
- Soft weighting instead of hard cutoff

## Usage

### Generate with Reachability Constraints

```bash
# Standard reachability (recommended)
python backend/training/generate_paths.py \
    --grade 7A \
    --num_samples 5 \
    --max_reach 5.0 \
    --visualize

# Adaptive reachability (easier start, harder finish)
python backend/training/generate_paths.py \
    --grade 7A \
    --reachability_mode adaptive \
    --visualize

# Progressive reachability (favors upward movement)
python backend/training/generate_paths.py \
    --grade 7A \
    --reachability_mode progressive \
    --visualize

# Disable reachability (compare with/without)
python backend/training/generate_paths.py \
    --grade 7A \
    --no_reachability \
    --visualize
```

### Validate Existing Paths

```python
from backend.models.reachability_constraints import (
    validate_path_reachability,
    analyze_path_reachability
)

path = [[5, 4], [6, 7], [4, 9], [7, 12], [5, 14], [6, 17]]

# Check if path is valid
is_valid, invalid_indices = validate_path_reachability(path, max_reach=5.0)
print(f"Valid: {is_valid}")

# Get statistics
stats = analyze_path_reachability(path)
print(f"Mean reach: {stats['mean_reach']:.2f}")
print(f"Max reach: {stats['max_reach']:.2f}")
```

## How It Works

### Sliding Window State Tracking

```
Path so far: [H1, H2, H3, H4, H5, H6]
                              ^^^^^^^^
                              Last 4 unique holds
                              = Current body position

Next hold must be reachable from ANY of these 4 holds
```

### Reachability Masking

```python
# For each candidate next hold:
for hold in all_possible_holds:
    is_reachable = False
    for current_hold in last_4_holds:
        if distance(hold, current_hold) <= max_reach:
            is_reachable = True
            break
    
    if not is_reachable:
        logits[hold] = -inf  # Mask out unreachable hold
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_reach` | 5.0 | Maximum reachable distance (from data: 95th percentile) |
| `window_size` | 4 | Number of recent holds for body position (4 limbs) |
| `reachability_mode` | 'standard' | 'standard', 'adaptive', or 'progressive' |

## Future Enhancements

### Potential Improvements

1. **Explicit 4-limb state modeling**
   - Reconstruct which limb is on which hold
   - Model state transitions explicitly
   - Requires heuristic or labeled data

2. **Directional reach constraints**
   - Different max reach for horizontal vs vertical
   - Reach "cone" instead of circle
   - Account for body position/orientation

3. **Balance and center-of-gravity**
   - Ensure 3-point contact during transitions
   - Model weight distribution
   - Prevent impossible balance positions

4. **Grade-specific reach constraints**
   - Harder grades may require bigger reaches
   - Easier grades use smaller, safer moves

5. **Training data augmentation**
   - Filter training data to only include reachable sequences
   - May improve model's learned patterns

## Testing

Run the analysis script to see distance distributions:

```bash
python backend/data_processing/analyze_hold_distances.py
```

This generates:
- Distance statistics (Euclidean, Manhattan, X, Y)
- Sliding window analysis
- Visualization plots
- Recommendations for max_reach parameter

## Impact on Generated Paths

**Before reachability constraints:**
- Paths could jump arbitrarily across the board
- Unrealistic sequences (e.g., [0, 4] → [10, 15])
- No consideration of physical limitations

**After reachability constraints:**
- All moves within 5.0 units of recent holds
- Realistic climbing progressions
- Physically plausible sequences
- Still allows variety and creativity within constraints

## Comparison with Training Data

The reachability constraints are **derived from actual climbing data**, so generated paths should match the statistical properties of real climbs:

- 95% of moves within 5.0-5.7 units
- Typical move: 3.16 units
- Allows for occasional big moves (dynos)
- Permits traversing (horizontal movement)

## Next Steps

1. ✅ Fixed path sequencing (Y-coordinate sorting)
2. ✅ Implemented reachability constraints
3. ✅ Integrated into generation pipeline
4. ⏳ **Test with actual model** - Generate paths and compare with/without constraints
5. ⏳ **Retrain model** with properly sequenced data
6. ⏳ **Evaluate quality** - Do paths look more realistic?
7. ⏳ Consider implementing explicit 4-limb state model if needed

## References

- Data analysis: `backend/data_processing/analyze_hold_distances.py`
- Constraints: `backend/models/reachability_constraints.py`
- Processors: `backend/models/reachability_processor.py`
- Generation: `backend/training/generate_paths.py`
- Documentation: `docs/MODELING_APPROACHES.md`
