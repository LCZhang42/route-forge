# Valid Holds Constraint Guide

## Problem

The MoonBoard 2016 has **141 physical holds** out of 198 possible grid positions (11 columns × 18 rows). The model was trained on coordinate data assuming all grid positions are valid, which causes it to generate holds at positions where no boulder exists on the physical board.

**Impact**: ~21% of generated holds were invalid (circles drawn on empty spaces).

## Solution: Option C - Logits Masking During Generation

We implemented a `ValidHoldsLogitsProcessor` that masks out invalid hold positions during generation, ensuring only the 141 valid holds can be generated.

### How It Works

1. **Valid Holds Extraction**: Extracts all unique hold positions from the training dataset
2. **Logits Masking**: During generation, masks logits for invalid X/Y coordinates
3. **Context-Aware**: For Y coordinates, only allows Y values that form valid (X, Y) pairs

### Files Created/Modified

#### New Files
- `backend/models/valid_holds.py` - Module for loading and validating hold positions
- `validate_and_visualize.py` - Script to validate and visualize paths with hold checking
- `VALID_HOLDS_GUIDE.md` - This documentation

#### Modified Files
- `backend/models/logits_processor.py` - Added `ValidHoldsLogitsProcessor` and `NoRepeatHoldsLogitsProcessor`
- `backend/training/generate_paths.py` - Added `--valid_holds_only` flag and valid holds support
- `backend/api/server.py` - Added `valid_holds_only` parameter to API

### Usage

#### Command Line Generation

```bash
# Generate paths with valid holds constraint
python backend/training/generate_paths.py \
  --checkpoint checkpoints/climb_path_cpu/best.pt \
  --grade 7A \
  --num_samples 5 \
  --valid_holds_only \
  --save_json generated_paths.json

# Without valid holds constraint (old behavior)
python backend/training/generate_paths.py \
  --checkpoint checkpoints/climb_path_cpu/best.pt \
  --grade 7A \
  --num_samples 5 \
  --save_json generated_paths.json
```

#### Validation and Visualization

```bash
# Validate generated paths and show statistics
python validate_and_visualize.py \
  --input generated_paths.json \
  --output validated_viz \
  --show_stats

# Filter out invalid holds before visualization
python validate_and_visualize.py \
  --input generated_paths.json \
  --output validated_viz \
  --filter_invalid \
  --show_stats
```

#### API Usage

The API now supports `valid_holds_only` parameter (default: `true`):

```javascript
// Frontend API call
const response = await fetch('http://localhost:8000/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    grade: '7A',
    temperature: 1.0,
    min_holds: 3,
    max_holds: 30,
    use_constraints: true,
    valid_holds_only: true  // Enable valid holds constraint
  })
});
```

### Results

**Before (without constraint)**:
- Total holds: 24
- Valid holds: 19 (79.2%)
- Invalid holds: 5 (20.8%)

**After (with constraint)**:
- Total holds: 90
- Valid holds: 90 (100.0%)
- Invalid holds: 0 (0.0%)

### Known Limitations

1. **Path Quality**: The model wasn't trained with this constraint, so it may generate repetitive or unrealistic paths when heavily constrained
2. **Performance**: Additional logits masking adds slight computational overhead
3. **Training Data Dependency**: Requires training data to extract valid holds

### Recommendations

For production use, consider:

1. **Retrain the model** with a tokenizer that only includes the 141 valid holds (Option A)
2. **Add path quality scoring** to filter out repetitive paths
3. **Implement beam search** instead of sampling for more diverse results
4. **Add hold reachability constraints** based on climbing biomechanics

### Technical Details

The `ValidHoldsLogitsProcessor` works by:

1. Precomputing valid X coordinates and valid Y coordinates for each X
2. During X coordinate generation: masking all X tokens except valid ones
3. During Y coordinate generation: looking at the previous X token and masking all Y tokens except those that form valid (X, Y) pairs with that X

This ensures every generated hold corresponds to an actual boulder on the MoonBoard 2016.
