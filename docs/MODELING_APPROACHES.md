# Climbing Path Generation: Modeling Approaches

## Problem Statement

Climbing is fundamentally a **4-limb state transition problem**, not a sequential hold-by-hold problem.

### Current Issues
1. Sequential model doesn't know which limb is on which hold
2. No physical reachability constraints (X/Y distance)
3. No body position or balance modeling
4. Sorted Y-coordinates help but don't solve the fundamental problem

## Proposed Approaches

### Option 1: Enhanced Sequential with Distance Constraints

**Model**: Keep autoregressive sequence generation, add constraints

**Changes**:
```python
# During generation, mask unreachable holds
def get_reachable_holds(recent_holds, all_holds, max_reach=5.0):
    """Only allow holds within reach distance of recent positions"""
    reachable = []
    for hold in all_holds:
        for recent in recent_holds[-4:]:  # Last 4 holds = approximate body position
            dist = euclidean_distance(hold, recent)
            if dist <= max_reach:
                reachable.append(hold)
                break
    return reachable
```

**Data representation**: No change needed
```
Input: [grade, start_holds]
Output: [hold_1, hold_2, hold_3, ..., hold_n]
```

**Pros**:
- Minimal code changes
- Works with existing data
- Easy to implement

**Cons**:
- Still doesn't model true climbing mechanics
- Approximation of body position
- May generate awkward sequences

---

### Option 2: State-Based Generation (4-Limb States)

**Model**: Generate state transitions explicitly

**State representation**:
```python
State = {
    'left_hand': [x, y],
    'right_hand': [x, y],
    'left_foot': [x, y],
    'right_foot': [x, y]
}

Action = {
    'limb': 'left_hand' | 'right_hand' | 'left_foot' | 'right_foot',
    'target_hold': [x, y]
}
```

**Sequence**:
```
State_0 → Action_1 → State_1 → Action_2 → State_2 → ... → State_final
```

**Data reconstruction** (heuristic from existing data):
```python
def reconstruct_states(hold_sequence):
    """Convert hold sequence to state transitions"""
    states = []
    
    # Initial state: first 4 holds = 2 hands, 2 feet
    state = {
        'left_hand': hold_sequence[0],
        'right_hand': hold_sequence[1] if len(hold_sequence) > 1 else hold_sequence[0],
        'left_foot': hold_sequence[2] if len(hold_sequence) > 2 else hold_sequence[0],
        'right_foot': hold_sequence[3] if len(hold_sequence) > 3 else hold_sequence[1]
    }
    states.append(state)
    
    # For each subsequent hold, determine which limb moves
    for i in range(4, len(hold_sequence)):
        next_hold = hold_sequence[i]
        
        # Heuristic: move the limb furthest from the next hold
        # or alternate hands/feet based on Y-progression
        limb_to_move = choose_limb_heuristic(state, next_hold, i)
        
        state = state.copy()
        state[limb_to_move] = next_hold
        states.append(state)
    
    return states
```

**Model architecture**:
```python
# Input: current state (4 holds) + grade
# Output: (limb_to_move, target_hold)

class ClimbingStateModel(nn.Module):
    def forward(self, state, grade):
        # Encode current 4-limb state
        state_embedding = encode_state(state)
        
        # Predict which limb to move (4-way classification)
        limb_logits = self.limb_head(state_embedding)
        
        # Predict target hold (grid position)
        hold_logits = self.hold_head(state_embedding)
        
        return limb_logits, hold_logits
```

**Pros**:
- Physically accurate model of climbing
- Can enforce reachability per limb
- Can model balance and body position
- More interpretable

**Cons**:
- Need to reconstruct states from existing data (heuristic, may be noisy)
- More complex model architecture
- Harder to train (two prediction heads)

---

### Option 3: Hybrid Approach (Practical)

**Model**: Sequential generation with sliding window state tracking

**Key idea**: 
- Generate holds sequentially (like current model)
- Track "active window" of last 4 unique holds as approximate body position
- Constrain next hold based on reachability from any of the 4 active holds

**Implementation**:
```python
def generate_with_state_tracking(model, grade, start_holds, max_length=20):
    """Generate path with state-aware constraints"""
    path = list(start_holds)
    
    while len(path) < max_length:
        # Active body position = last 4 unique holds
        active_holds = get_last_n_unique(path, n=4)
        
        # Get next hold from model
        next_hold_logits = model.predict_next(path, grade)
        
        # Mask unreachable holds
        reachable_mask = get_reachability_mask(active_holds, all_holds, max_reach=5.0)
        next_hold_logits = next_hold_logits * reachable_mask
        
        # Sample next hold
        next_hold = sample_from_logits(next_hold_logits)
        path.append(next_hold)
        
        # Stop if reached top
        if next_hold[1] >= 17:
            break
    
    return path

def get_reachability_mask(active_holds, all_holds, max_reach=5.0):
    """Return mask of holds reachable from any active hold"""
    mask = torch.zeros(len(all_holds))
    
    for i, hold in enumerate(all_holds):
        for active in active_holds:
            dist = euclidean_distance(hold, active)
            if dist <= max_reach:
                mask[i] = 1.0
                break
    
    return mask

def euclidean_distance(hold1, hold2):
    """Calculate Euclidean distance between two holds"""
    return ((hold1[0] - hold2[0])**2 + (hold1[1] - hold2[1])**2)**0.5
```

**Additional heuristics**:
```python
# Prefer alternating hands/feet (hands move more than feet)
# Prefer upward progression (Y-coordinate increasing)
# Prefer holds within "reach cone" (not just circle)
```

**Pros**:
- Works with existing data and model
- More realistic than pure sequential
- Easy to add constraints incrementally
- Can tune max_reach parameter

**Cons**:
- Still an approximation
- Doesn't explicitly model which limb is where
- May miss some valid sequences

---

## Recommendation

**Start with Option 3 (Hybrid)**, then potentially move to Option 2 if needed.

### Immediate next steps:
1. Add distance-based reachability constraints to generation
2. Implement sliding window state tracking (last 4 holds)
3. Add max_reach parameter tuning (estimate from data)
4. Test if generated paths are more realistic

### Future improvements:
1. Analyze real climbing data to estimate typical reach distances
2. Add directional reach constraints (easier to reach up than down)
3. Consider implementing Option 2 if hybrid approach is insufficient
4. Potentially collect data with explicit limb annotations

---

## Distance Analysis Needed

To implement any of these approaches, we need to analyze the training data:

```python
# Calculate typical distances between consecutive holds
def analyze_hold_distances(csv_path):
    df = pd.read_csv(csv_path)
    
    distances = []
    for path_str in df['full_path']:
        path = ast.literal_eval(path_str)
        for i in range(len(path) - 1):
            dist = euclidean_distance(path[i], path[i+1])
            distances.append(dist)
    
    print(f"Mean distance: {np.mean(distances):.2f}")
    print(f"Median distance: {np.median(distances):.2f}")
    print(f"95th percentile: {np.percentile(distances, 95):.2f}")
    print(f"Max distance: {np.max(distances):.2f}")
```

This will tell us what `max_reach` should be.
