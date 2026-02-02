"""
Filter training data by quality metrics.

Implements various filtering strategies:
1. Minimum repeats threshold
2. Benchmark-only filtering
3. Quality-weighted sampling
4. Stratified sampling by grade and quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

print('=' * 70)
print('QUALITY-BASED DATA FILTERING')
print('=' * 70)

# Load cleaned data
df = pd.read_csv('data/moonboard_cleaned.csv')
print(f'\nOriginal dataset: {len(df)} problems')

# Display quality distribution
print('\n1. QUALITY DISTRIBUTION')
print('-' * 70)
print(f'Benchmark problems: {df["is_benchmark"].sum()} ({df["is_benchmark"].sum()/len(df)*100:.1f}%)')
print(f'\nRepeats distribution:')
print(df['repeats'].describe())

print('\n   Repeats breakdown:')
print(f'   0 repeats:     {(df["repeats"] == 0).sum():>6} problems')
print(f'   1-9 repeats:   {((df["repeats"] >= 1) & (df["repeats"] < 10)).sum():>6} problems')
print(f'   10-49 repeats: {((df["repeats"] >= 10) & (df["repeats"] < 50)).sum():>6} problems')
print(f'   50-99 repeats: {((df["repeats"] >= 50) & (df["repeats"] < 100)).sum():>6} problems')
print(f'   100+ repeats:  {(df["repeats"] >= 100).sum():>6} problems')

# Strategy 1: Filter by minimum repeats
print('\n2. STRATEGY 1: MINIMUM REPEATS FILTER')
print('-' * 70)

min_repeats_options = [1, 5, 10, 20]
for min_rep in min_repeats_options:
    filtered = df[df['repeats'] >= min_rep]
    print(f'   >= {min_rep:>2} repeats: {len(filtered):>6} problems ({len(filtered)/len(df)*100:>5.1f}%)')

# Apply recommended filter (>= 10 repeats)
df_quality = df[df['repeats'] >= 10].copy()
print(f'\n   ✓ Recommended: >= 10 repeats → {len(df_quality)} problems')

# Strategy 2: Benchmark-only
print('\n3. STRATEGY 2: BENCHMARK-ONLY FILTER')
print('-' * 70)

df_benchmark = df[df['is_benchmark'] == True].copy()
print(f'   Benchmark problems: {len(df_benchmark)} problems')
print(f'   Grade distribution:')
for grade in sorted(df_benchmark['grade'].unique()):
    count = (df_benchmark['grade'] == grade).sum()
    print(f'      {grade:>4}: {count:>4} problems')

# Strategy 3: Quality weighting
print('\n4. STRATEGY 3: QUALITY WEIGHTING')
print('-' * 70)

# Create quality weights (log scale to avoid extreme weights)
df_quality['quality_weight'] = np.log1p(df_quality['repeats'])
df_quality['quality_weight'] = df_quality['quality_weight'] / df_quality['quality_weight'].max()

print(f'   Weight range: {df_quality["quality_weight"].min():.3f} to {df_quality["quality_weight"].max():.3f}')
print(f'   Mean weight: {df_quality["quality_weight"].mean():.3f}')
print(f'\n   Example weights:')
print(f'      10 repeats  → weight {np.log1p(10) / np.log1p(df_quality["repeats"].max()):.3f}')
print(f'      50 repeats  → weight {np.log1p(50) / np.log1p(df_quality["repeats"].max()):.3f}')
print(f'      100 repeats → weight {np.log1p(100) / np.log1p(df_quality["repeats"].max()):.3f}')
print(f'      500 repeats → weight {np.log1p(500) / np.log1p(df_quality["repeats"].max()):.3f}')

# Strategy 4: Stratified sampling by grade and quality
print('\n5. STRATEGY 4: STRATIFIED SAMPLING')
print('-' * 70)

# Create quality bins
df_quality['quality_bin'] = pd.qcut(df_quality['repeats'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')

# Create stratification key
df_quality['strata'] = df_quality['grade'].astype(str) + '_' + df_quality['quality_bin'].astype(str)

print(f'   Created {df_quality["strata"].nunique()} strata (grade × quality)')
print(f'\n   Strata distribution:')
strata_counts = df_quality['strata'].value_counts().sort_index()
for strata, count in strata_counts.head(10).items():
    print(f'      {strata:>15}: {count:>4} problems')
if len(strata_counts) > 10:
    print(f'      ... and {len(strata_counts) - 10} more strata')

# Check for strata with too few samples (< 2)
small_strata = strata_counts[strata_counts < 2]
if len(small_strata) > 0:
    print(f'\n   Warning: {len(small_strata)} strata have < 2 samples')
    print(f'   Falling back to grade-only stratification')
    stratify_by = df_quality['grade']
else:
    stratify_by = df_quality['strata']

# Save filtered datasets
print('\n6. SAVING FILTERED DATASETS')
print('-' * 70)

# Save quality-filtered dataset (>= 10 repeats)
output_dir = Path('data')

# Check if any grade has too few samples for stratification
grade_counts = df_quality['grade'].value_counts()
small_grades = grade_counts[grade_counts < 2]

if len(small_grades) > 0:
    print(f'\n   Warning: {len(small_grades)} grades have < 2 samples: {small_grades.index.tolist()}')
    print(f'   Removing these grades for proper stratification')
    df_quality = df_quality[~df_quality['grade'].isin(small_grades.index)]
    print(f'   Remaining: {len(df_quality)} problems')
    
    # Update stratify_by after filtering
    if isinstance(stratify_by, pd.Series):
        stratify_by = df_quality['grade'] if 'grade' in str(stratify_by.name) else df_quality['strata']

# Train/val/test split with stratification
try:
    train_data, temp_data = train_test_split(
        df_quality, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_quality['grade']  # Always use grade for safety
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_data['grade']
    )
except ValueError as e:
    print(f'\n   Warning: Stratification failed: {e}')
    print(f'   Falling back to random split without stratification')
    train_data, temp_data = train_test_split(
        df_quality, 
        test_size=0.2, 
        random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=42
    )

# Save quality-filtered splits
train_data.to_csv(output_dir / 'moonboard_train_quality.csv', index=False)
val_data.to_csv(output_dir / 'moonboard_val_quality.csv', index=False)
test_data.to_csv(output_dir / 'moonboard_test_quality.csv', index=False)

print(f'   ✓ Saved quality-filtered splits (>= 10 repeats):')
print(f'      data/moonboard_train_quality.csv ({len(train_data)} problems)')
print(f'      data/moonboard_val_quality.csv   ({len(val_data)} problems)')
print(f'      data/moonboard_test_quality.csv  ({len(test_data)} problems)')

# Save benchmark-only splits
if len(df_benchmark) > 100:  # Only if we have enough benchmark problems
    # Check for grades with too few samples in benchmark set
    bench_grade_counts = df_benchmark['grade'].value_counts()
    small_bench_grades = bench_grade_counts[bench_grade_counts < 2]
    
    if len(small_bench_grades) > 0:
        print(f'\n   Warning: Benchmark set has {len(small_bench_grades)} grades with < 2 samples: {small_bench_grades.index.tolist()}')
        print(f'   Removing these grades from benchmark set')
        df_benchmark_filtered = df_benchmark[~df_benchmark['grade'].isin(small_bench_grades.index)]
        print(f'   Remaining benchmark problems: {len(df_benchmark_filtered)}')
    else:
        df_benchmark_filtered = df_benchmark
    
    try:
        train_bench, temp_bench = train_test_split(
            df_benchmark_filtered, 
            test_size=0.2, 
            random_state=42, 
            stratify=df_benchmark_filtered['grade']
        )
        val_bench, test_bench = train_test_split(
            temp_bench, 
            test_size=0.5, 
            random_state=42, 
            stratify=temp_bench['grade']
        )
    except ValueError as e:
        print(f'\n   Warning: Benchmark stratification failed: {e}')
        print(f'   Using random split for benchmark set')
        train_bench, temp_bench = train_test_split(
            df_benchmark_filtered, 
            test_size=0.2, 
            random_state=42
        )
        val_bench, test_bench = train_test_split(
            temp_bench, 
            test_size=0.5, 
            random_state=42
        )
    
    train_bench.to_csv(output_dir / 'moonboard_train_benchmark.csv', index=False)
    val_bench.to_csv(output_dir / 'moonboard_val_benchmark.csv', index=False)
    test_bench.to_csv(output_dir / 'moonboard_test_benchmark.csv', index=False)
    
    print(f'\n   ✓ Saved benchmark-only splits:')
    print(f'      data/moonboard_train_benchmark.csv ({len(train_bench)} problems)')
    print(f'      data/moonboard_val_benchmark.csv   ({len(val_bench)} problems)')
    print(f'      data/moonboard_test_benchmark.csv  ({len(test_bench)} problems)')
else:
    print(f'\n   ⚠ Skipping benchmark splits (only {len(df_benchmark)} benchmark problems, need > 100)')

# Save quality weights for training
weights_df = df_quality[['problem_id', 'quality_weight']].copy()
weights_df.to_csv(output_dir / 'quality_weights.csv', index=False)
print(f'\n   ✓ Saved quality weights:')
print(f'      data/quality_weights.csv')

print('\n7. RECOMMENDATIONS')
print('-' * 70)
print('   Training strategies:')
print('')
print('   A. Quick prototype (benchmark only):')
print('      python backend/training/train_autoregressive.py \\')
print('          --train_csv data/moonboard_train_benchmark.csv \\')
print('          --val_csv data/moonboard_val_benchmark.csv \\')
print('          --num_epochs 30')
print('')
print('   B. Quality-filtered (recommended):')
print('      python backend/training/train_autoregressive.py \\')
print('          --train_csv data/moonboard_train_quality.csv \\')
print('          --val_csv data/moonboard_val_quality.csv \\')
print('          --num_epochs 50')
print('')
print('   C. Full dataset (maximum data):')
print('      python backend/training/train_autoregressive.py \\')
print('          --train_csv data/moonboard_train.csv \\')
print('          --val_csv data/moonboard_val.csv \\')
print('          --num_epochs 100')

print('\n' + '=' * 70)
print('FILTERING COMPLETE!')
print('=' * 70)
