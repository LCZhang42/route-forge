"""
Analyze grade distribution in training data to identify imbalance issues.
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load training data
print("Loading training data...")
df = pd.read_csv('data_reordered/moonboard_train_quality.csv')

print(f"\nTotal paths: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Analyze grade distribution
print("\n" + "="*70)
print("GRADE DISTRIBUTION ANALYSIS")
print("="*70)

grade_counts = df['grade'].value_counts().sort_index()

print("\nPaths per grade:")
print("-" * 40)
for grade, count in grade_counts.items():
    percentage = (count / len(df)) * 100
    bar = "█" * int(count / 50)  # Visual bar
    print(f"{grade:6s}: {count:5d} ({percentage:5.2f}%) {bar}")

# Identify problematic grades
print("\n" + "="*70)
print("POTENTIAL ISSUES")
print("="*70)

min_threshold = 100  # Minimum recommended paths per grade
low_count_grades = grade_counts[grade_counts < min_threshold]

if len(low_count_grades) > 0:
    print(f"\n⚠️  Grades with < {min_threshold} paths (may cause training issues):")
    print("-" * 40)
    for grade, count in low_count_grades.items():
        print(f"  {grade}: {count} paths")
    
    print(f"\n💡 Recommendation:")
    print(f"   - Consider removing grades with < 50 paths")
    print(f"   - Or use data augmentation for underrepresented grades")
    print(f"   - Or combine similar grades (e.g., 6B and 6B+)")
else:
    print("\n✅ All grades have sufficient data (>= 100 paths)")

# Statistics
print("\n" + "="*70)
print("STATISTICS")
print("="*70)
print(f"Mean paths per grade: {grade_counts.mean():.1f}")
print(f"Median paths per grade: {grade_counts.median():.1f}")
print(f"Min paths per grade: {grade_counts.min()}")
print(f"Max paths per grade: {grade_counts.max()}")
print(f"Std deviation: {grade_counts.std():.1f}")

# Calculate imbalance ratio
imbalance_ratio = grade_counts.max() / grade_counts.min()
print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}x")

if imbalance_ratio > 10:
    print("⚠️  HIGH IMBALANCE - Model may overfit to common grades")
elif imbalance_ratio > 5:
    print("⚠️  MODERATE IMBALANCE - Consider class weighting")
else:
    print("✅ BALANCED - Good for training")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if len(low_count_grades) > 0 or imbalance_ratio > 5:
    print("\n1. Filter out rare grades:")
    print("   Remove grades with < 50 paths to avoid overfitting")
    
    print("\n2. Use class weighting:")
    print("   Weight loss by inverse frequency to balance learning")
    
    print("\n3. Data augmentation:")
    print("   - Mirror paths horizontally (x -> 10-x)")
    print("   - Small random perturbations")
    
    print("\n4. Combine similar grades:")
    print("   - Merge 6B and 6B+ into '6B-6B+'")
    print("   - Merge 8A+ and 8B+ into '8A+'")
else:
    print("\n✅ Dataset is well-balanced for training!")

# Save filtered dataset recommendation
if len(low_count_grades) > 0:
    print("\n" + "="*70)
    print("CREATING FILTERED DATASET")
    print("="*70)
    
    # Filter out grades with < 50 paths
    min_paths = 50
    valid_grades = grade_counts[grade_counts >= min_paths].index.tolist()
    
    df_filtered = df[df['grade'].isin(valid_grades)]
    
    output_path = 'data_reordered/moonboard_train_quality_filtered.csv'
    df_filtered.to_csv(output_path, index=False)
    
    print(f"\n✅ Filtered dataset saved to: {output_path}")
    print(f"   Original: {len(df)} paths")
    print(f"   Filtered: {len(df_filtered)} paths ({len(df_filtered)/len(df)*100:.1f}%)")
    print(f"   Removed grades: {[g for g in grade_counts.index if g not in valid_grades]}")
    print(f"   Kept grades: {valid_grades}")

print("\n" + "="*70)
