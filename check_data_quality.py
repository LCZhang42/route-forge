import pandas as pd
import ast

def check_duplicates_and_leakage(train_path, test_path):
    """
    Check for duplicate paths within datasets and data leakage between train/test sets.
    """
    print("=" * 80)
    print("DATA QUALITY CHECK: Duplicates and Data Leakage Analysis")
    print("=" * 80)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Training set: {len(train_df)} samples")
    print(f"   Test set: {len(test_df)} samples")
    
    # Check for duplicates within training set
    print("\n2. Checking for duplicate paths in TRAINING set...")
    train_paths = train_df['full_path'].apply(str)
    train_duplicates = train_paths.duplicated()
    train_dup_count = train_duplicates.sum()
    
    if train_dup_count > 0:
        print(f"   WARNING: Found {train_dup_count} duplicate paths in training set!")
        dup_paths = train_df[train_duplicates][['problem_id', 'grade', 'full_path', 'repeats', 'quality_score']]
        print("\n   Duplicate entries:")
        print(dup_paths.to_string(index=False))
        
        # Show which paths are duplicated
        print("\n   Paths that appear multiple times:")
        dup_path_values = train_paths[train_duplicates].unique()
        for path in dup_path_values[:10]:  # Show first 10
            matching = train_df[train_paths == path][['problem_id', 'grade', 'repeats']]
            print(f"\n   Path: {path}")
            print(f"   Appears in problems: {matching['problem_id'].tolist()}")
    else:
        print(f"   OK: No duplicate paths found in training set")
    
    # Check for duplicates within test set
    print("\n3. Checking for duplicate paths in TEST set...")
    test_paths = test_df['full_path'].apply(str)
    test_duplicates = test_paths.duplicated()
    test_dup_count = test_duplicates.sum()
    
    if test_dup_count > 0:
        print(f"   WARNING: Found {test_dup_count} duplicate paths in test set!")
        dup_paths = test_df[test_duplicates][['problem_id', 'grade', 'full_path', 'repeats', 'quality_score']]
        print("\n   Duplicate entries:")
        print(dup_paths.to_string(index=False))
        
        # Show which paths are duplicated
        print("\n   Paths that appear multiple times:")
        dup_path_values = test_paths[test_duplicates].unique()
        for path in dup_path_values[:10]:  # Show first 10
            matching = test_df[test_paths == path][['problem_id', 'grade', 'repeats']]
            print(f"\n   Path: {path}")
            print(f"   Appears in problems: {matching['problem_id'].tolist()}")
    else:
        print(f"   OK: No duplicate paths found in test set")
    
    # Check for data leakage (same paths in both train and test)
    print("\n4. Checking for DATA LEAKAGE (paths appearing in both train and test)...")
    train_path_set = set(train_paths)
    test_path_set = set(test_paths)
    
    leaked_paths = train_path_set.intersection(test_path_set)
    
    if len(leaked_paths) > 0:
        print(f"   CRITICAL: Found {len(leaked_paths)} paths that appear in BOTH train and test sets!")
        print(f"   This is DATA LEAKAGE and will inflate model performance metrics!\n")
        
        # Show details of leaked paths
        print("   Leaked path details:")
        for i, path in enumerate(list(leaked_paths)[:10]):  # Show first 10
            train_entry = train_df[train_paths == path].iloc[0]
            test_entry = test_df[test_paths == path].iloc[0]
            print(f"\n   Leak #{i+1}:")
            print(f"   Path: {path}")
            print(f"   Train: problem_id={train_entry['problem_id']}, grade={train_entry['grade']}, repeats={train_entry['repeats']}")
            print(f"   Test:  problem_id={test_entry['problem_id']}, grade={test_entry['grade']}, repeats={test_entry['repeats']}")
        
        if len(leaked_paths) > 10:
            print(f"\n   ... and {len(leaked_paths) - 10} more leaked paths")
        
        # Check if it's the same problem_id or different problems with same path
        print("\n   Analyzing leak type:")
        same_problem_count = 0
        diff_problem_count = 0
        
        for path in leaked_paths:
            train_ids = set(train_df[train_paths == path]['problem_id'])
            test_ids = set(test_df[test_paths == path]['problem_id'])
            if train_ids.intersection(test_ids):
                same_problem_count += 1
            else:
                diff_problem_count += 1
        
        print(f"   - Same problem_id in both sets: {same_problem_count}")
        print(f"   - Different problem_ids with same path: {diff_problem_count}")
        
    else:
        print(f"   OK: No data leakage detected - train and test sets are properly separated")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training set duplicates: {train_dup_count}")
    print(f"Test set duplicates: {test_dup_count}")
    print(f"Data leakage (paths in both sets): {len(leaked_paths)}")
    
    if train_dup_count == 0 and test_dup_count == 0 and len(leaked_paths) == 0:
        print("\nOK: All checks passed! Data quality is good.")
    else:
        print("\nWARNING: Issues found that need attention!")
    print("=" * 80)
    
    return {
        'train_duplicates': train_dup_count,
        'test_duplicates': test_dup_count,
        'leaked_paths': len(leaked_paths)
    }

if __name__ == "__main__":
    train_path = r"e:\climb-path\data\moonboard_train_quality.csv"
    test_path = r"e:\climb-path\data\moonboard_test_quality.csv"
    
    results = check_duplicates_and_leakage(train_path, test_path)
