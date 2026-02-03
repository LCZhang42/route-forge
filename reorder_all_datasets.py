import pandas as pd
import ast
import os
from pathlib import Path

def reorder_path_by_y(path_str):
    """Reorder path holds by y-coordinate (ascending - bottom to top)"""
    try:
        path = ast.literal_eval(path_str)
        # Sort by y-coordinate (row number)
        sorted_path = sorted(path, key=lambda hold: hold[1])
        return str(sorted_path)
    except:
        # If parsing fails, return original
        return path_str

def process_csv_file(input_path, output_path):
    """Process a CSV file and reorder paths by y-coordinate"""
    print(f'\nProcessing: {input_path}')
    
    # Read the CSV
    df = pd.read_csv(input_path)
    
    # Check if the file has path columns
    path_columns = []
    if 'full_path' in df.columns:
        path_columns.append('full_path')
    if 'start_holds' in df.columns:
        path_columns.append('start_holds')
    if 'mid_holds' in df.columns:
        path_columns.append('mid_holds')
    if 'end_holds' in df.columns:
        path_columns.append('end_holds')
    
    if not path_columns:
        print(f'  ⚠ No path columns found, skipping')
        return False
    
    # Reorder each path column
    for col in path_columns:
        print(f'  Reordering column: {col}')
        df[col] = df[col].apply(reorder_path_by_y)
    
    # Save the reordered dataset
    df.to_csv(output_path, index=False)
    print(f'  ✓ Saved to: {output_path}')
    print(f'  Total rows: {len(df)}')
    
    return True

# Main processing
data_dir = Path('data')
output_dir = Path('data_reordered')

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# List of CSV files to process (excluding quality_weights.csv)
csv_files = [
    'moonboard_cleaned.csv',
    'moonboard_test.csv',
    'moonboard_test_benchmark.csv',
    'moonboard_test_quality.csv',
    'moonboard_train.csv',
    'moonboard_train_benchmark.csv',
    'moonboard_train_quality.csv',
    'moonboard_val.csv',
    'moonboard_val_benchmark.csv',
    'moonboard_val_quality.csv'
]

print('=' * 70)
print('Reordering all datasets by Y-coordinate')
print('=' * 70)

processed_count = 0
skipped_count = 0

for csv_file in csv_files:
    input_path = data_dir / csv_file
    output_path = output_dir / csv_file
    
    if input_path.exists():
        if process_csv_file(input_path, output_path):
            processed_count += 1
        else:
            skipped_count += 1
    else:
        print(f'\n⚠ File not found: {input_path}')
        skipped_count += 1

print('\n' + '=' * 70)
print(f'Processing complete!')
print(f'  Processed: {processed_count} files')
print(f'  Skipped: {skipped_count} files')
print(f'  Output directory: {output_dir}')
print('=' * 70)

# Show a sample comparison
print('\nSample comparison (first row of moonboard_train_quality.csv):')
if (data_dir / 'moonboard_train_quality.csv').exists():
    df_original = pd.read_csv(data_dir / 'moonboard_train_quality.csv', nrows=1)
    df_reordered = pd.read_csv(output_dir / 'moonboard_train_quality.csv', nrows=1)
    
    print(f'\nOriginal full_path:')
    print(f'  {df_original["full_path"].iloc[0]}')
    print(f'\nReordered full_path:')
    print(f'  {df_reordered["full_path"].iloc[0]}')
