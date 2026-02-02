import pandas as pd
import ast
import json
import numpy as np

print('=' * 70)
print('DATA CLEANING SCRIPT FOR CLIMB PATH GENERATION MODEL')
print('=' * 70)

df = pd.read_excel('data/moonGen_scrape_2016_final.xlsx', index_col=0)
print(f'\nOriginal dataset: {len(df)} problems')

print('\n1. PARSING HOLD COORDINATES')
print('-' * 70)

def parse_holds(hold_string):
    """Parse hold coordinate string into list of [x, y] coordinates"""
    try:
        return ast.literal_eval(hold_string)
    except:
        return None

df['start_parsed'] = df['start'].apply(parse_holds)
df['mid_parsed'] = df['mid'].apply(parse_holds)
df['end_parsed'] = df['end'].apply(parse_holds)

invalid_start = df['start_parsed'].isna().sum()
invalid_mid = df['mid_parsed'].isna().sum()
invalid_end = df['end_parsed'].isna().sum()

print(f'   Invalid start positions: {invalid_start}')
print(f'   Invalid mid positions: {invalid_mid}')
print(f'   Invalid end positions: {invalid_end}')

df_clean = df.dropna(subset=['start_parsed', 'mid_parsed', 'end_parsed'])
print(f'   After removing invalid holds: {len(df_clean)} problems')

print('\n2. CREATING FULL PATH SEQUENCE')
print('-' * 70)

def create_path_sequence(row):
    """Combine start, mid, end into single ordered path, sorted by climbing sequence"""
    all_holds = row['start_parsed'] + row['mid_parsed'] + row['end_parsed']
    # Sort by Y-coordinate (ascending) for natural climbing progression (bottom to top)
    # Secondary sort by X-coordinate for consistency when holds are at same height
    sorted_holds = sorted(all_holds, key=lambda hold: (hold[1], hold[0]))
    return sorted_holds

df_clean['full_path'] = df_clean.apply(create_path_sequence, axis=1)
df_clean['path_length'] = df_clean['full_path'].apply(len)

print(f'   Path length statistics:')
print(f'   - Min: {df_clean["path_length"].min()} holds')
print(f'   - Max: {df_clean["path_length"].max()} holds')
print(f'   - Mean: {df_clean["path_length"].mean():.1f} holds')
print(f'   - Median: {df_clean["path_length"].median():.0f} holds')

print('\n3. GRADE ENCODING')
print('-' * 70)

grade_order = ['6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']
df_clean['grade_numeric'] = df_clean['grade'].apply(lambda x: grade_order.index(x) if x in grade_order else -1)

invalid_grades = (df_clean['grade_numeric'] == -1).sum()
print(f'   Invalid grades: {invalid_grades}')
df_clean = df_clean[df_clean['grade_numeric'] != -1]
print(f'   After removing invalid grades: {len(df_clean)} problems')

print('\n   Grade distribution:')
for grade in grade_order:
    count = (df_clean['grade'] == grade).sum()
    if count > 0:
        print(f'   {grade:>4}: {count:>6} problems')

print('\n4. QUALITY FILTERING')
print('-' * 70)

print(f'   Original: {len(df_clean)} problems')

df_quality = df_clean.copy()
print(f'   Benchmark problems: {df_quality["is_benchmark"].sum()}')
print(f'   Non-benchmark problems: {(~df_quality["is_benchmark"]).sum()}')

df_quality['quality_score'] = df_quality['repeats']
print(f'\n   Quality score (repeats) distribution:')
print(f'   - 0 repeats: {(df_quality["repeats"] == 0).sum()} problems')
print(f'   - 1-10 repeats: {((df_quality["repeats"] >= 1) & (df_quality["repeats"] <= 10)).sum()} problems')
print(f'   - 11-50 repeats: {((df_quality["repeats"] >= 11) & (df_quality["repeats"] <= 50)).sum()} problems')
print(f'   - 51-100 repeats: {((df_quality["repeats"] >= 51) & (df_quality["repeats"] <= 100)).sum()} problems')
print(f'   - >100 repeats: {(df_quality["repeats"] > 100).sum()} problems')

min_repeats = 1
df_quality = df_quality[df_quality['repeats'] >= min_repeats]
print(f'\n   After filtering (min {min_repeats} repeats): {len(df_quality)} problems')

print('\n5. COORDINATE VALIDATION')
print('-' * 70)

def validate_coordinates(path, max_x=10, max_y=17):
    """Check if all coordinates are within valid MoonBoard range"""
    for hold in path:
        if len(hold) != 2:
            return False
        x, y = hold
        if not (0 <= x <= max_x and 1 <= y <= max_y):
            return False
    return True

df_quality['valid_coords'] = df_quality['full_path'].apply(validate_coordinates)
invalid_coords = (~df_quality['valid_coords']).sum()
print(f'   Invalid coordinates: {invalid_coords}')
df_quality = df_quality[df_quality['valid_coords']]
print(f'   After coordinate validation: {len(df_quality)} problems')

print('\n6. CREATING TRAINING DATASET')
print('-' * 70)

training_data = pd.DataFrame({
    'problem_id': df_quality.index,
    'grade': df_quality['grade'],
    'grade_numeric': df_quality['grade_numeric'],
    'start_holds': df_quality['start_parsed'],
    'mid_holds': df_quality['mid_parsed'],
    'end_holds': df_quality['end_parsed'],
    'full_path': df_quality['full_path'],
    'path_length': df_quality['path_length'],
    'is_benchmark': df_quality['is_benchmark'],
    'repeats': df_quality['repeats'],
    'quality_score': df_quality['quality_score']
})

print(f'   Final training dataset: {len(training_data)} problems')
print(f'   Features: {list(training_data.columns)}')

print('\n7. SAVING CLEANED DATA')
print('-' * 70)

output_csv = 'data/moonboard_cleaned.csv'
training_data.to_csv(output_csv, index=False)
print(f'   Saved to: {output_csv}')

output_json = 'data/moonboard_cleaned.json'
training_data_json = training_data.to_dict(orient='records')
with open(output_json, 'w') as f:
    json.dump(training_data_json, f, indent=2)
print(f'   Saved to: {output_json}')

print('\n8. CREATING TEXT FORMAT FOR LLM FINE-TUNING')
print('-' * 70)

def format_for_llm(row):
    """Format climb data as text for LLM training"""
    prompt = f"Generate a MoonBoard climb path for grade {row['grade']}."
    
    path_str = ' -> '.join([f"[{h[0]},{h[1]}]" for h in row['full_path']])
    response = f"Start: {row['start_holds']}\nPath: {path_str}\nEnd: {row['end_holds']}"
    
    return {
        'prompt': prompt,
        'response': response,
        'grade': row['grade'],
        'path_length': row['path_length']
    }

llm_data = training_data.apply(format_for_llm, axis=1).tolist()

output_llm = 'data/moonboard_llm_format.jsonl'
with open(output_llm, 'w') as f:
    for item in llm_data:
        f.write(json.dumps(item) + '\n')
print(f'   Saved LLM format to: {output_llm}')
print(f'   Format: JSONL with prompt/response pairs')

print('\n9. TRAIN/VAL/TEST SPLIT')
print('-' * 70)

from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(training_data, test_size=0.2, random_state=42, stratify=training_data['grade'])
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['grade'])

print(f'   Train set: {len(train_data)} problems ({len(train_data)/len(training_data)*100:.1f}%)')
print(f'   Val set:   {len(val_data)} problems ({len(val_data)/len(training_data)*100:.1f}%)')
print(f'   Test set:  {len(test_data)} problems ({len(test_data)/len(training_data)*100:.1f}%)')

train_data.to_csv('data/moonboard_train.csv', index=False)
val_data.to_csv('data/moonboard_val.csv', index=False)
test_data.to_csv('data/moonboard_test.csv', index=False)

print(f'\n   Saved splits:')
print(f'   - data/moonboard_train.csv')
print(f'   - data/moonboard_val.csv')
print(f'   - data/moonboard_test.csv')

print('\n10. DATA SUMMARY')
print('-' * 70)
print(f'   Original problems: {len(df):,}')
print(f'   Cleaned problems: {len(training_data):,}')
print(f'   Removed: {len(df) - len(training_data):,} ({(len(df) - len(training_data))/len(df)*100:.1f}%)')
print(f'\n   Grade range: {training_data["grade"].min()} to {training_data["grade"].max()}')
print(f'   Path length range: {training_data["path_length"].min()} to {training_data["path_length"].max()} holds')
print(f'   MoonBoard grid: 11 x 18 (coordinates: x=0-10, y=1-17)')

print('\n' + '=' * 70)
print('CLEANING COMPLETE!')
print('=' * 70)
