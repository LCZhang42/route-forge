import pandas as pd
import json

df = pd.read_excel('data/moonGen_scrape_2016_final.xlsx', index_col=0)

print('=' * 60)
print('MOONBOARD CLIMBING PROBLEMS DATA ANALYSIS (2016)')
print('=' * 60)

print(f'\nDATASET OVERVIEW')
print(f'   Total problems: {len(df):,}')
print(f'   Columns: {", ".join(df.columns)}')

print(f'\nGRADE DISTRIBUTION')
grade_counts = df['grade'].value_counts().sort_index()
for grade, count in grade_counts.items():
    bar = '#' * int(count / 500)
    print(f'   {grade:>4}: {count:>5} {bar}')

print(f'\nBENCHMARK STATUS')
benchmark_counts = df['is_benchmark'].value_counts()
print(f'   Benchmark problems: {benchmark_counts.get(True, 0):,} ({benchmark_counts.get(True, 0)/len(df)*100:.1f}%)')
print(f'   Regular problems: {benchmark_counts.get(False, 0):,} ({benchmark_counts.get(False, 0)/len(df)*100:.1f}%)')

print(f'\nREPEATS STATISTICS')
print(f'   Mean repeats: {df["repeats"].mean():.1f}')
print(f'   Median repeats: {df["repeats"].median():.0f}')
print(f'   Max repeats: {df["repeats"].max():,}')
print(f'   Problems with 0 repeats: {(df["repeats"] == 0).sum():,}')
print(f'   Problems with >100 repeats: {(df["repeats"] > 100).sum():,}')

print(f'\nTOP 10 MOST POPULAR PROBLEMS')
top10 = df.nlargest(10, 'repeats')[['grade', 'repeats', 'is_benchmark']]
for i, (idx, row) in enumerate(top10.iterrows(), 1):
    benchmark_tag = '[BENCHMARK]' if row['is_benchmark'] else ''
    print(f'   {i:>2}. Problem #{idx}: Grade {row["grade"]:>4} | {row["repeats"]:>5} repeats {benchmark_tag}')

print(f'\nHOLD POSITIONS')
print(f'   Unique start positions: {df["start"].nunique():,}')
print(f'   Unique mid positions: {df["mid"].nunique():,}')
print(f'   Unique end positions: {df["end"].nunique():,}')

if 'user_grade' in df.columns:
    user_graded = df['user_grade'].notna().sum()
    print(f'\nUSER GRADES')
    print(f'   Problems with user grades: {user_graded:,} ({user_graded/len(df)*100:.1f}%)')
    if user_graded > 0:
        print(f'   User grade distribution:')
        user_grade_counts = df['user_grade'].value_counts().head(10)
        for grade, count in user_grade_counts.items():
            print(f'      {grade:>4}: {count:>5}')

print(f'\nSAMPLE PROBLEM DETAILS')
sample = df.iloc[0]
print(f'   Problem ID: {df.index[0]}')
print(f'   Grade: {sample["grade"]}')
print(f'   Repeats: {sample["repeats"]}')
print(f'   Start holds: {sample["start"]}')
print(f'   End holds: {sample["end"]}')
print(f'   Benchmark: {sample["is_benchmark"]}')
if isinstance(sample["setter"], str) and sample["setter"].startswith('{'):
    setter_info = json.loads(sample["setter"].replace("'", '"'))
    print(f'   Setter: {setter_info.get("Nickname", "Unknown")} ({setter_info.get("City", "")}, {setter_info.get("Country", "")})')

print('\n' + '=' * 60)
