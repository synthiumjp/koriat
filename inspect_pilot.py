"""
inspect_pilot.py
Quick spot-check of raw_responses.parquet after a pilot run.
Run from C:\sdt_calibration\koriat_project_b\

Usage:
    python inspect_pilot.py
"""
import pandas as pd

df = pd.read_parquet("raw_responses.parquet")

print(f"Total rows: {len(df)}")
print(f"Models: {df['model'].unique().tolist()}")
print(f"Conditions: {df['condition'].unique().tolist()}")
print(f"Parse OK: {df['parse_failure_reason'].isna().sum()}")
print(f"Parse failures: {df['parse_failure_reason'].notna().sum()}")
print()

# Confidence distribution
print("=== Confidence distribution ===")
print(df['confidence_parsed'].describe().round(1))
print()

# Sample: one easy, one hard item
print("=== Sample outputs (3 easy, 3 hard) ===")
for tier in ['easy', 'hard']:
    sample = df[df['difficulty_tier'] == tier].head(3)
    for _, row in sample.iterrows():
        print(f"\n[{tier.upper()}] item_id={row['item_id']} domain={row['domain']}")
        print(f"  Q: {row['prompt'][:120].strip()}")
        print(f"  A: {repr(row['raw_output'][:120])}")
        print(f"  answer={row['answer_parsed']}  conf={row['confidence_parsed']}")
        print(f"  logprob={row['answer_mean_log_prob']}  top1={row['first_token_top1_prob']}")

# Logprob coverage
print()
print("=== Logprob coverage ===")
print(f"first_token_top1_prob non-null: {df['first_token_top1_prob'].notna().sum()}/90")
print(f"answer_mean_log_prob non-null:  {df['answer_mean_log_prob'].notna().sum()}/90")

# Flag any suspiciously uniform confidence
conf_counts = df['confidence_parsed'].value_counts()
print()
print("=== Top confidence values (check for blanket responding) ===")
print(conf_counts.head(10).to_string())
