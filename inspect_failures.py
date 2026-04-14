import pandas as pd
df = pd.read_parquet(r"C:\sdt_calibration\koriat_project_b\sanity_run.parquet")

print("=" * 70)
print("M4 FAILURES (expecting inference_error with empty raw_response)")
print("=" * 70)
m4 = df[df["model_id"] == "M4"]
for _, row in m4.iterrows():
    print(f"\n--- M4 {row['condition']} item {row['item_index']} ---")
    print(f"parse_status: {row['parse_status']}")
    print(f"response_length_tokens: {row['response_length_tokens']}")
    print(f"raw_response: {row['raw_response']!r}")

print()
print("=" * 70)
print("M1 FAILURES (expecting degenerate_loop)")
print("=" * 70)
m1 = df[df["model_id"] == "M1"]
for _, row in m1.iterrows():
    print(f"\n--- M1 {row['condition']} item {row['item_index']} ---")
    print(f"parse_status: {row['parse_status']}")
    print(f"response_length_tokens: {row['response_length_tokens']}")
    print(f"parsed_answer: {row['parsed_answer']!r}")
    print(f"raw_response (first 400 chars): {row['raw_response'][:400]!r}")

print()
print("=" * 70)
print("M3 CAT FAILURES (expecting no_confidence_field on 2 of 5)")
print("=" * 70)
m3cat = df[(df["model_id"] == "M3") & (df["condition"] == "CAT")]
for _, row in m3cat.iterrows():
    print(f"\n--- M3 CAT item {row['item_index']} ---")
    print(f"parse_status: {row['parse_status']}")
    print(f"parsed_answer: {row['parsed_answer']!r}")
    print(f"parsed_confidence: {row['parsed_confidence']}")
    print(f"raw_response (first 400 chars): {row['raw_response'][:400]!r}")