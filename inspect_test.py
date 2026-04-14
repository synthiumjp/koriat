# inspect_test.py
import pandas as pd
df = pd.read_parquet(r"C:\sdt_calibration\koriat_project_b\test_run.parquet")
row = df.iloc[0]

print("=== TOP-LINE ===")
print(f"model:          {row['model_id']}")
print(f"condition:      {row['condition']}")
print(f"parse_status:   {row['parse_status']}")
print(f"parsed_answer:  {row['parsed_answer']!r}")
print(f"parsed_conf:    {row['parsed_confidence']}")
print(f"correct:        {row['correct']}")
print()
print("=== THE IMPORTANT BITS ===")
print(f"response_length_tokens:       {row['response_length_tokens']}")
print(f"response_length_chars:        {row['response_length_chars']}")
print(f"mean_logprob:                 {row['mean_logprob']}")
print(f"sum_logprob:                  {row['sum_logprob']}")
print(f"min_logprob:                  {row['min_logprob']}")
print(f"length_normalised_logprob:    {row['length_normalised_logprob']}")
print()
print("=== RAW RESPONSE ===")
print(repr(row['raw_response'][:500]))