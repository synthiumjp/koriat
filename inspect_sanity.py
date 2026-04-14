"""
inspect_sanity.py — Post-sanity-run inspection checklist (v1.1)

Run this AFTER `python collect_saturation.py --sanity` completes.
Produces a per-cell summary and checks against the pre-reg v1.1 §6.5 / §6.6 rules.

Expected output: 16 cells (8 models × 2 conditions), 5 trials each.

Key checks:
  1. Parse rate per cell for the 7 instruct models (pre-reg §6.6 exclusion threshold:
     >30% parse failure = exclude from confirmatory)
  2. M1 retention (pre-reg v1.1 §6.6: ≥80% of M1 sanity trials must have
     response_length_tokens > 0; M1 is retained for E-base only, not confirmatory)
  3. M8 pilot-prediction match (pre-reg §2.1): do NUM trials parse, do CAT trials refuse?
  4. Logprob fields populated on all success trials (infrastructure check)
  5. response_length_tokens > 0 on all success trials (infrastructure check)
  6. No inference errors on the 7 instruct models
  7. M8 finish_reason check — any trials hit max_tokens=1024?

This is an infrastructure inspection, not an analysis. It does NOT compute validity
indices, AUROC, or hypothesis tests. Those are computed only after main collection.
"""

import pandas as pd
import sys
from pathlib import Path

SANITY_PARQUET = Path(r"C:\sdt_calibration\koriat_project_b\sanity_run.parquet")

if not SANITY_PARQUET.exists():
    print(f"ERROR: {SANITY_PARQUET} not found. Run `python collect_saturation.py --sanity` first.")
    sys.exit(1)

df = pd.read_parquet(SANITY_PARQUET)
print(f"Loaded {len(df)} rows from {SANITY_PARQUET}")
print(f"Expected: 80 rows (8 models × 2 conditions × 5 items)")
print()

# ------------------------------------------------------------
# 1. Per-cell summary
# ------------------------------------------------------------
print("=" * 78)
print("PER-CELL SUMMARY")
print("=" * 78)
print(f"{'Model':<5} {'Cond':<5} {'N':>3} {'Success':>8} {'ParseRate':>10} {'MeanConf':>10} {'MeanTokens':>11}")
print("-" * 78)

for (model, cond), grp in df.groupby(["model_id", "condition"], sort=False):
    n = len(grp)
    n_success = (grp["parse_status"] == "success").sum()
    parse_rate = n_success / n if n > 0 else 0.0
    mean_conf = grp.loc[grp["parse_status"] == "success", "parsed_confidence"].mean()
    mean_tokens = grp["response_length_tokens"].mean()
    print(f"{model:<5} {cond:<5} {n:>3} {n_success:>8} {parse_rate:>10.1%} {mean_conf:>10.3f} {mean_tokens:>11.1f}")

print()

# ------------------------------------------------------------
# 2. M1 retention check (pre-reg v1.1 §6.6)
# ------------------------------------------------------------
print("=" * 78)
print("CHECK 1 — M1 retention for E-base (pre-reg v1.1 §6.6)")
print("=" * 78)
m1 = df[df["model_id"] == "M1"]
m1_nonempty = (m1["response_length_tokens"] > 0).sum()
m1_n = len(m1)
m1_rate = m1_nonempty / m1_n if m1_n > 0 else 0.0
print(f"M1 non-empty-output rate: {m1_nonempty}/{m1_n} = {m1_rate:.1%}")
print("(v1.1 rule: ≥80% response_length_tokens > 0. Parse rate no longer applies to M1.)")
if m1_rate >= 0.80:
    print("  → PASS: M1 stays in main collection for E-base.")
    print("         M1 is NOT in the confirmatory sample (H1, H2, H4, H5 = 7 instruct).")
else:
    print("  → FAIL: M1 non-empty rate < 80%. Per pre-reg v1.1 §6.6, M1 is dropped.")
    print("           E-base (exploratory) cannot be computed. Document in limitations.")
print()

# ------------------------------------------------------------
# 3. M8 pilot-prediction check (pre-reg §2.1, §3.7)
# ------------------------------------------------------------
print("=" * 78)
print("CHECK 2 — M8 behavior vs pilot prediction (pre-reg §2.1)")
print("=" * 78)
m8_num = df[(df["model_id"] == "M8") & (df["condition"] == "NUM")]
m8_cat = df[(df["model_id"] == "M8") & (df["condition"] == "CAT")]
m8_num_success = (m8_num["parse_status"] == "success").sum()
m8_cat_success = (m8_cat["parse_status"] == "success").sum()
print(f"M8 NUM parse rate: {m8_num_success}/{len(m8_num)}")
print(f"M8 CAT parse rate: {m8_cat_success}/{len(m8_cat)}")
print()
print("Pilot prediction: M8 CAT will show substantially reduced parse rate vs M8 NUM.")
if m8_cat_success < m8_num_success:
    print(f"  → Consistent with pilot ({m8_cat_success} < {m8_num_success}).")
elif m8_cat_success == 0 and m8_num_success >= 3:
    print("  → Strongly consistent with pilot: CAT produces no parseable confidence.")
else:
    print(f"  → DIVERGENT from pilot: CAT ({m8_cat_success}) >= NUM ({m8_num_success}).")
    print("    This is informative. Main collection will characterize the full pattern.")
print()

# ------------------------------------------------------------
# 4. M8 finish_reason check — budget sufficiency
# ------------------------------------------------------------
print("=" * 78)
print("CHECK 3 — M8 max_tokens budget (locked at 1024 per pre-reg §3.4)")
print("=" * 78)
m8 = df[df["model_id"] == "M8"]
if "finish_reason" in m8.columns:
    finish_reasons = m8["finish_reason"].value_counts()
    print(f"M8 finish_reason distribution:")
    for fr, cnt in finish_reasons.items():
        print(f"  {fr!r}: {cnt}")
    n_length_truncated = (m8["finish_reason"] == "length").sum()
    if n_length_truncated == 0:
        print("  → PASS: No M8 trials hit max_tokens=1024. Budget is sufficient.")
    else:
        pct = n_length_truncated / len(m8)
        print(f"  → {n_length_truncated}/{len(m8)} ({pct:.0%}) M8 trials hit max_tokens=1024.")
        print("    If this rate is low (<20%) it is acceptable.")
        print("    If high, flag in main collection results as a known limitation.")
else:
    print("  WARN: finish_reason column not in parquet. Falling back to token-count proxy.")
    n_at_budget = (m8["response_length_tokens"] == 1024).sum()
    print(f"  M8 trials at exactly 1024 tokens: {n_at_budget}/{len(m8)}")
print()

# ------------------------------------------------------------
# 5. Logprob infrastructure check
# ------------------------------------------------------------
print("=" * 78)
print("CHECK 4 — Logprob fields populated on success trials")
print("=" * 78)
success_rows = df[df["parse_status"] == "success"]
n_success = len(success_rows)
n_none_logprob = success_rows["mean_logprob"].isna().sum()
n_zero_tokens = (success_rows["response_length_tokens"] == 0).sum()
print(f"Success trials: {n_success}")
print(f"Success trials with mean_logprob=None: {n_none_logprob}")
print(f"Success trials with response_length_tokens=0: {n_zero_tokens}")
if n_none_logprob == 0 and n_zero_tokens == 0:
    print("  → PASS: All success trials have populated logprobs and non-zero token counts.")
else:
    print("  → FAIL: Infrastructure bug. Inference wrapper not returning logprobs.")
    print("    DO NOT proceed to main collection until this is fixed.")
print()

# ------------------------------------------------------------
# 6. Inference error check (on the 7 instruct models)
# ------------------------------------------------------------
print("=" * 78)
print("CHECK 5 — No inference errors on the 7 instruct models (M2-M8)")
print("=" * 78)
instruct = df[df["model_id"].isin(["M2", "M3", "M4", "M5", "M6", "M7", "M8"])]
n_errors = (instruct["parse_status"] == "inference_error").sum()
if n_errors == 0:
    print("  → PASS: Zero inference errors across 70 instruct-model trials.")
else:
    print(f"  → FAIL: {n_errors} inference errors. Inspect:")
    err_rows = instruct[instruct["parse_status"] == "inference_error"]
    for _, row in err_rows.iterrows():
        print(f"    {row['model_id']} / {row['condition']} / item {row['item_index']}")
    print("  DO NOT proceed to main collection until these are diagnosed and fixed.")
print()

# ------------------------------------------------------------
# 7. Parse-rate warning on the 7 instruct models (pre-reg §6.6)
# ------------------------------------------------------------
print("=" * 78)
print("CHECK 6 — Parse-rate warning (pre-reg §6.6: >30% failure = excluded from confirmatory)")
print("=" * 78)
print("Note: The 30% exclusion rule applies to MAIN COLLECTION, not sanity.")
print("Sanity with 5 items per cell gives low statistical resolution.")
print("Cells with 0 or 1 success in sanity are flagged for attention, not exclusion.")
print("M1 is NOT subject to this check (M1 has no verbal-confidence parsing in v1.1).")
print("M8 CAT low success is expected per pilot (§2.1); not flagged.")
print()
any_flagged = False
for (model, cond), grp in df.groupby(["model_id", "condition"], sort=False):
    if model == "M1":
        continue  # M1 is retained on token-production criterion, not parse rate
    n_success = (grp["parse_status"] == "success").sum()
    if n_success <= 1 and not (model == "M8" and cond == "CAT"):
        print(f"  FLAG: {model}/{cond} has only {n_success}/5 success. Inspect raw_response values.")
        any_flagged = True
if not any_flagged:
    print("  No instruct cells flagged (M1 and M8 CAT excluded per notes above).")
print()

# ------------------------------------------------------------
# Overall verdict
# ------------------------------------------------------------
print("=" * 78)
print("OVERALL VERDICT")
print("=" * 78)

all_pass = (
    m1_rate >= 0.80
    and n_none_logprob == 0
    and n_zero_tokens == 0
    and n_errors == 0
)

if all_pass:
    print("All critical infrastructure checks PASS.")
    print("Per pre-reg §6.1 order of operations:")
    print("  1. Review the M8 CAT cell and confirm it matches the pilot pattern.")
    print("  2. Commit sanity_run.parquet and this output to OSF as supplementary material.")
    print("  3. Post pre-reg v1.1 + m8_pilot_verification.json to OSF (new project).")
    print("  4. Launch main collection.")
else:
    print("One or more critical checks FAILED. DO NOT post to OSF or launch main collection.")
    print("Fix the failing check and rerun sanity.")
