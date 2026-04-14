"""
collect_data_v2.py — Koriat cue-utilisation project (lean v5)

Pre-reg-faithful data collection. Replaces the initial collect_data.py which
was retired after a debugging step identified systemic departures from the
locked pre-registered design (raw-completion prompting without chat templates,
wrong max_tokens, missing top_p/seed, asymmetric repeat_penalty, permissive
format instructions). See koriat_session_log_2026-04-14.md for the debugging
audit trail.

This script:
  1. Uses inference_engine_koriat.KoriatInferenceEngine for all six models
     (Llama 3.1, Mistral v0.3, Qwen 2.5 3B, Qwen 2.5 7B, Gemma 2 9B IT,
     DeepSeek R1 Distill Llama 8B).
  2. Builds system + user prompts per pre-registered exact wording.
  3. Applies each model's native chat template (Llama 3 / Mistral / ChatML /
     Gemma 2 turn format as appropriate).
  4. Uses pre-reg-locked inference parameters throughout.
  5. Writes to raw_responses_v1.parquet. Does NOT touch the v0 parquet.

Parsing logic (ANSWER / CONFIDENCE regex) is retained from the original
collect_data.py unchanged.

Usage:
    python collect_data_v2.py --sanity-check              # 5 items on Gemma only
    python collect_data_v2.py --sanity-check --model gemma
    python collect_data_v2.py                             # full run
    python collect_data_v2.py --resume                    # skip already collected
    python collect_data_v2.py --model Llama               # single model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from inference_engine_koriat import (
    KORIAT_MODEL_CONFIGS,
    KORIAT_MODEL_ORDER,
    KoriatInferenceEngine,
    SYSTEM_C1,
    SYSTEM_C2,
)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(r"C:\sdt_calibration\koriat_project_b")
MODELS_DIR = Path(r"C:\sdt_calibration\models")
ITEMS_FILE = PROJECT_DIR / "items_v5.json"
OUTPUT_FILE = PROJECT_DIR / "raw_responses_v1.parquet"


# ----------------------------------------------------------------------------
# Response parsing — unchanged from original collect_data.py
# ----------------------------------------------------------------------------
ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_response(raw_text: str):
    """
    Returns (answer, confidence, failure_reason).
    Categories match pre-reg §"Data inclusion and exclusion":
      'no_answer_field', 'empty_answer', 'no_confidence_field',
      'confidence_non_numeric', 'confidence_out_of_range'
    """
    m_ans = ANSWER_RE.search(raw_text)
    if not m_ans:
        return None, None, "no_answer_field"
    answer = m_ans.group(1).strip()
    if not answer:
        return None, None, "empty_answer"

    m_conf = CONFIDENCE_RE.search(raw_text)
    if not m_conf:
        return answer, None, "no_confidence_field"
    try:
        confidence = float(m_conf.group(1))
    except ValueError:
        return answer, None, "confidence_non_numeric"
    if not (0 <= confidence <= 100):
        return answer, confidence, "confidence_out_of_range"
    return answer, confidence, None


# ----------------------------------------------------------------------------
# Item loading
# ----------------------------------------------------------------------------
def load_items(items_file: Path) -> list:
    if not items_file.exists():
        sys.exit(f"items_v5.json not found at {items_file}")
    with open(items_file) as f:
        data = json.load(f)
    items = data.get("items", data)  # accept either flat list or {items: [...]}
    if not items:
        sys.exit("No items loaded")
    if "question" not in items[0]:
        sys.exit(
            "items_v5.json does not contain 'question' field. "
            "Enrichment step was not completed."
        )
    return items


# ----------------------------------------------------------------------------
# Resume support
# ----------------------------------------------------------------------------
def load_existing_keys(output_file: Path) -> set:
    if not output_file.exists():
        return set()
    df = pd.read_parquet(output_file)
    return set(zip(df["model"], df["item_id"], df["condition"]))


def save_rows(new_rows: list, output_file: Path):
    """Append new rows to the output parquet atomically."""
    new_df = pd.DataFrame(new_rows)
    if output_file.exists():
        existing_df = pd.read_parquet(output_file)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False)


# ----------------------------------------------------------------------------
# Per-model collection
# ----------------------------------------------------------------------------
def run_model(
    model_key: str,
    items: list,
    conditions: list,
    existing_keys: set,
    sanity_check: bool = False,
    sanity_n: int = 5,
) -> list:
    """Run one model across all (item x condition) pairs. Returns list of row dicts."""
    config = KORIAT_MODEL_CONFIGS[model_key]
    path = MODELS_DIR / config["filename"]
    if not path.exists():
        print(f"  SKIP (file not found): {path}")
        return []

    print(f"\n{'=' * 60}")
    print(f"Model: {config['short_name']}")
    engine = KoriatInferenceEngine(model_key, models_dir=str(MODELS_DIR))

    rows = []
    items_to_run = items[:sanity_n] if sanity_check else items
    for item in items_to_run:
        question = item["question"]
        for cond in conditions:
            key = (config["short_name"], item["item_id"], cond)
            if key in existing_keys:
                continue
            try:
                result = engine.generate(question=question, condition=cond)
                answer, confidence, failure = parse_response(result["raw_output"])
            except Exception as e:
                result = {
                    "prompt": "",
                    "system_text": "",
                    "user_text": "",
                    "raw_output": "",
                    "first_token_top1_prob": None,
                    "answer_mean_log_prob": None,
                    "generation_time_s": 0.0,
                }
                answer, confidence, failure = None, None, f"inference_error: {e}"

            rows.append({
                "model": config["short_name"],
                "prereg_name": config["prereg_name"],
                "item_id": item["item_id"],
                "condition": cond,
                "difficulty_tier": item.get("difficulty_tier"),
                "item_type": item.get("item_type"),
                "domain": item.get("domain"),
                "correct_answer": item.get("correct_answer"),
                "system_text": result["system_text"],
                "user_text": result["user_text"],
                "raw_output": result["raw_output"].strip(),
                "answer_parsed": answer,
                "confidence_parsed": confidence,
                "parse_failure_reason": failure,
                "first_token_top1_prob": result["first_token_top1_prob"],
                "answer_mean_log_prob": result["answer_mean_log_prob"],
                "generation_time_s": result["generation_time_s"],
            })

    engine.unload()
    return rows


# ----------------------------------------------------------------------------
# Sanity-check inspector
# ----------------------------------------------------------------------------
def print_sanity_summary(rows: list):
    """Print a detailed view of sanity-check rows so a human can eyeball them."""
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)
    for r in rows:
        print(f"\n--- {r['model']} | item {r['item_id']} | C{r['condition']} ---")
        print(f"Parse: {r['parse_failure_reason'] or 'OK'}")
        print(f"Answer: {r['answer_parsed']!r}")
        print(f"Confidence: {r['confidence_parsed']}")
        print(f"Raw output length: {len(r['raw_output'])} chars")
        print("Raw output:")
        print(r["raw_output"])
    print("\n" + "=" * 60)
    print("END SANITY CHECK")
    print("=" * 60)
    print("\nCheck:")
    print("  1. Does each raw_output look like natural model voice (not scaffold-echo)?")
    print("  2. Is length varied across items (not all 28-30 chars)?")
    print("  3. Is the ANSWER field correctly populated?")
    print("  4. Is the CONFIDENCE field an integer in [0, 100]?")
    print("  5. Does C2 output appear meaningfully shorter than C1?")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-check", action="store_true",
                    help="Run 5 items on Gemma only (or --model) and print outputs")
    ap.add_argument("--model", type=str, default=None,
                    help="Run only one model (partial name match on short_name)")
    ap.add_argument("--conditions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    items = load_items(ITEMS_FILE)
    print(f"Loaded {len(items)} items from {ITEMS_FILE}")

    if args.resume and not args.sanity_check:
        existing_keys = load_existing_keys(OUTPUT_FILE)
        print(f"Resume: {len(existing_keys)} rows already in {OUTPUT_FILE.name}")
    else:
        existing_keys = set()

    # Sanity check: default to Gemma (the model that was broken in v0)
    if args.sanity_check:
        if args.model is None:
            model_keys = ["gemma2_instruct"]
        else:
            model_keys = [
                k for k in KORIAT_MODEL_ORDER
                if args.model.lower() in KORIAT_MODEL_CONFIGS[k]["short_name"].lower()
            ]
        if not model_keys:
            sys.exit(f"No model matching '{args.model}'")
        all_rows = []
        for key in model_keys:
            rows = run_model(
                key, items, args.conditions, existing_keys=set(),
                sanity_check=True, sanity_n=5,
            )
            all_rows.extend(rows)
        print_sanity_summary(all_rows)
        print("\nSanity check complete. No data written to parquet.")
        print("Review the output above. If it looks good, run:")
        print("    python collect_data_v2.py")
        return

    # Full run
    if args.model:
        model_keys = [
            k for k in KORIAT_MODEL_ORDER
            if args.model.lower() in KORIAT_MODEL_CONFIGS[k]["short_name"].lower()
        ]
        if not model_keys:
            sys.exit(f"No model matching '{args.model}'")
    else:
        model_keys = KORIAT_MODEL_ORDER

    run_start = time.time()
    for model_key in model_keys:
        t0 = time.time()
        rows = run_model(model_key, items, args.conditions, existing_keys)
        if rows:
            save_rows(rows, OUTPUT_FILE)
            n_fail = sum(1 for r in rows if r["parse_failure_reason"])
            print(f"  Saved {len(rows)} rows "
                  f"({n_fail} parse failures, {100 * n_fail / len(rows):.1f}%) "
                  f"-> {OUTPUT_FILE.name}")
        print(f"  Elapsed this model: {(time.time() - t0) / 60:.1f} min")

    total_min = (time.time() - run_start) / 60
    print(f"\nTotal runtime: {total_min:.1f} min")

    # Final summary
    if OUTPUT_FILE.exists():
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"\nTotal rows in {OUTPUT_FILE.name}: {len(df)}")
        print(f"Parse OK: {df['parse_failure_reason'].isna().sum()} | "
              f"Parse failures: {df['parse_failure_reason'].notna().sum()}")
        print("\nPer-model parse failure rates:")
        fail_rates = df.groupby(["model", "condition"]).parse_failure_reason.apply(
            lambda s: f"{s.notna().sum()}/{len(s)} ({100 * s.notna().mean():.1f}%)"
        )
        print(fail_rates.to_string())


if __name__ == "__main__":
    main()
