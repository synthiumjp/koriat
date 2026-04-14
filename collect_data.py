"""
collect_data.py
Koriat Cue-Utilisation Project B — Stage 1 Data Collection

Runs all 90 items x 2 conditions x 6 models and saves to parquet.

Conditions:
  1 = baseline (no length constraint)
  2 = length-controlled (max 25 words instruction)

Output: raw_responses.parquet
  columns: model, item_id, condition, prompt, raw_output,
           answer_parsed, confidence_parsed, parse_failure_reason,
           first_token_top1_prob, answer_mean_log_prob, correct

Usage (inference venv active):
    python collect_data.py --dry-run        # show what would run, no inference
    python collect_data.py                  # full run (~10.5 GPU hours)
    python collect_data.py --model Mistral  # single model
    python collect_data.py --resume         # skip already-collected rows
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from llama_cpp import Llama
except ImportError:
    sys.exit("llama_cpp not installed.")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path(r"C:\sdt_calibration\models")
ITEMS_FILE  = Path(r"C:\sdt_calibration\koriat_project_b\items_v5.json")
OUTPUT_FILE = Path(r"C:\sdt_calibration\koriat_project_b\raw_responses.parquet")

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "Llama-3.1-8B-Instruct",
        "prereg_name": "Llama 3.1 8B Instruct",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "repeat_penalty": 1.1,   # prevents repetition loops seen in verify
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "prereg_name": "Mistral 7B Instruct v0.3",
        "filename": "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
        "repeat_penalty": 1.0,
    },
    {
        "name": "Qwen2.5-3B-Instruct",
        "prereg_name": "Qwen 2.5 3B Instruct",
        "filename": "Qwen2.5-3B-Instruct-Q5_K_M.gguf",
        "repeat_penalty": 1.0,
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "prereg_name": "Qwen 2.5 7B Instruct",
        "filename": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        "repeat_penalty": 1.0,
    },
    {
        "name": "gemma-2-9b-it",
        "prereg_name": "Gemma 2 9B IT",
        "filename": "gemma-2-9b-it-Q5_K_M.gguf",
        "repeat_penalty": 1.0,
    },
    {
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "prereg_name": "DeepSeek R1 Distill Llama 8B",
        "filename": "DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
        "repeat_penalty": 1.0,
    },
]

# ── Prompt templates ──────────────────────────────────────────────────────────
# Condition 1: baseline
# Condition 2: length-controlled (max 25 words)
#
# Design rationale:
#   - ANSWER and CONFIDENCE fields are on separate lines for reliable regex
#   - Confidence is 0-100 integer to avoid decimal ambiguity
#   - Condition 2 adds explicit word limit on the answer only (not confidence)
#   - DeepSeek R1 is allowed to reason freely; structured fields must appear
#     at the end, which the prompt enforces with "End your response with:"

PROMPT_C1 = """Answer the following question. End your response with these two lines in exactly this format:
ANSWER: [your answer]
CONFIDENCE: [integer 0-100]

Question: {question}"""

PROMPT_C2 = """Answer the following question in 25 words or fewer. End your response with these two lines in exactly this format:
ANSWER: [your answer, max 25 words]
CONFIDENCE: [integer 0-100]

Question: {question}"""

# ── Parse functions ───────────────────────────────────────────────────────────
ANSWER_RE     = re.compile(r'ANSWER:\s*(.+?)(?:\n|$)', re.IGNORECASE)
CONFIDENCE_RE = re.compile(r'CONFIDENCE:\s*(\d+(?:\.\d+)?)', re.IGNORECASE)


def parse_response(raw_text):
    """
    Returns (answer, confidence, failure_reason).
    failure_reason is None on success, else a categorical string.
    Categories (pre-registered):
      'no_answer_field'     - ANSWER: line not found
      'no_confidence_field' - CONFIDENCE: line not found
      'confidence_out_range'- confidence not in [0, 100]
      'empty_answer'        - ANSWER field present but blank
    """
    answer, confidence, reason = None, None, None

    m_ans = ANSWER_RE.search(raw_text)
    if not m_ans:
        return None, None, "no_answer_field"
    answer = m_ans.group(1).strip()
    if not answer:
        return None, None, "empty_answer"

    m_conf = CONFIDENCE_RE.search(raw_text)
    if not m_conf:
        return answer, None, "no_confidence_field"
    confidence = float(m_conf.group(1))
    if not (0 <= confidence <= 100):
        return answer, confidence, "confidence_out_range"

    return answer, confidence, None


def compute_answer_mean_log_prob(response):
    """
    Compute mean log-probability of the answer tokens from logprobs data.
    Returns None if logprobs unavailable.
    """
    logprobs_data = response["choices"][0].get("logprobs")
    if not logprobs_data:
        return None
    token_logprobs = logprobs_data.get("token_logprobs", [])
    if not token_logprobs:
        return None
    # Filter out None values (e.g. first token sometimes None)
    valid = [lp for lp in token_logprobs if lp is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


def get_first_token_top1_prob(response):
    """
    Probability of the most likely first generated token.
    Returns None if logprobs unavailable.
    """
    logprobs_data = response["choices"][0].get("logprobs")
    if not logprobs_data:
        return None
    top_logprobs = logprobs_data.get("top_logprobs", [])
    if not top_logprobs or not top_logprobs[0]:
        return None
    max_logprob = max(top_logprobs[0].values())
    return round(math.exp(max_logprob), 6)


# ── Main collection loop ──────────────────────────────────────────────────────
def run_model(model_cfg, items, conditions, existing_keys, dry_run=False):
    """Run one model across all items and conditions. Returns list of row dicts."""
    rows = []
    path = MODELS_DIR / model_cfg["filename"]

    if not path.exists():
        print(f"  SKIP (file not found): {path}")
        return rows

    if dry_run:
        for item in items:
            for cond in conditions:
                key = (model_cfg["name"], item["item_id"], cond)
                if key not in existing_keys:
                    print(f"  [DRY RUN] would run: {model_cfg['name']} item={item['item_id']} cond={cond}")
        return rows

    print(f"  Loading {model_cfg['name']}...")
    t0 = time.time()
    llm = Llama(
        model_path=str(path),
        n_gpu_layers=-1,
        n_ctx=2048,
        logits_all=True,
        verbose=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    prompt_templates = {1: PROMPT_C1, 2: PROMPT_C2}

    for item in items:
        question = item["question"]
        for cond in conditions:
            key = (model_cfg["name"], item["item_id"], cond)
            if key in existing_keys:
                continue

            prompt = prompt_templates[cond].format(question=question)

            try:
                response = llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.0,
                    repeat_penalty=model_cfg.get("repeat_penalty", 1.0),
                    logprobs=5,
                    echo=False,
                )
                raw = response["choices"][0]["text"]
                answer, confidence, failure = parse_response(raw)
                first_prob = get_first_token_top1_prob(response)
                mean_lp    = compute_answer_mean_log_prob(response)

            except Exception as e:
                raw = ""
                answer, confidence, failure = None, None, f"inference_error: {e}"
                first_prob, mean_lp = None, None

            rows.append({
                "model":                  model_cfg["name"],
                "prereg_name":            model_cfg["prereg_name"],
                "item_id":                item["item_id"],
                "condition":              cond,
                "difficulty_tier":        item["difficulty_tier"],
                "item_type":              item["item_type"],
                "domain":                 item["domain"],
                "correct_answer":         item.get("correct_answer", None),
                "prompt":                 prompt,
                "raw_output":             raw.strip(),
                "answer_parsed":          answer,
                "confidence_parsed":      confidence,
                "parse_failure_reason":   failure,
                "first_token_top1_prob":  first_prob,
                "answer_mean_log_prob":   mean_lp,
            })

    del llm
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only one model (partial name match)")
    parser.add_argument("--conditions", type=int, nargs="+", default=[1, 2],
                        help="Conditions to run (default: 1 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already in output parquet")
    args = parser.parse_args()

    # Load items
    if not ITEMS_FILE.exists():
        sys.exit(f"items_v5.json not found at {ITEMS_FILE}\n"
                 f"Copy it there from wherever you saved it.")
    with open(ITEMS_FILE) as f:
        items_data = json.load(f)
    items = items_data["items"]
    print(f"Loaded {len(items)} items from {ITEMS_FILE}")

    # NOTE: items_v5.json has item metadata but not question text.
    # Question text must be added — see note below.
    # For now, flag if question field is missing.
    if "question" not in items[0]:
        print("\nWARNING: items_v5.json does not contain 'question' text fields.")
        print("You need to join question text before running data collection.")
        print("See add_question_text.py (to be built in next step).")
        if not args.dry_run:
            sys.exit(1)

    # Load existing rows for resume
    existing_keys = set()
    if args.resume and OUTPUT_FILE.exists():
        existing_df = pd.read_parquet(OUTPUT_FILE)
        existing_keys = set(
            zip(existing_df["model"], existing_df["item_id"], existing_df["condition"])
        )
        print(f"Resume: {len(existing_keys)} rows already collected")

    # Filter models
    models_to_run = MODELS
    if args.model:
        models_to_run = [m for m in MODELS if args.model.lower() in m["name"].lower()]
        if not models_to_run:
            sys.exit(f"No model matching '{args.model}'")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model_cfg in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {model_cfg['name']}")
        rows = run_model(
            model_cfg, items, args.conditions, existing_keys,
            dry_run=args.dry_run
        )
        all_rows.extend(rows)

        # Save incrementally after each model (crash safety)
        if rows and not args.dry_run:
            new_df = pd.DataFrame(rows)
            if OUTPUT_FILE.exists() and args.resume:
                existing_df = pd.read_parquet(OUTPUT_FILE)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined.to_parquet(OUTPUT_FILE, index=False)
            else:
                if OUTPUT_FILE.exists():
                    existing_df = pd.read_parquet(OUTPUT_FILE)
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined.to_parquet(OUTPUT_FILE, index=False)
                else:
                    new_df.to_parquet(OUTPUT_FILE, index=False)
            print(f"  Saved {len(rows)} rows -> {OUTPUT_FILE}")

    if not args.dry_run and all_rows:
        final_df = pd.read_parquet(OUTPUT_FILE)
        print(f"\nTotal rows in parquet: {len(final_df)}")
        parse_ok = final_df["parse_failure_reason"].isna().sum()
        parse_fail = final_df["parse_failure_reason"].notna().sum()
        print(f"Parse OK: {parse_ok} | Parse failures: {parse_fail}")
        if parse_fail > 0:
            print("Failure breakdown:")
            print(final_df["parse_failure_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
