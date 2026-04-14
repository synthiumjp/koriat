"""
build_cues.py — Koriat cue-utilisation project (lean v5)

Computes the 6 pre-registered cues for each row in raw_responses_v1.parquet.

IMPORTANT: Per pre-reg §"Indices" (lines 488-495), surface and linguistic
cues are computed on the PARSED ANSWER text (answer_parsed column), NOT on
the full raw_output. Quoting the pre-reg:

    length_words - Whitespace-tokenized word count of the parsed answer.
    length_sentences - Sentence count of the parsed answer via NLTK sent_tokenize.
    hedging_count_core40 is the total number of regex matches against the
        40-term Core-40 hedging lexicon in the parsed answer text.

The full raw_output is retained for provenance but is not the substrate for
cue computation.

Cue families (from pre-registration, locked 13 April 2026):
  Surface (computed on answer_parsed):
    - length_words          : whitespace-split word count of answer_parsed
    - length_sentences      : NLTK sent_tokenize count of answer_parsed
  Linguistic (computed on answer_parsed):
    - hedging_density_core40       : Core-40 hedging terms / length_words
                                     (pre-reg literal: independent word-boundary
                                      matching)
    - confidence_marker_net        : (19 high - 9 low) / length_words
  Probabilistic (already in raw_responses_v1.parquet):
    - first_token_top1_prob
    - answer_mean_log_prob

Robustness column (not confirmatory):
    - hedging_density_core40_longest_match : multi-word-first matching,
                                              no double-count on "tend to"

Input  : C:\\sdt_calibration\\koriat_project_b\\raw_responses_v1.parquet
Output : C:\\sdt_calibration\\koriat_project_b\\cues.parquet

Parse failures (parse_failure_reason notna()) are PRESERVED with NaN cue
values. Exclusion logic lives in build_sdt.py.

Run:
    python build_cues.py
    python build_cues.py --input path\\to\\raw.parquet --output path\\to\\cues.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# NLTK setup: sent_tokenize needs the 'punkt' tokenizer data
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    sys.exit("nltk is required. Install with: pip install nltk")

def _ensure_punkt():
    """Download punkt if missing. Safe to call repeatedly."""
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            pass
    # Try both punkt and punkt_tab — newer NLTK uses punkt_tab
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

_ensure_punkt()

# ---------------------------------------------------------------------------
# LEXICONS — locked per pre-registration §"Indices" (lines 563-587)
# ---------------------------------------------------------------------------
# Source of truth: OSF pre-registration, locked 13 April 2026. These lists
# differ from the 14 April session log, which contained paraphrased/earlier
# versions. The pre-reg is authoritative. Full list reconstructed by direct
# reading of koriat_pre-reg.docx on 14 April 2026.

# "Core-40" hedging lexicon — pre-reg contains 39 terms, not 40.
# The label "Core-40" is retained for consistency with the pre-registration.
# This discrepancy is a pre-registration internal inconsistency (count vs
# label), not a deviation from the locked analysis: the 39 terms below are
# exactly what the pre-reg specifies.
CORE40_HEDGES: tuple[str, ...] = (
    "might", "may", "perhaps", "possibly", "maybe", "likely", "unlikely",
    "uncertain", "probably", "presumably", "seemingly", "apparently",
    "i think", "i believe", "i suppose", "i guess", "i'm not sure",
    "i am not sure", "not sure", "sort of", "kind of", "somewhat",
    "roughly", "approximately", "around", "it could be", "it may be",
    "could potentially", "may potentially", "i'm uncertain", "i am uncertain",
    "hard to say", "hard to tell", "to my knowledge", "as far as i know",
    "if i remember correctly", "if i recall correctly", "i would guess",
    "i would say",
)
assert len(CORE40_HEDGES) == 39, (
    f"Pre-reg Core-40 list has 39 terms; got {len(CORE40_HEDGES)}. "
    f"The label 'Core-40' is preserved for consistency with the pre-reg."
)

# 19 high-confidence markers (pre-reg line 501-504)
HIGH_CONF_MARKERS: tuple[str, ...] = (
    "certainly", "definitely", "clearly", "obviously", "undoubtedly",
    "surely", "indisputably", "unquestionably", "absolutely", "precisely",
    "without doubt", "no doubt", "of course", "i'm certain", "i am certain",
    "i'm confident", "i am confident", "i'm sure", "i am sure",
)
assert len(HIGH_CONF_MARKERS) == 19

# 9 low-confidence markers (pre-reg line 505-507)
LOW_CONF_MARKERS: tuple[str, ...] = (
    "doubtful", "i doubt", "i'm skeptical", "i am skeptical", "i question",
    "i wouldn't say", "i can't be sure", "i'm not confident",
    "i am not confident",
)
assert len(LOW_CONF_MARKERS) == 9


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------
# Word-boundary regex per pre-reg: r'\b(term)\b' with re.IGNORECASE.
# For multi-word terms ("tend to"), \b still works at each end and any
# internal whitespace in the input matches the literal space in the pattern
# — but robustness under multiple spaces / tabs / newlines is safer with \s+.
def _compile_term(term: str) -> re.Pattern:
    parts = [re.escape(p) for p in term.split()]
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)

CORE40_PATTERNS: dict[str, re.Pattern] = {t: _compile_term(t) for t in CORE40_HEDGES}
HIGH_CONF_PATTERNS: dict[str, re.Pattern] = {t: _compile_term(t) for t in HIGH_CONF_MARKERS}
LOW_CONF_PATTERNS: dict[str, re.Pattern] = {t: _compile_term(t) for t in LOW_CONF_MARKERS}

# For longest-match robustness variant: sort by word count descending,
# so "tend to" is attempted before "tend".
CORE40_SORTED_FOR_LONGEST = sorted(
    CORE40_HEDGES, key=lambda t: (-len(t.split()), -len(t))
)


# ---------------------------------------------------------------------------
# Cue computation
# ---------------------------------------------------------------------------
def count_words(text: str) -> int:
    """Whitespace-split word count. Matches pre-reg 'word count of raw_output'."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """NLTK sent_tokenize count per pre-reg."""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        return len(sent_tokenize(text))
    except Exception:
        # Fallback: period-split if punkt fails for any reason.
        return max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))


def count_core40_literal(text: str) -> int:
    """
    Pre-reg literal: independent word-boundary matching of each Core-40 term.
    'tend to' in the input contributes 2 matches (one for 'tend', one for 'tend to').
    This is the CONFIRMATORY column for H2.
    """
    if not isinstance(text, str):
        return 0
    return sum(len(p.findall(text)) for p in CORE40_PATTERNS.values())


def count_core40_longest_match(text: str) -> int:
    """
    Robustness variant: longest-match-first with span masking so each character
    can contribute to at most one term. 'tend to' counts once, not twice.
    Reported as a Methods robustness column, NOT used for confirmatory H2.
    """
    if not isinstance(text, str):
        return 0
    # Work on a mutable char list and blank out matched spans so single-word
    # terms cannot re-match inside an already-matched multi-word span.
    chars = list(text)
    total = 0
    for term in CORE40_SORTED_FOR_LONGEST:
        pat = CORE40_PATTERNS[term]
        current = "".join(chars)
        for m in pat.finditer(current):
            start, end = m.span()
            # Skip if this span has been blanked out in a prior (longer-term) pass.
            # We check the *original* working copy via the blanked chars array,
            # but because we rebuild `current` from `chars` at the top of each
            # term iteration, any prior blanking is already reflected.
            total += 1
            for i in range(start, end):
                chars[i] = "\x00"
    return total


def count_high_conf(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(len(p.findall(text)) for p in HIGH_CONF_PATTERNS.values())


def count_low_conf(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(len(p.findall(text)) for p in LOW_CONF_PATTERNS.values())


def compute_row_cues(answer_parsed: str) -> dict:
    """
    Compute all cues for a single parsed-answer string. Returns dict of cue
    name -> value.

    Per pre-reg §"Indices":
      - hedging_density_core40 = hedging_count / max(length_words, 1)
      - confidence_marker_net = high_count - low_count  (RAW, not normalized)
    """
    import math
    wc = count_words(answer_parsed)
    sc = count_sentences(answer_parsed)
    hedges_literal = count_core40_literal(answer_parsed)
    hedges_longest = count_core40_longest_match(answer_parsed)
    high = count_high_conf(answer_parsed)
    low = count_low_conf(answer_parsed)

    # Hedging density: normalized, with max(wc, 1) denominator per pre-reg
    denom = max(wc, 1)
    hedging_density = hedges_literal / denom
    hedging_density_lm = hedges_longest / denom

    # Confidence marker net: RAW count per pre-reg, NOT normalized
    conf_marker_net = high - low

    return {
        "length_words": wc,
        "length_sentences": sc,
        "hedging_density_core40": hedging_density,
        "hedging_density_core40_longest_match": hedging_density_lm,
        "confidence_marker_net": conf_marker_net,
        # Raw counts retained for audit / sensitivity
        "_hedges_core40_count_literal": hedges_literal,
        "_hedges_core40_count_longest_match": hedges_longest,
        "_high_conf_count": high,
        "_low_conf_count": low,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_cues(input_path: Path, output_path: Path) -> pd.DataFrame:
    print(f"[build_cues] Loading {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[build_cues] {len(df)} rows loaded")

    # Sanity-check expected columns. Per pre-reg, cues are computed on
    # answer_parsed, not raw_output.
    required = {"item_id", "model", "condition", "answer_parsed"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"[build_cues] FATAL: input parquet missing columns: {missing}. "
            f"This pipeline requires 'answer_parsed' (the parsed ANSWER field "
            f"contents). Make sure you are running on the v1 (pre-reg-faithful) "
            f"collection output, not v0."
        )

    # Probabilistic cues should already be present — warn if not
    for prob_col in ("first_token_top1_prob", "answer_mean_log_prob"):
        if prob_col not in df.columns:
            print(f"[build_cues] WARNING: {prob_col} not in input parquet")

    # Parse failures: preserve rows, emit NaN cues so exclusion logic lives
    # in build_sdt.py
    has_parse_fail_col = "parse_failure_reason" in df.columns
    if has_parse_fail_col:
        n_failed = df["parse_failure_reason"].notna().sum()
        print(f"[build_cues] {n_failed} rows have parse_failure_reason set (cues -> NaN)")

    cue_records = []
    for i, row in df.iterrows():
        # A row is excluded from cue computation if parsing failed OR if
        # answer_parsed is NaN/None for any reason.
        if has_parse_fail_col and pd.notna(row.get("parse_failure_reason")):
            cue_records.append(_nan_cue_record())
            continue
        answer = row.get("answer_parsed")
        if not isinstance(answer, str) or not answer.strip():
            cue_records.append(_nan_cue_record())
            continue
        cue_records.append(compute_row_cues(answer))

    cues_df = pd.DataFrame(cue_records, index=df.index)

    # Assemble output: keys + existing probabilistic cues + new cues
    keep_cols = ["item_id", "model", "condition"]
    for opt_col in ("difficulty_tier", "answer_parsed", "confidence_parsed",
                    "correct_answer", "parse_failure_reason",
                    "first_token_top1_prob", "answer_mean_log_prob"):
        if opt_col in df.columns:
            keep_cols.append(opt_col)

    out = pd.concat([df[keep_cols].reset_index(drop=True),
                     cues_df.reset_index(drop=True)], axis=1)

    print(f"[build_cues] Writing {output_path}")
    out.to_parquet(output_path, index=False)
    print(f"[build_cues] Done. {len(out)} rows, {len(out.columns)} columns.")

    # Quick per-model summary so you can eyeball it immediately
    print("\n[build_cues] Per-model summary (non-failed rows):")
    summary_cols = ["length_words", "length_sentences",
                    "hedging_density_core40", "confidence_marker_net"]
    available = [c for c in summary_cols if c in out.columns]
    if "parse_failure_reason" in out.columns:
        clean = out[out["parse_failure_reason"].isna()]
    else:
        clean = out
    if len(clean) > 0:
        print(clean.groupby(["model", "condition"])[available].mean().round(3))
    return out


def _nan_cue_record() -> dict:
    return {
        "length_words": pd.NA,
        "length_sentences": pd.NA,
        "hedging_density_core40": pd.NA,
        "hedging_density_core40_longest_match": pd.NA,
        "confidence_marker_net": pd.NA,
        "_hedges_core40_count_literal": pd.NA,
        "_hedges_core40_count_longest_match": pd.NA,
        "_high_conf_count": pd.NA,
        "_low_conf_count": pd.NA,
    }


def main():
    default_in = Path(r"C:\sdt_calibration\koriat_project_b\raw_responses_v1.parquet")
    default_out = Path(r"C:\sdt_calibration\koriat_project_b\cues.parquet")
    ap = argparse.ArgumentParser(description="Compute pre-registered Koriat cue features")
    ap.add_argument("--input", type=Path, default=default_in,
                    help="raw_responses_v1.parquet path")
    ap.add_argument("--output", type=Path, default=default_out,
                    help="cues.parquet output path")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"[build_cues] FATAL: input not found: {args.input}")
    build_cues(args.input, args.output)


if __name__ == "__main__":
    main()
