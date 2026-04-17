"""
analyze_saturation.py — Pre-registered analysis pipeline for the Saturation Study

Implements the locked §4 analysis plan from saturation_prereg_v1_2.md exactly.
All confirmatory analyses operate on raw_responses.parquet from main collection.
Sanity data is disjoint and never used for confirmatory inference.

Analyses:
    A1  — Confidence distribution per cell (16-row table)
    A2  — Validity screen classification per cell (VRS table)
    A3  — Type-2 AUROC₂ per cell with bootstrap CI
    A4  — Format shift effect (H4 test)
    A5  — (retired in v1.1, placeholder)
    A6  — Logprob-confidence concordance (H5 test, ridge R²_CV)
    A7  — Worst-case interface composite (descriptive)
    A8  — H1 sensitivity analysis (descriptive)

Confirmatory hypotheses: H1, H2, H4, H5
Exploratory: E-base, E1–E10, E5

Author: JP Cacioli
Date: April 2026

Usage:
    python analyze_saturation.py                    # Full pipeline
    python analyze_saturation.py --analysis A1      # Single analysis
    python analyze_saturation.py --confirmatory     # H1, H2, H4, H5 only
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr, ks_2samp, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path("raw_responses.parquet")
OUTPUT_DIR = Path("analysis_outputs")
SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 10_000
RIDGE_CV_SEED = 42
BINARISE_THRESHOLD = 0.50
INSTRUCT_MODELS = ["M2", "M3", "M4", "M5", "M6", "M7", "M8"]
ALL_MODELS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
CONDITIONS = ["NUM", "CAT"]

# Validity thresholds from Cacioli (2026d/e)
L_THRESHOLD = 0.95
FP_THRESHOLD = 0.50
RBS_THRESHOLD = 0.0

# Exclusion threshold from §6.6
PARSE_FAILURE_EXCLUSION = 0.30

# H1 threshold
H1_CEILING_THRESHOLD = 0.95
H1_PREVALENCE_THRESHOLD = 0.60

# H2 threshold
H2_COUNT_THRESHOLD = 4

# H4 threshold
H4_RESCUE_THRESHOLD = 2

# H5 threshold
H5_R2_THRESHOLD = 0.20


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load main collection parquet. Never loads sanity data."""
    df = pd.read_parquet(path)
    assert len(df) == 8384, f"Expected 8384 rows, got {len(df)}"
    return df


def success_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return only parse_status == 'success' rows."""
    return df[df["parse_status"] == "success"].copy()


def cell_key(model_id: str, condition: str) -> str:
    return f"{model_id}_{condition}"


# ============================================================================
# HELPER: Wilson score CI for a proportion
# ============================================================================

def wilson_ci(successes: int, n: int, alpha: float = 0.05):
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, (0.0, 1.0)
    p_hat = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    lo = max(0, centre - spread)
    hi = min(1, centre + spread)
    return p_hat, (lo, hi)


# ============================================================================
# A1 — CONFIDENCE DISTRIBUTION PER CELL
# ============================================================================

def run_a1(df: pd.DataFrame) -> pd.DataFrame:
    """A1: Confidence distribution per cell. 16-row table."""
    rows = []
    for model in ALL_MODELS:
        for cond in CONDITIONS:
            cell = df[(df["model_id"] == model) & (df["condition"] == cond)]
            n_total = len(cell)
            n_success = len(cell[cell["parse_status"] == "success"])
            parse_rate = n_success / n_total if n_total > 0 else 0.0

            succ = cell[cell["parse_status"] == "success"]
            conf = succ["parsed_confidence"]

            row = {
                "model_id": model,
                "condition": cond,
                "n_total": n_total,
                "n_success": n_success,
                "parse_success_rate": round(parse_rate, 4),
                "mean_conf": round(conf.mean(), 4) if len(conf) > 0 else None,
                "sd_conf": round(conf.std(), 4) if len(conf) > 0 else None,
                "skewness": round(conf.skew(), 4) if len(conf) > 0 else None,
                "pct_at_ceiling": round((conf >= H1_CEILING_THRESHOLD).mean(), 4) if len(conf) > 0 else None,
                "pct_below_050": round((conf < 0.50).mean(), 4) if len(conf) > 0 else None,
                "pct_below_020": round((conf < 0.20).mean(), 4) if len(conf) > 0 else None,
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("A1 — CONFIDENCE DISTRIBUTION PER CELL")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


# ============================================================================
# A2 — VALIDITY SCREEN CLASSIFICATION PER CELL
# ============================================================================

def compute_validity_indices(conf_binary: np.ndarray, correct: np.ndarray):
    """
    Compute validity indices from binarised confidence and correctness.

    2x2 table:
        a = high conf & correct
        b = high conf & incorrect
        c = low conf & correct
        d = low conf & incorrect
    """
    high = conf_binary == 1
    low = conf_binary == 0
    corr = correct == 1
    incorr = correct == 0

    a = int((high & corr).sum())
    b = int((high & incorr).sum())
    c = int((low & corr).sum())
    d = int((low & incorr).sum())

    n = a + b + c + d
    n_correct = a + c
    n_incorrect = b + d
    n_high = a + b
    n_low = c + d

    # L = P(high conf | incorrect)
    L_val, L_ci = wilson_ci(b, n_incorrect) if n_incorrect > 0 else (None, (None, None))

    # Fp = P(low conf | correct)
    Fp_val, Fp_ci = wilson_ci(c, n_correct) if n_correct > 0 else (None, (None, None))

    # RBS = Fp - (1 - L)
    if L_val is not None and Fp_val is not None:
        RBS_val = Fp_val - (1 - L_val)
        # CI for RBS using component SEs
        se_Fp = np.sqrt(Fp_val * (1 - Fp_val) / n_correct) if n_correct > 0 else 0
        se_L = np.sqrt(L_val * (1 - L_val) / n_incorrect) if n_incorrect > 0 else 0
        se_RBS = np.sqrt(se_Fp**2 + se_L**2)
        z = stats.norm.ppf(0.975)
        RBS_ci = (RBS_val - z * se_RBS, RBS_val + z * se_RBS)
    else:
        RBS_val = None
        RBS_ci = (None, None)

    # TRIN
    TRIN_val = max(n_high, n_low) / n if n > 0 else None
    TRIN_dir = "fixed-high" if n_high >= n_low else "fixed-low"

    # r(confidence, correct) — point-biserial
    if len(np.unique(conf_binary)) > 1 and len(np.unique(correct)) > 1:
        r_val, r_p = pointbiserialr(conf_binary, correct)
        # CI via Fisher z
        z_r = np.arctanh(r_val) if abs(r_val) < 1.0 else np.sign(r_val) * 3.0
        se_z = 1 / np.sqrt(n - 3) if n > 3 else float("inf")
        z_crit = stats.norm.ppf(0.975)
        r_ci = (np.tanh(z_r - z_crit * se_z), np.tanh(z_r + z_crit * se_z))
    else:
        r_val, r_p = None, None
        r_ci = (None, None)

    return {
        "a": a, "b": b, "c": c, "d": d,
        "n": n, "n_correct": n_correct, "n_incorrect": n_incorrect,
        "n_high": n_high, "n_low": n_low,
        "L": L_val, "L_ci": L_ci,
        "Fp": Fp_val, "Fp_ci": Fp_ci,
        "RBS": RBS_val, "RBS_ci": RBS_ci,
        "TRIN": TRIN_val, "TRIN_dir": TRIN_dir,
        "r": r_val, "r_p": r_p, "r_ci": r_ci,
    }


def classify_validity(indices: dict) -> tuple:
    """
    Apply the ordered screening sequence from Cacioli (2026e) §2.5,
    with the §2.3 degeneracy pre-check applied first.
    Returns (tier, reason) where tier is 'Invalid', 'Indeterminate', or 'Valid'.
    """
    n = indices["n"]
    n_high = indices["n_high"]
    n_low = indices["n_low"]

    # §2.3 Degeneracy pre-check: "When the confidence signal has fewer than
    # 3 distinct values or more than 95% of responses fall in a single
    # category, the signal is degenerate and should be flagged without
    # further analysis." This fires BEFORE the Step 1 cell-count check,
    # because a cell with >95% in one binarised category is degenerate by
    # definition — the "Insufficient" label would be misleading (it implies
    # ambiguity, but the signal has collapsed).
    if n > 0:
        max_category_pct = max(n_high, n_low) / n
        dominant = "high" if n_high >= n_low else "low"
        if max_category_pct > 0.95:
            return ("Invalid",
                    f"§2.3 degeneracy: {max_category_pct:.1%} in single category "
                    f"(fixed-{dominant}), L={indices['L']:.3f}"
                    + (f", Fp={indices['Fp']:.3f}" if indices['Fp'] is not None else ""))

    # Step 1: cell counts (only reached if degeneracy criterion did not fire)
    for cell_name in ["a", "b", "c", "d"]:
        if indices[cell_name] < 5:
            return ("Insufficient", f"Cell '{cell_name}' has {indices[cell_name]} < 5 observations")

    # Step 2: TRIN (reported, does not trigger Invalid alone)
    # (computed but no classification action)

    # Step 3: Fp >= 0.50 → Invalid
    if indices["Fp"] is not None and indices["Fp"] >= FP_THRESHOLD:
        Fp_ci_lo = indices["Fp_ci"][0]
        if Fp_ci_lo is not None and Fp_ci_lo > 0.40:
            return ("Invalid", f"Fp = {indices['Fp']:.3f} >= 0.50, CI lower = {Fp_ci_lo:.3f} > 0.40")
        else:
            return ("Indeterminate", f"Fp = {indices['Fp']:.3f} >= 0.50 but CI lower = {Fp_ci_lo:.3f} <= 0.40")

    # Step 4: L >= 0.95 → Invalid
    if indices["L"] is not None and indices["L"] >= L_THRESHOLD:
        L_ci_lo = indices["L_ci"][0]
        if L_ci_lo is not None and L_ci_lo > 0.90:
            return ("Invalid", f"L = {indices['L']:.3f} >= 0.95, CI lower = {L_ci_lo:.3f} > 0.90")
        else:
            return ("Indeterminate", f"L = {indices['L']:.3f} >= 0.95 but CI lower = {L_ci_lo:.3f} <= 0.90")

    # Step 5: RBS > 0 → Invalid if CI excludes zero, else Indeterminate
    if indices["RBS"] is not None and indices["RBS"] > RBS_THRESHOLD:
        RBS_ci_lo = indices["RBS_ci"][0]
        if RBS_ci_lo is not None and RBS_ci_lo > 0:
            if indices["RBS"] > 0.05:
                return ("Invalid", f"RBS = {indices['RBS']:.3f} > 0, CI excludes zero, point > 0.05")
            else:
                return ("Indeterminate", f"RBS = {indices['RBS']:.3f} > 0, CI excludes zero but point <= 0.05")
        else:
            return ("Indeterminate", f"RBS = {indices['RBS']:.3f} > 0 but CI includes zero")

    # Step 6: r reported (no classification action from r alone)
    return ("Valid", "No threshold violations")


def run_a2(df: pd.DataFrame) -> pd.DataFrame:
    """A2: Validity screen classification per cell. VRS table."""
    rows = []
    for model in ALL_MODELS:
        for cond in CONDITIONS:
            cell = df[(df["model_id"] == model) & (df["condition"] == cond)]
            succ = cell[cell["parse_status"] == "success"].copy()

            if len(succ) == 0:
                rows.append({
                    "model_id": model, "condition": cond,
                    "n_success": 0, "tier": "Insufficient",
                    "reason": "No parse-success trials",
                })
                continue

            # Binarise at 0.50 per protocol §2.3
            conf_binary = (succ["parsed_confidence"] >= BINARISE_THRESHOLD).astype(int).values
            correct = succ["correct"].astype(int).values

            indices = compute_validity_indices(conf_binary, correct)
            tier, reason = classify_validity(indices)

            row = {
                "model_id": model,
                "condition": cond,
                "n_success": len(succ),
                "accuracy": round(correct.mean(), 4),
                "TRIN": round(indices["TRIN"], 4) if indices["TRIN"] is not None else None,
                "TRIN_dir": indices["TRIN_dir"],
                "L": round(indices["L"], 4) if indices["L"] is not None else None,
                "L_ci": f"({indices['L_ci'][0]:.3f}, {indices['L_ci'][1]:.3f})" if indices["L_ci"][0] is not None else None,
                "Fp": round(indices["Fp"], 4) if indices["Fp"] is not None else None,
                "Fp_ci": f"({indices['Fp_ci'][0]:.3f}, {indices['Fp_ci'][1]:.3f})" if indices["Fp_ci"][0] is not None else None,
                "RBS": round(indices["RBS"], 4) if indices["RBS"] is not None else None,
                "RBS_ci": f"({indices['RBS_ci'][0]:.3f}, {indices['RBS_ci'][1]:.3f})" if indices["RBS_ci"][0] is not None else None,
                "r": round(indices["r"], 4) if indices["r"] is not None else None,
                "r_p": f"{indices['r_p']:.4f}" if indices["r_p"] is not None else None,
                "r_ci": f"({indices['r_ci'][0]:.3f}, {indices['r_ci'][1]:.3f})" if indices["r_ci"][0] is not None else None,
                "tier": tier,
                "reason": reason,
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("A2 — VALIDITY SCREEN CLASSIFICATION (VRS TABLE)")
    print("=" * 80)
    for _, r in result.iterrows():
        print(f"  {r['model_id']} × {r['condition']}: {r['tier']} — {r['reason']}")
        if r.get("L") is not None:
            print(f"    L={r['L']}, Fp={r['Fp']}, RBS={r['RBS']}, TRIN={r['TRIN']} ({r['TRIN_dir']}), r={r['r']}")
    return result


# ============================================================================
# A3 — TYPE-2 AUROC₂ PER CELL
# ============================================================================

def compute_auroc2(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Non-parametric AUROC of confidence predicting correctness."""
    if len(np.unique(correct)) < 2:
        return float("nan")
    # AUROC = P(conf_correct > conf_incorrect) + 0.5 * P(conf_correct == conf_incorrect)
    # Use scipy rankdata approach
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(correct, confidence)


def bootstrap_auroc2(confidence: np.ndarray, correct: np.ndarray,
                     n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED):
    """Bootstrap 95% CI for AUROC₂."""
    rng = np.random.default_rng(seed)
    n = len(confidence)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = correct[idx]
        if len(np.unique(c)) < 2:
            continue
        aucs.append(compute_auroc2(confidence[idx], c))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def run_a3(df: pd.DataFrame) -> pd.DataFrame:
    """A3: Type-2 AUROC₂ per cell with bootstrap CI."""
    rows = []
    for model in ALL_MODELS:
        for cond in CONDITIONS:
            cell = df[(df["model_id"] == model) & (df["condition"] == cond)]
            succ = cell[cell["parse_status"] == "success"].copy()

            if len(succ) < 10 or len(succ["correct"].unique()) < 2:
                rows.append({
                    "model_id": model, "condition": cond,
                    "n_success": len(succ),
                    "auroc2": None, "ci_lo": None, "ci_hi": None,
                })
                continue

            conf = succ["parsed_confidence"].values
            corr = succ["correct"].astype(int).values
            auc = compute_auroc2(conf, corr)
            ci_lo, ci_hi = bootstrap_auroc2(conf, corr)

            rows.append({
                "model_id": model, "condition": cond,
                "n_success": len(succ),
                "auroc2": round(auc, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
            })

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("A3 — TYPE-2 AUROC₂ PER CELL")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


# ============================================================================
# A4 — FORMAT SHIFT EFFECT (H4)
# ============================================================================

def run_a4(a2_results: pd.DataFrame) -> pd.DataFrame:
    """A4: Format shift — tier transitions between NUM and CAT per model."""
    rows = []
    for model in INSTRUCT_MODELS:
        num_row = a2_results[(a2_results["model_id"] == model) & (a2_results["condition"] == "NUM")]
        cat_row = a2_results[(a2_results["model_id"] == model) & (a2_results["condition"] == "CAT")]

        if len(num_row) == 0 or len(cat_row) == 0:
            continue

        num_tier = num_row.iloc[0]["tier"]
        cat_tier = cat_row.iloc[0]["tier"]
        rescued = (num_tier == "Invalid") and (cat_tier in ["Indeterminate", "Valid"])

        rows.append({
            "model_id": model,
            "num_tier": num_tier,
            "cat_tier": cat_tier,
            "rescued": rescued,
        })

    result = pd.DataFrame(rows)
    n_rescued = result["rescued"].sum()

    print("\n" + "=" * 80)
    print("A4 — FORMAT SHIFT EFFECT (TIER TRANSITIONS)")
    print("=" * 80)
    print(result.to_string(index=False))
    print(f"\nModels rescued (NUM-Invalid → CAT-non-Invalid): {n_rescued}")
    return result


# ============================================================================
# A6 — LOGPROB-CONFIDENCE CONCORDANCE (H5)
# ============================================================================

def run_a6(df: pd.DataFrame) -> pd.DataFrame:
    """A6: Ridge R²_CV of logprob predicting confidence, per cell."""
    rows = []
    for model in INSTRUCT_MODELS:
        for cond in CONDITIONS:
            cell = df[(df["model_id"] == model) & (df["condition"] == cond)]
            succ = cell[cell["parse_status"] == "success"].copy()
            succ = succ.dropna(subset=["length_normalised_logprob", "parsed_confidence"])

            if len(succ) < 20:
                rows.append({
                    "model_id": model, "condition": cond,
                    "n": len(succ), "r2_cv": None,
                })
                continue

            X = succ["length_normalised_logprob"].values.reshape(-1, 1)
            y = succ["parsed_confidence"].values

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    ridge, X, y, cv=5, scoring="r2",
                    # Use fixed random state for reproducibility
                )
            r2_cv = scores.mean()

            rows.append({
                "model_id": model, "condition": cond,
                "n": len(succ), "r2_cv": round(r2_cv, 4),
            })

    result = pd.DataFrame(rows)

    # Per-condition means
    for cond in CONDITIONS:
        cond_vals = result[(result["condition"] == cond) & (result["r2_cv"].notna())]
        mean_r2 = cond_vals["r2_cv"].mean()
        print(f"  Mean R²_CV ({cond}): {mean_r2:.4f}")

    print("\n" + "=" * 80)
    print("A6 — LOGPROB-CONFIDENCE CONCORDANCE (RIDGE R²_CV)")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


# ============================================================================
# A7 — WORST-CASE INTERFACE COMPOSITE (descriptive)
# ============================================================================

def run_a7(df: pd.DataFrame, a2_results: pd.DataFrame) -> pd.DataFrame:
    """A7: 1 - parse_success_rate * P(Valid) per cell."""
    rows = []
    for model in ALL_MODELS:
        for cond in CONDITIONS:
            cell = df[(df["model_id"] == model) & (df["condition"] == cond)]
            n_total = len(cell)
            n_success = len(cell[cell["parse_status"] == "success"])
            parse_rate = n_success / n_total if n_total > 0 else 0.0

            a2_row = a2_results[(a2_results["model_id"] == model) & (a2_results["condition"] == cond)]
            if len(a2_row) > 0:
                tier = a2_row.iloc[0]["tier"]
                p_valid = 1.0 if tier == "Valid" else 0.0
            else:
                p_valid = 0.0

            composite = 1 - parse_rate * p_valid

            rows.append({
                "model_id": model, "condition": cond,
                "parse_rate": round(parse_rate, 4),
                "tier": tier if len(a2_row) > 0 else "N/A",
                "worst_case_composite": round(composite, 4),
            })

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("A7 — WORST-CASE INTERFACE COMPOSITE")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


# ============================================================================
# A8 — H1 SENSITIVITY ANALYSIS (descriptive)
# ============================================================================

def run_a8(df: pd.DataFrame) -> dict:
    """
    A8: H1 recomputed with parse failures coded as non-variable
    (saturated-equivalent) responses.
    """
    per_model = []
    for model in INSTRUCT_MODELS:
        cell = df[(df["model_id"] == model) & (df["condition"] == "NUM")]
        n_total = len(cell)
        succ = cell[cell["parse_status"] == "success"]
        n_at_ceiling = (succ["parsed_confidence"] >= H1_CEILING_THRESHOLD).sum()
        n_parse_fail = n_total - len(succ)

        # Sensitivity: parse failures counted as non-variable (ceiling-equivalent)
        pct_sens = (n_at_ceiling + n_parse_fail) / n_total if n_total > 0 else 0.0
        per_model.append({
            "model_id": model,
            "n_total": n_total,
            "n_success": len(succ),
            "n_at_ceiling": int(n_at_ceiling),
            "n_parse_fail": int(n_parse_fail),
            "pct_ceiling_sensitivity": round(pct_sens, 4),
        })

    per_model_df = pd.DataFrame(per_model)
    mean_sens = per_model_df["pct_ceiling_sensitivity"].mean()

    print("\n" + "=" * 80)
    print("A8 — H1 SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(per_model_df.to_string(index=False))
    print(f"\nMean % at ceiling (sensitivity, parse failures = non-variable): {mean_sens:.4f}")
    print(f"H1 sensitivity threshold (> 0.60): {'EXCEEDS' if mean_sens > H1_PREVALENCE_THRESHOLD else 'DOES NOT EXCEED'}")

    return {"per_model": per_model_df, "mean_sensitivity": mean_sens}


# ============================================================================
# CONFIRMATORY HYPOTHESIS TESTS
# ============================================================================

def test_h1(df: pd.DataFrame, a1_results: pd.DataFrame) -> dict:
    """H1: Mean of (% conf >= 0.95) across 7 instruct models on NUM > 0.60."""
    num_instruct = a1_results[
        (a1_results["model_id"].isin(INSTRUCT_MODELS)) &
        (a1_results["condition"] == "NUM")
    ]
    # Apply exclusion: >30% parse failure → exclude
    included = []
    excluded = []
    for _, row in num_instruct.iterrows():
        if row["parse_success_rate"] < (1 - PARSE_FAILURE_EXCLUSION):
            excluded.append(row["model_id"])
        else:
            included.append(row)

    if len(included) == 0:
        return {"confirmed": None, "mean_pct_ceiling": None, "n_models": 0,
                "excluded": excluded, "reason": "All models excluded"}

    included_df = pd.DataFrame(included)
    mean_ceiling = included_df["pct_at_ceiling"].mean()
    confirmed = bool(mean_ceiling > H1_PREVALENCE_THRESHOLD)

    result = {
        "confirmed": confirmed,
        "mean_pct_ceiling": round(float(mean_ceiling), 4),
        "n_models": len(included_df),
        "excluded": excluded,
        "per_model": included_df[["model_id", "pct_at_ceiling"]].to_dict("records"),
    }

    print("\n" + "=" * 80)
    print("H1 — SATURATION PREVALENCE (NUMERIC)")
    print("=" * 80)
    print(f"  Mean % at ceiling (>= 0.95) across instruct models on NUM: {mean_ceiling:.4f}")
    print(f"  Threshold: > 0.60")
    print(f"  Models included: {len(included_df)} ({', '.join(included_df['model_id'].tolist())})")
    if excluded:
        print(f"  Models excluded (>30% parse failure): {', '.join(excluded)}")
    print(f"  RESULT: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    return result


def test_h2(a2_results: pd.DataFrame, a1_results: pd.DataFrame) -> dict:
    """H2: At least 4 of 7 instruct models classified Invalid on NUM."""
    num_instruct = a2_results[
        (a2_results["model_id"].isin(INSTRUCT_MODELS)) &
        (a2_results["condition"] == "NUM")
    ]

    # Apply exclusion rule
    excluded = []
    included = []
    for _, row in num_instruct.iterrows():
        a1_row = a1_results[
            (a1_results["model_id"] == row["model_id"]) &
            (a1_results["condition"] == "NUM")
        ]
        if len(a1_row) > 0 and a1_row.iloc[0]["parse_success_rate"] < (1 - PARSE_FAILURE_EXCLUSION):
            excluded.append(row["model_id"])
        else:
            included.append(row)

    included_df = pd.DataFrame(included)
    n_invalid = int((included_df["tier"] == "Invalid").sum())
    confirmed = bool(n_invalid >= H2_COUNT_THRESHOLD)

    result = {
        "confirmed": confirmed,
        "n_invalid": int(n_invalid),
        "n_models": len(included_df),
        "excluded": excluded,
        "per_model": included_df[["model_id", "tier", "reason"]].to_dict("records"),
    }

    print("\n" + "=" * 80)
    print("H2 — VALIDITY SCREENING (NUMERIC)")
    print("=" * 80)
    print(f"  Invalid models on NUM: {n_invalid} of {len(included_df)}")
    print(f"  Threshold: >= 4")
    if excluded:
        print(f"  Models excluded (>30% parse failure): {', '.join(excluded)}")
    for r in result["per_model"]:
        print(f"    {r['model_id']}: {r['tier']}")
    print(f"  RESULT: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    return result


def test_h4(a4_results: pd.DataFrame) -> dict:
    """H4: At least 2 NUM-Invalid models reclassified to non-Invalid under CAT."""
    n_rescued = int(a4_results["rescued"].sum())
    confirmed = n_rescued >= H4_RESCUE_THRESHOLD

    rescued_models = a4_results[a4_results["rescued"]]["model_id"].tolist()

    result = {
        "confirmed": confirmed,
        "n_rescued": n_rescued,
        "rescued_models": rescued_models,
    }

    print("\n" + "=" * 80)
    print("H4 — FORMAT RESCUE")
    print("=" * 80)
    print(f"  Models rescued (NUM-Invalid → CAT-non-Invalid): {n_rescued}")
    print(f"  Threshold: >= 2")
    if rescued_models:
        print(f"  Rescued: {', '.join(rescued_models)}")
    print(f"  RESULT: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    return result


def test_h5(a6_results: pd.DataFrame) -> dict:
    """H5: Mean R²_CV < 0.20 in both conditions."""
    results_by_cond = {}
    for cond in CONDITIONS:
        cond_vals = a6_results[
            (a6_results["condition"] == cond) & (a6_results["r2_cv"].notna())
        ]
        mean_r2 = cond_vals["r2_cv"].mean()
        results_by_cond[cond] = {
            "mean_r2_cv": round(mean_r2, 4),
            "n_models": len(cond_vals),
            "below_threshold": mean_r2 < H5_R2_THRESHOLD,
        }

    confirmed = all(v["below_threshold"] for v in results_by_cond.values())

    result = {
        "confirmed": confirmed,
        "per_condition": results_by_cond,
    }

    print("\n" + "=" * 80)
    print("H5 — LOGPROB DOES NOT PREDICT VERBAL CONFIDENCE")
    print("=" * 80)
    for cond, v in results_by_cond.items():
        print(f"  {cond}: Mean R²_CV = {v['mean_r2_cv']:.4f} (n={v['n_models']}), < 0.20: {v['below_threshold']}")
    print(f"  RESULT: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    return result


# ============================================================================
# EXPLORATORY ANALYSES
# ============================================================================

def run_e_base(df: pd.DataFrame) -> dict:
    """E-base: M1 vs M2 length_normalised_logprob distributions on NUM."""
    m1 = df[(df["model_id"] == "M1") & (df["condition"] == "NUM") &
            (df["response_length_tokens"] > 0)]
    m2 = df[(df["model_id"] == "M2") & (df["condition"] == "NUM") &
            (df["response_length_tokens"] > 0)]

    lp_m1 = m1["length_normalised_logprob"].dropna().values
    lp_m2 = m2["length_normalised_logprob"].dropna().values

    ks_stat, ks_p = ks_2samp(lp_m1, lp_m2)
    mean_m1, mean_m2 = lp_m1.mean(), lp_m2.mean()

    # Entropy proxy: use histogram-based entropy
    def hist_entropy(x, bins=50):
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    ent_m1, ent_m2 = hist_entropy(lp_m1), hist_entropy(lp_m2)

    result = {
        "n_m1": len(lp_m1), "n_m2": len(lp_m2),
        "mean_m1": round(mean_m1, 4), "mean_m2": round(mean_m2, 4),
        "entropy_m1": round(ent_m1, 4), "entropy_m2": round(ent_m2, 4),
        "ks_stat": round(ks_stat, 4), "ks_p": ks_p,
        "direction": "M2 more compressed" if mean_m2 > mean_m1 else "M1 more compressed",
    }

    print("\n" + "=" * 80)
    print("E-BASE — IMPLICIT BASE-VS-INSTRUCT CONTRAST (LOGPROB)")
    print("=" * 80)
    print(f"  M1 (base): mean={mean_m1:.4f}, entropy={ent_m1:.4f}, n={len(lp_m1)}")
    print(f"  M2 (instruct): mean={mean_m2:.4f}, entropy={ent_m2:.4f}, n={len(lp_m2)}")
    print(f"  KS stat={ks_stat:.4f}, p={ks_p:.2e}")
    print(f"  Direction: {result['direction']}")

    return result


def run_e5(df: pd.DataFrame) -> dict:
    """E5: Reasoning contamination probe — thought_block_token_count vs confidence
    in M8 NUM, controlling for item difficulty."""
    m8_num = df[(df["model_id"] == "M8") & (df["condition"] == "NUM") &
                (df["parse_status"] == "success")].copy()

    if len(m8_num) < 20:
        print("\nE5: Insufficient M8 NUM success trials.")
        return {"n": len(m8_num), "skipped": True}

    # Item difficulty: proportion of M2-M7 that got each item correct
    other_models = ["M2", "M3", "M4", "M5", "M6", "M7"]
    item_diff = df[df["model_id"].isin(other_models)].groupby("item_index")["correct"].mean()
    item_diff.name = "item_difficulty"
    m8_num = m8_num.merge(item_diff, left_on="item_index", right_index=True, how="left")

    think_tok = m8_num["thought_block_token_count"].values
    conf = m8_num["parsed_confidence"].values
    diff = m8_num["item_difficulty"].values

    # Zero-order Spearman
    rho_zero, p_zero = spearmanr(think_tok, conf)

    # Partial Spearman: rank-residualise both on difficulty
    from scipy.stats import rankdata
    think_rank = rankdata(think_tok)
    conf_rank = rankdata(conf)
    diff_rank = rankdata(diff)

    # Residualise via OLS on ranks
    def residualise(y_rank, x_rank):
        x_rank = x_rank.reshape(-1, 1)
        from numpy.linalg import lstsq
        beta, _, _, _ = lstsq(np.c_[np.ones(len(x_rank)), x_rank], y_rank, rcond=None)
        return y_rank - (beta[0] + beta[1] * x_rank.ravel())

    think_resid = residualise(think_rank, diff_rank)
    conf_resid = residualise(conf_rank, diff_rank)
    rho_partial, p_partial = spearmanr(think_resid, conf_resid)

    result = {
        "n": len(m8_num),
        "rho_zero": round(rho_zero, 4), "p_zero": p_zero,
        "rho_partial": round(rho_partial, 4), "p_partial": p_partial,
    }

    print("\n" + "=" * 80)
    print("E5 — REASONING CONTAMINATION PROBE (M8 NUM)")
    print("=" * 80)
    print(f"  n = {len(m8_num)}")
    print(f"  Zero-order Spearman(think_tokens, confidence): ρ = {rho_zero:.4f}, p = {p_zero:.4f}")
    print(f"  Partial Spearman (controlling item difficulty): ρ = {rho_partial:.4f}, p = {p_partial:.4f}")

    return result


def run_e1(df: pd.DataFrame) -> pd.DataFrame:
    """E1: Per-trial hedge marker counts by accuracy condition."""
    succ = df[df["parse_status"] == "success"].copy()
    hedge_cols = ["hedge_epistemic_count", "hedge_self_count", "hedge_uncertainty_count"]

    rows = []
    for model in INSTRUCT_MODELS:
        for cond in CONDITIONS:
            cell = succ[(succ["model_id"] == model) & (succ["condition"] == cond)]
            for correct_val in [True, False]:
                subset = cell[cell["correct"] == correct_val]
                if len(subset) == 0:
                    continue
                row = {"model_id": model, "condition": cond,
                       "correct": correct_val, "n": len(subset)}
                for col in hedge_cols:
                    row[f"mean_{col}"] = round(subset[col].mean(), 4)
                rows.append(row)

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("E1 — HEDGE MARKERS BY ACCURACY")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


def run_e2(df: pd.DataFrame) -> pd.DataFrame:
    """E2: Within-correct vs within-incorrect response length distributions."""
    succ = df[df["parse_status"] == "success"].copy()
    rows = []
    for model in INSTRUCT_MODELS:
        for cond in CONDITIONS:
            cell = succ[(succ["model_id"] == model) & (succ["condition"] == cond)]
            for correct_val in [True, False]:
                subset = cell[cell["correct"] == correct_val]
                if len(subset) == 0:
                    continue
                rows.append({
                    "model_id": model, "condition": cond,
                    "correct": correct_val, "n": len(subset),
                    "mean_length_tokens": round(subset["response_length_tokens"].mean(), 2),
                    "sd_length_tokens": round(subset["response_length_tokens"].std(), 2),
                    "median_length_tokens": subset["response_length_tokens"].median(),
                })

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("E2 — RESPONSE LENGTH BY ACCURACY")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


def run_e3(df: pd.DataFrame) -> dict:
    """E3: Item difficulty vs mean confidence across models (descriptive)."""
    succ = df[(df["parse_status"] == "success") & (df["model_id"].isin(INSTRUCT_MODELS))].copy()

    # Item difficulty = proportion of models correct per item
    item_stats = succ.groupby("item_index").agg(
        mean_correct=("correct", "mean"),
        mean_confidence=("parsed_confidence", "mean"),
        n_models=("model_id", "nunique"),
    ).reset_index()

    rho, p = spearmanr(item_stats["mean_correct"], item_stats["mean_confidence"])

    print("\n" + "=" * 80)
    print("E3 — ITEM DIFFICULTY VS MEAN CONFIDENCE")
    print("=" * 80)
    print(f"  n_items = {len(item_stats)}")
    print(f"  Spearman(item_difficulty, mean_confidence): ρ = {rho:.4f}, p = {p:.4f}")

    return {"n_items": len(item_stats), "rho": round(rho, 4), "p": p}


def run_e4(df: pd.DataFrame, a1_results: pd.DataFrame) -> pd.DataFrame:
    """E4: Family-level descriptive analysis on saturation metrics."""
    family_map = {
        "M1": "Llama (base)", "M2": "Llama", "M3": "Llama",
        "M4": "Mistral", "M5": "Qwen", "M6": "Qwen",
        "M7": "Gemma", "M8": "DeepSeek-distilled",
    }

    a1_with_family = a1_results.copy()
    a1_with_family["family"] = a1_with_family["model_id"].map(family_map)

    # Only instruct models
    instruct = a1_with_family[a1_with_family["model_id"].isin(INSTRUCT_MODELS)]
    family_summary = instruct.groupby(["family", "condition"]).agg(
        mean_pct_ceiling=("pct_at_ceiling", "mean"),
        mean_conf=("mean_conf", "mean"),
        n_models=("model_id", "nunique"),
    ).reset_index()

    print("\n" + "=" * 80)
    print("E4 — FAMILY-LEVEL SATURATION SUMMARY")
    print("=" * 80)
    print(family_summary.to_string(index=False))
    return family_summary


def run_e6(df: pd.DataFrame) -> pd.DataFrame:
    """E6: Degenerate-loop rates by model."""
    rows = []
    for model in ALL_MODELS:
        for cond in CONDITIONS:
            cell = df[(df["model_id"] == model) & (df["condition"] == cond)]
            n_total = len(cell)
            n_degen = (cell["parse_status"] == "degenerate_loop").sum()
            rows.append({
                "model_id": model, "condition": cond,
                "n_total": n_total,
                "n_degenerate": int(n_degen),
                "pct_degenerate": round(n_degen / n_total, 4) if n_total > 0 else 0,
            })

    result = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("E6 — DEGENERATE LOOP RATES")
    print("=" * 80)
    print(result.to_string(index=False))
    return result


def run_e7(df: pd.DataFrame) -> dict:
    """E7: Relationship between hedge markers and parsed confidence."""
    succ = df[(df["parse_status"] == "success") & (df["model_id"].isin(INSTRUCT_MODELS))].copy()
    hedge_cols = ["hedge_epistemic_count", "hedge_self_count", "hedge_uncertainty_count"]
    total_hedge = succ[hedge_cols].sum(axis=1)

    rho, p = spearmanr(total_hedge, succ["parsed_confidence"])

    print("\n" + "=" * 80)
    print("E7 — HEDGE MARKERS VS CONFIDENCE")
    print("=" * 80)
    print(f"  Spearman(total_hedge, confidence): ρ = {rho:.4f}, p = {p:.4f}")

    return {"rho": round(rho, 4), "p": p}


def run_e8(df: pd.DataFrame) -> dict:
    """E8: MAR plausibility check — parse failure rates by difficulty and length."""
    instruct = df[df["model_id"].isin(INSTRUCT_MODELS)].copy()

    # Item difficulty across models
    item_correct = instruct.groupby("item_index")["correct"].mean()
    instruct = instruct.merge(item_correct.rename("item_difficulty"),
                              left_on="item_index", right_index=True, how="left")

    instruct["parse_fail"] = (instruct["parse_status"] != "success").astype(int)

    # Correlation of parse failure with item difficulty
    rho_diff, p_diff = spearmanr(instruct["item_difficulty"], instruct["parse_fail"])

    # Correlation of parse failure with response length
    rho_len, p_len = spearmanr(instruct["response_length_tokens"], instruct["parse_fail"])

    # Parse failure rate by correctness
    correct_fail = instruct[instruct["correct"] == True]["parse_fail"].mean()
    incorrect_fail = instruct[instruct["correct"] == False]["parse_fail"].mean()

    result = {
        "rho_difficulty": round(rho_diff, 4), "p_difficulty": p_diff,
        "rho_length": round(rho_len, 4), "p_length": p_len,
        "parse_fail_rate_correct": round(correct_fail, 4),
        "parse_fail_rate_incorrect": round(incorrect_fail, 4),
    }

    print("\n" + "=" * 80)
    print("E8 — MAR PLAUSIBILITY CHECK")
    print("=" * 80)
    print(f"  Spearman(item_difficulty, parse_fail): ρ = {rho_diff:.4f}, p = {p_diff:.4f}")
    print(f"  Spearman(response_length, parse_fail): ρ = {rho_len:.4f}, p = {p_len:.4f}")
    print(f"  Parse failure rate | correct: {correct_fail:.4f}")
    print(f"  Parse failure rate | incorrect: {incorrect_fail:.4f}")

    return result


def run_e9(df: pd.DataFrame) -> dict:
    """E9: Split-half stability of validity classifications."""
    rng = np.random.default_rng(SEED)
    items = df["item_index"].unique()
    rng.shuffle(items)
    half = len(items) // 2
    split_a = set(items[:half])
    split_b = set(items[half:])

    agreements = 0
    total = 0
    results = []

    for model in INSTRUCT_MODELS:
        for cond in CONDITIONS:
            for split_name, split_items in [("A", split_a), ("B", split_b)]:
                cell = df[(df["model_id"] == model) & (df["condition"] == cond) &
                          (df["item_index"].isin(split_items))]
                succ = cell[cell["parse_status"] == "success"]
                if len(succ) < 10:
                    continue
                conf_binary = (succ["parsed_confidence"] >= BINARISE_THRESHOLD).astype(int).values
                correct = succ["correct"].astype(int).values
                indices = compute_validity_indices(conf_binary, correct)
                tier, _ = classify_validity(indices)
                results.append({"model_id": model, "condition": cond,
                               "split": split_name, "tier": tier})

    results_df = pd.DataFrame(results)
    # Check agreement
    for model in INSTRUCT_MODELS:
        for cond in CONDITIONS:
            a_tier = results_df[(results_df["model_id"] == model) &
                               (results_df["condition"] == cond) &
                               (results_df["split"] == "A")]
            b_tier = results_df[(results_df["model_id"] == model) &
                               (results_df["condition"] == cond) &
                               (results_df["split"] == "B")]
            if len(a_tier) > 0 and len(b_tier) > 0:
                total += 1
                if a_tier.iloc[0]["tier"] == b_tier.iloc[0]["tier"]:
                    agreements += 1

    agreement_rate = agreements / total if total > 0 else None

    print("\n" + "=" * 80)
    print("E9 — SPLIT-HALF STABILITY")
    print("=" * 80)
    print(f"  Agreement: {agreements}/{total} = {agreement_rate:.4f}" if agreement_rate else "  Insufficient data")

    return {"agreements": agreements, "total": total, "agreement_rate": agreement_rate}


def run_e10(df: pd.DataFrame, a1_results: pd.DataFrame) -> pd.DataFrame:
    """E10: Scale (parameter count) vs saturation severity, descriptive."""
    scale_map = {
        "M2": 8, "M3": 8, "M4": 7, "M5": 3, "M6": 7, "M7": 9, "M8": 8,
    }

    num_instruct = a1_results[
        (a1_results["model_id"].isin(INSTRUCT_MODELS)) &
        (a1_results["condition"] == "NUM")
    ].copy()
    num_instruct["params_b"] = num_instruct["model_id"].map(scale_map)

    rho, p = spearmanr(num_instruct["params_b"], num_instruct["pct_at_ceiling"])

    print("\n" + "=" * 80)
    print("E10 — SCALE VS SATURATION")
    print("=" * 80)
    print(num_instruct[["model_id", "params_b", "pct_at_ceiling"]].to_string(index=False))
    print(f"\n  Spearman(params_b, pct_ceiling): ρ = {rho:.4f}, p = {p:.4f}")

    return num_instruct


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Saturation study analysis pipeline")
    parser.add_argument("--analysis", type=str, default=None,
                        help="Run a single analysis (A1, A2, A3, A4, A6, A7, A8, "
                             "H1, H2, H4, H5, E-base, E1-E10)")
    parser.add_argument("--confirmatory", action="store_true",
                        help="Run confirmatory hypotheses only (H1, H2, H4, H5)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Path to raw_responses.parquet")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows.")

    # Storage for all results
    results = {}

    if args.analysis:
        # Single analysis mode
        target = args.analysis.upper().replace("-", "_")
        if target == "A1":
            results["A1"] = run_a1(df)
        elif target == "A2":
            results["A2"] = run_a2(df)
        elif target == "A3":
            results["A3"] = run_a3(df)
        elif target == "A6":
            results["A6"] = run_a6(df)
        elif target in ("A4", "H4"):
            results["A2"] = run_a2(df)
            results["A4"] = run_a4(results["A2"])
        elif target in ("A7",):
            results["A2"] = run_a2(df)
            results["A7"] = run_a7(df, results["A2"])
        elif target in ("A8",):
            results["A8"] = run_a8(df)
        elif target == "H1":
            results["A1"] = run_a1(df)
            results["H1"] = test_h1(df, results["A1"])
        elif target == "H2":
            results["A1"] = run_a1(df)
            results["A2"] = run_a2(df)
            results["H2"] = test_h2(results["A2"], results["A1"])
        elif target == "H5":
            results["A6"] = run_a6(df)
            results["H5"] = test_h5(results["A6"])
        elif target == "E_BASE":
            results["E_base"] = run_e_base(df)
        elif target == "E5":
            results["E5"] = run_e5(df)
        elif target.startswith("E") and target[1:].isdigit():
            e_num = int(target[1:])
            e_funcs = {1: run_e1, 2: run_e2, 6: run_e6}
            if e_num == 3:
                results["E3"] = run_e3(df)
            elif e_num == 4:
                results["A1"] = run_a1(df)
                results["E4"] = run_e4(df, results["A1"])
            elif e_num == 7:
                results["E7"] = run_e7(df)
            elif e_num == 8:
                results["E8"] = run_e8(df)
            elif e_num == 9:
                results["E9"] = run_e9(df)
            elif e_num == 10:
                results["A1"] = run_a1(df)
                results["E10"] = run_e10(df, results["A1"])
            elif e_num in e_funcs:
                results[f"E{e_num}"] = e_funcs[e_num](df)
        else:
            print(f"Unknown analysis: {args.analysis}")
            return
    elif args.confirmatory:
        # Confirmatory only
        print("\n*** CONFIRMATORY ANALYSES ONLY ***\n")
        results["A1"] = run_a1(df)
        results["A2"] = run_a2(df)
        results["H1"] = test_h1(df, results["A1"])
        results["H2"] = test_h2(results["A2"], results["A1"])
        results["A4"] = run_a4(results["A2"])
        results["H4"] = test_h4(results["A4"])
        results["A6"] = run_a6(df)
        results["H5"] = test_h5(results["A6"])
    else:
        # Full pipeline
        print("\n*** FULL ANALYSIS PIPELINE ***\n")

        # Primary analyses
        results["A1"] = run_a1(df)
        results["A2"] = run_a2(df)
        results["A3"] = run_a3(df)
        results["A4"] = run_a4(results["A2"])
        # A5 retired
        results["A6"] = run_a6(df)
        results["A7"] = run_a7(df, results["A2"])
        results["A8"] = run_a8(df)

        # Confirmatory hypotheses
        results["H1"] = test_h1(df, results["A1"])
        results["H2"] = test_h2(results["A2"], results["A1"])
        results["H4"] = test_h4(results["A4"])
        results["H5"] = test_h5(results["A6"])

        # Exploratory
        results["E_base"] = run_e_base(df)
        results["E1"] = run_e1(df)
        results["E2"] = run_e2(df)
        results["E3"] = run_e3(df)
        results["E4"] = run_e4(df, results["A1"])
        results["E5"] = run_e5(df)
        results["E6"] = run_e6(df)
        results["E7"] = run_e7(df)
        results["E8"] = run_e8(df)
        results["E9"] = run_e9(df)
        results["E10"] = run_e10(df, results["A1"])

    # Save DataFrames to CSV
    for key, val in results.items():
        if isinstance(val, pd.DataFrame):
            out_path = output_dir / f"{key}.csv"
            val.to_csv(out_path, index=False)
            print(f"\nSaved {key} → {out_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for h in ["H1", "H2", "H4", "H5"]:
        if h in results and isinstance(results[h], dict):
            status = results[h].get("confirmed")
            if status is None:
                print(f"  {h}: INDETERMINATE")
            elif bool(status):
                print(f"  {h}: CONFIRMED")
            else:
                print(f"  {h}: NOT CONFIRMED")

    print("\nDone.")


if __name__ == "__main__":
    main()
