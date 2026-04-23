"""
generate_figures.py — Publication figures for the saturation manuscript.

Reads raw_responses.parquet and produces:
    Figure 1: Ceiling rate by model and condition (grouped bar)
    Figure 2: M8 reasoning-trace length vs confidence scatter (E5)
    Figure 3: AUROC2 forest plot by model and condition

Usage:
    python generate_figures.py

Output:
    figures/fig1_ceiling_rates.pdf
    figures/fig2_e5_reasoning_scatter.pdf
    figures/fig3_auroc_forest.pdf

Author: JP Cacioli
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATH = Path("raw_responses.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

INSTRUCT_MODELS = ["M2", "M3", "M4", "M5", "M6", "M7", "M8"]
ALL_MODELS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

# Colour palette — accessible, print-safe
C_NUM = "#2166AC"   # blue
C_CAT = "#B2182B"   # red
C_GREY = "#888888"
C_CORRECT = "#4DAF4A"
C_INCORRECT = "#E41A1C"

# Global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================================
# DATA
# ============================================================================

def load():
    df = pd.read_parquet(DATA_PATH)
    assert len(df) == 8384
    return df


# ============================================================================
# FIGURE 1: Ceiling rates by model and condition
# ============================================================================

def fig1_ceiling_rates(df):
    """Grouped bar chart: % confidence >= 0.95 per model, NUM vs CAT."""
    models = INSTRUCT_MODELS
    num_rates = []
    cat_rates = []

    for m in models:
        for cond, store in [("NUM", num_rates), ("CAT", cat_rates)]:
            cell = df[(df["model_id"] == m) & (df["condition"] == cond)]
            succ = cell[cell["parse_status"] == "success"]
            if len(succ) > 0:
                rate = (succ["parsed_confidence"] >= 0.95).mean() * 100
            else:
                rate = 0.0
            store.append(rate)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_num = ax.bar(x - width/2, num_rates, width, color=C_NUM, label="NUM (0–100)", zorder=3)
    bars_cat = ax.bar(x + width/2, cat_rates, width, color=C_CAT, label="CAT (10-class)", zorder=3)

    # H1 threshold line
    ax.axhline(60, color=C_GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(len(models) - 0.5, 62, "H1 threshold (60%)", ha="right", va="bottom",
            fontsize=8, color=C_GREY)

    # Value labels on NUM bars
    for bar, val in zip(bars_num, num_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color=C_NUM)

    # Value labels on CAT bars
    for bar, val in zip(bars_cat, cat_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color=C_CAT)

    ax.set_xlabel("Model")
    ax.set_ylabel("% trials with confidence ≥ 95%")
    ax.set_title("Confidence ceiling rates by model and elicitation format")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    out = FIG_DIR / "fig1_ceiling_rates.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# FIGURE 2: E5 reasoning scatter (M8 NUM)
# ============================================================================

def fig2_e5_scatter(df):
    """Scatter: thought_block_token_count vs parsed_confidence, M8 NUM."""
    m8 = df[(df["model_id"] == "M8") & (df["condition"] == "NUM") &
            (df["parse_status"] == "success")].copy()

    if len(m8) < 20:
        print("Skipping fig2: insufficient M8 NUM data")
        return

    # Item difficulty for partial correlation
    other = df[df["model_id"].isin(["M2", "M3", "M4", "M5", "M6", "M7"])]
    item_diff = other.groupby("item_index")["correct"].mean()
    m8 = m8.merge(item_diff.rename("item_difficulty"),
                  left_on="item_index", right_index=True, how="left")

    think = m8["thought_block_token_count"].values
    conf = m8["parsed_confidence"].values
    correct = m8["correct"].values

    from scipy.stats import spearmanr
    rho_zero, _ = spearmanr(think, conf)

    # Partial correlation
    from scipy.stats import rankdata
    think_r = rankdata(think)
    conf_r = rankdata(conf)
    diff_r = rankdata(m8["item_difficulty"].values)

    def residualise(y, x):
        x = x.reshape(-1, 1)
        beta = np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
        return y - (beta[0] + beta[1] * x.ravel())

    think_resid = residualise(think_r, diff_r)
    conf_resid = residualise(conf_r, diff_r)
    rho_partial, _ = spearmanr(think_resid, conf_resid)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Colour by correctness
    colours = [C_CORRECT if c else C_INCORRECT for c in correct]
    ax.scatter(think, conf, c=colours, alpha=0.4, s=15, edgecolors="none", zorder=3)

    # Trend line (linear fit for visual only)
    z = np.polyfit(think, conf, 1)
    x_line = np.linspace(think.min(), think.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="black", linewidth=1.5,
            linestyle="-", zorder=4, alpha=0.7)

    # Annotation
    ax.text(0.97, 0.97,
            f"ρ = {rho_zero:.2f} (zero-order)\n"
            f"ρ = {rho_partial:.2f} (partial, controlling item difficulty)\n"
            f"n = {len(m8)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    legend_elements = [
        Patch(facecolor=C_CORRECT, label="Correct"),
        Patch(facecolor=C_INCORRECT, label="Incorrect"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9)

    ax.set_xlabel("Reasoning trace length (tokens)")
    ax.set_ylabel("Verbalised confidence")
    ax.set_title("M8 (DeepSeek-R1-Distill): reasoning length vs confidence (NUM)")
    ax.grid(alpha=0.2, zorder=0)

    out = FIG_DIR / "fig2_e5_reasoning_scatter.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# FIGURE 3: AUROC2 forest plot
# ============================================================================

def bootstrap_auroc(conf, correct, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(conf)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = correct[idx]
        if len(np.unique(c)) < 2:
            continue
        aucs.append(roc_auc_score(c, conf[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def fig3_auroc_forest(df):
    """Forest plot: AUROC2 point estimates and bootstrap CIs."""
    models = INSTRUCT_MODELS
    results = []

    for m in models:
        for cond in ["NUM", "CAT"]:
            cell = df[(df["model_id"] == m) & (df["condition"] == cond)]
            succ = cell[cell["parse_status"] == "success"]
            conf = succ["parsed_confidence"].values
            correct = succ["correct"].astype(int).values

            if len(succ) < 10 or len(np.unique(correct)) < 2:
                results.append({
                    "model": m, "condition": cond,
                    "auroc": None, "ci_lo": None, "ci_hi": None, "n": len(succ)
                })
                continue

            auc = roc_auc_score(correct, conf)
            ci_lo, ci_hi = bootstrap_auroc(conf, correct)
            results.append({
                "model": m, "condition": cond,
                "auroc": auc, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(succ)
            })

    res = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    y_positions = []
    y_labels = []
    y = 0
    for m in reversed(models):
        for cond in ["CAT", "NUM"]:
            row = res[(res["model"] == m) & (res["condition"] == cond)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            colour = C_NUM if cond == "NUM" else C_CAT

            if row["auroc"] is not None:
                ax.plot(row["auroc"], y, "o", color=colour, markersize=6, zorder=4)
                ax.plot([row["ci_lo"], row["ci_hi"]], [y, y], "-", color=colour,
                        linewidth=1.5, zorder=3)
                # n label
                ax.text(row["ci_hi"] + 0.01, y, f"n={row['n']}",
                        va="center", fontsize=7, color=C_GREY)

            y_positions.append(y)
            y_labels.append(f"{m} {cond}")
            y += 1
        y += 0.5  # gap between models

    ax.axvline(0.5, color=C_GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.505, y - 1, "chance", fontsize=8, color=C_GREY, va="top")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("AUROC2")
    ax.set_title("Type-2 AUROC2 by model and condition (bootstrap 95% CI)")
    ax.set_xlim(0.35, 1.05)
    ax.grid(axis="x", alpha=0.2, zorder=0)

    legend_elements = [
        Patch(facecolor=C_NUM, label="NUM"),
        Patch(facecolor=C_CAT, label="CAT"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    out = FIG_DIR / "fig3_auroc_forest.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {out}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load()
    print(f"Loaded {len(df)} rows.")

    print("\nGenerating Figure 1: Ceiling rates...")
    fig1_ceiling_rates(df)

    print("\nGenerating Figure 2: E5 reasoning scatter...")
    fig2_e5_scatter(df)

    print("\nGenerating Figure 3: AUROC2 forest plot...")
    fig3_auroc_forest(df)

    print("\nDone. Figures in:", FIG_DIR)
