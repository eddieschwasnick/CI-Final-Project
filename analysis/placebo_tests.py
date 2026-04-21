"""
placebo_tests.py
----------------
Diagnostics required by the project rubric:
  1. Permutation placebo test — shuffle condition labels many times and
     recompute treatment effects. If the observed effects are truly causal
     (driven by the information injected in the prompt), they should be
     much larger than the null distribution generated under random assignment.
  2. Within-control split test — randomly split the control condition into
     two halves and compare. Since they come from the same population,
     the diff-in-means should be ~0 and p-values should be uniformly distributed.
  3. Weight-perturbation sensitivity — re-run the primary analysis after
     dropping the top and bottom HHI observation per condition (trimming).

Outputs:
  outputs/placebo_permutation.csv
  outputs/placebo_within_control.csv
  outputs/sensitivity_trimmed.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_CSV  = os.path.join(BASE_DIR, "outputs", "portfolio_metrics.csv")
PERM_CSV     = os.path.join(BASE_DIR, "outputs", "placebo_permutation.csv")
WITHIN_CSV   = os.path.join(BASE_DIR, "outputs", "placebo_within_control.csv")
TRIM_CSV     = os.path.join(BASE_DIR, "outputs", "sensitivity_trimmed.csv")

OUTCOMES   = ["hhi", "sector_hhi", "portfolio_beta", "portfolio_vol"]
TREATMENTS = ["fundamental", "technical", "combined"]
N_PERM     = 5000
RNG        = np.random.default_rng(42)


def diff_in_means(df: pd.DataFrame, outcome: str, treat: str) -> float:
    y_t = df.loc[df["condition_name"] == treat, outcome].values
    y_c = df.loc[df["condition_name"] == "control", outcome].values
    return y_t.mean() - y_c.mean()


# ── 1. Permutation placebo ────────────────────────────────────────────────────
def permutation_placebo(df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle condition labels and recompute diff-in-means."""
    rows = []
    for outcome in OUTCOMES:
        for treat in TREATMENTS:
            observed = diff_in_means(df, outcome, treat)

            # Build permutation null: use only the treat-vs-control pair
            sub = df[df["condition_name"].isin([treat, "control"])].copy()
            y = sub[outcome].values
            labels = sub["condition_name"].values
            treat_mask_size = int((labels == treat).sum())

            null_diffs = np.empty(N_PERM)
            for i in range(N_PERM):
                perm = RNG.permutation(len(y))
                # assign the first `treat_mask_size` shuffled obs to pseudo-treat
                y_perm = y[perm]
                pseudo_treat = y_perm[:treat_mask_size].mean()
                pseudo_ctrl  = y_perm[treat_mask_size:].mean()
                null_diffs[i] = pseudo_treat - pseudo_ctrl

            # two-sided p-value
            p_perm = float(np.mean(np.abs(null_diffs) >= abs(observed)))

            rows.append({
                "outcome":       outcome,
                "treatment":     treat,
                "observed_diff": observed,
                "null_mean":     float(null_diffs.mean()),
                "null_std":      float(null_diffs.std(ddof=1)),
                "p_permutation": p_perm,
                "n_permutations": N_PERM,
            })
    return pd.DataFrame(rows)


# ── 2. Within-control placebo ─────────────────────────────────────────────────
def within_control_placebo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split control group randomly in half many times and run t-tests.
    Under the null, p-values should be roughly uniform and diffs centered at 0.
    """
    ctrl = df[df["condition_name"] == "control"].copy()
    n_splits = 1000
    rows = []
    for outcome in OUTCOMES:
        y = ctrl[outcome].values
        diffs = np.empty(n_splits)
        pvals = np.empty(n_splits)
        for i in range(n_splits):
            idx = RNG.permutation(len(y))
            half = len(y) // 2
            a = y[idx[:half]]
            b = y[idx[half:2 * half]]
            diffs[i] = a.mean() - b.mean()
            _, p = stats.ttest_ind(a, b, equal_var=False)
            pvals[i] = p
        rows.append({
            "outcome":           outcome,
            "n_splits":          n_splits,
            "mean_diff":         float(diffs.mean()),
            "std_diff":          float(diffs.std(ddof=1)),
            "pct_p_below_05":    float((pvals < 0.05).mean()),
            "expected_if_null":  0.05,
        })
    return pd.DataFrame(rows)


# ── 3. Sensitivity: trimmed sample ────────────────────────────────────────────
def trimmed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-estimate treatment effects after dropping the highest- and lowest-HHI
    observation per condition. If the findings are driven by outliers, they
    should weaken materially here.
    """
    trimmed_idx = []
    for cond in df["condition_name"].unique():
        sub = df[df["condition_name"] == cond].sort_values("hhi")
        keep = sub.iloc[1:-1]  # drop top and bottom
        trimmed_idx.extend(keep.index.tolist())
    trimmed = df.loc[trimmed_idx].copy()

    rows = []
    for outcome in OUTCOMES:
        for treat in TREATMENTS:
            y_t = trimmed.loc[trimmed["condition_name"] == treat, outcome].values
            y_c = trimmed.loc[trimmed["condition_name"] == "control", outcome].values
            diff = y_t.mean() - y_c.mean()
            t_stat, p = stats.ttest_ind(y_t, y_c, equal_var=False)
            rows.append({
                "outcome":       outcome,
                "treatment":     treat,
                "n_treat":       len(y_t),
                "n_ctrl":        len(y_c),
                "diff_trimmed":  diff,
                "p_value":       p,
            })
    return pd.DataFrame(rows)


def main():
    print("Loading portfolio_metrics.csv...")
    df = pd.read_csv(METRICS_CSV)
    print(f"  {len(df)} runs.\n")

    print(f"Running permutation placebo ({N_PERM:,} permutations per test)...")
    perm = permutation_placebo(df)
    perm.to_csv(PERM_CSV, index=False)
    print(perm[["outcome", "treatment", "observed_diff",
                "p_permutation"]].round(4).to_string(index=False))

    print("\nRunning within-control placebo (1,000 random splits)...")
    within = within_control_placebo(df)
    within.to_csv(WITHIN_CSV, index=False)
    print(within.round(4).to_string(index=False))
    print("  Expected pct_p_below_05 under null: ~0.05")

    print("\nRunning trimmed-sample sensitivity (drop top/bottom HHI per condition)...")
    trim = trimmed_sensitivity(df)
    trim.to_csv(TRIM_CSV, index=False)
    print(trim.round(4).to_string(index=False))

    print("\nDone. Wrote:")
    print(f"  {PERM_CSV}")
    print(f"  {WITHIN_CSV}")
    print(f"  {TRIM_CSV}")


if __name__ == "__main__":
    main()
