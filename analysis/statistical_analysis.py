"""
statistical_analysis.py
-----------------------
Phase 5: Estimate treatment effects and run regressions.

Inputs:
  outputs/portfolio_metrics.csv   (Phase 4)

Outputs:
  outputs/descriptive_stats.csv
  outputs/treatment_effects.csv     — diff in means vs control with 95% CI and p-values
  outputs/regression_results.txt    — OLS regression summaries per outcome
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_CSV     = os.path.join(BASE_DIR, "outputs", "portfolio_metrics.csv")
DESC_CSV        = os.path.join(BASE_DIR, "outputs", "descriptive_stats.csv")
EFFECTS_CSV     = os.path.join(BASE_DIR, "outputs", "treatment_effects.csv")
REG_TXT         = os.path.join(BASE_DIR, "outputs", "regression_results.txt")

OUTCOMES   = ["hhi", "sector_hhi", "portfolio_beta", "portfolio_vol", "breadth"]
CONDITIONS = ["control", "fundamental", "technical", "combined"]
TREATMENTS = ["fundamental", "technical", "combined"]


# ── 5A. Descriptive Statistics ────────────────────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        sub = df[df["condition_name"] == cond]
        for outcome in OUTCOMES:
            vals = sub[outcome].dropna()
            rows.append({
                "condition": cond,
                "outcome":   outcome,
                "n":         len(vals),
                "mean":      vals.mean(),
                "median":    vals.median(),
                "std":       vals.std(ddof=1),
                "min":       vals.min(),
                "max":       vals.max(),
            })
    return pd.DataFrame(rows)


# ── 5B. Treatment Effects (diff in means vs control) ──────────────────────────
def treatment_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    control = df[df["condition_name"] == "control"]
    for outcome in OUTCOMES:
        y_ctrl = control[outcome].dropna().values
        for treat in TREATMENTS:
            y_treat = df[df["condition_name"] == treat][outcome].dropna().values

            diff = y_treat.mean() - y_ctrl.mean()

            # Welch's t-test (unequal variances) — most defensible for pilot data
            t_stat, p_t = stats.ttest_ind(y_treat, y_ctrl, equal_var=False)

            # Wilcoxon rank-sum (non-parametric backup)
            u_stat, p_u = stats.ranksums(y_treat, y_ctrl)

            # Welch-style 95% CI on the difference in means
            se = np.sqrt(y_treat.var(ddof=1) / len(y_treat) +
                         y_ctrl.var(ddof=1)  / len(y_ctrl))
            # degrees of freedom via Welch-Satterthwaite
            df_welch = (se ** 4) / (
                (y_treat.var(ddof=1) / len(y_treat)) ** 2 / (len(y_treat) - 1) +
                (y_ctrl.var(ddof=1)  / len(y_ctrl))  ** 2 / (len(y_ctrl)  - 1)
            )
            t_crit = stats.t.ppf(0.975, df_welch)
            ci_low  = diff - t_crit * se
            ci_high = diff + t_crit * se

            rows.append({
                "outcome":         outcome,
                "treatment":       treat,
                "n_treat":         len(y_treat),
                "n_ctrl":          len(y_ctrl),
                "mean_treat":      y_treat.mean(),
                "mean_ctrl":       y_ctrl.mean(),
                "diff_in_means":   diff,
                "ci95_low":        ci_low,
                "ci95_high":       ci_high,
                "welch_t":         t_stat,
                "p_value_t":       p_t,
                "wilcoxon_stat":   u_stat,
                "p_value_wilcox":  p_u,
                "significant_5pct": p_t < 0.05,
            })
    return pd.DataFrame(rows)


# ── 5C. OLS Regression ────────────────────────────────────────────────────────
def run_regression(df: pd.DataFrame, outcome: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Y_i = α + β1*Fundamental_i + β2*Technical_i + β3*Combined_i + ε_i
    (Control is the omitted baseline category.)
    """
    df = df.copy()
    for treat in TREATMENTS:
        df[treat] = (df["condition_name"] == treat).astype(int)

    X = df[TREATMENTS]
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.OLS(y, X, missing="drop").fit()
    return model


# ── 5D. Robustness Check: outliers ─────────────────────────────────────────────
def outlier_flags(df: pd.DataFrame) -> dict:
    """Simple outlier sanity check on HHI — flag extreme cases."""
    q1, q3 = df["hhi"].quantile([0.25, 0.75])
    iqr = q3 - q1
    hi_thresh = q3 + 3 * iqr
    outliers = df[df["hhi"] > hi_thresh]
    return {
        "n_outliers": len(outliers),
        "threshold":  float(hi_thresh),
        "outlier_run_ids": outliers["run_id"].tolist(),
    }


def main():
    print("Loading portfolio_metrics.csv...")
    df = pd.read_csv(METRICS_CSV)
    print(f"  {len(df)} runs across {df['condition_name'].nunique()} conditions.\n")

    # 5A
    desc = descriptive_stats(df)
    desc.to_csv(DESC_CSV, index=False)
    print("── 5A. Descriptive Stats ──")
    pivot = desc.pivot(index="outcome", columns="condition", values="mean").round(4)
    print(pivot.to_string())

    # 5B
    effects = treatment_effects(df)
    effects.to_csv(EFFECTS_CSV, index=False)
    print("\n── 5B. Treatment Effects (diff vs control) ──")
    print(effects[["outcome", "treatment", "diff_in_means",
                   "ci95_low", "ci95_high", "p_value_t",
                   "significant_5pct"]].round(4).to_string(index=False))

    # 5C
    with open(REG_TXT, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("OLS Regressions — Treatment Effects on Portfolio Metrics\n")
        f.write("Model: Y = α + β1*Fundamental + β2*Technical + β3*Combined + ε\n")
        f.write("(Control is the omitted baseline.)\n")
        f.write("=" * 70 + "\n\n")
        for outcome in OUTCOMES:
            model = run_regression(df, outcome)
            header = f"\n>>> OUTCOME: {outcome}\n"
            f.write(header)
            f.write("-" * 70 + "\n")
            f.write(model.summary().as_text())
            f.write("\n\n")
    print(f"\nSaved regression results → {REG_TXT}")

    # Print regression summary snapshot
    print("\n── 5C. Regression coefficients (treatment vs control) ──")
    snap_rows = []
    for outcome in OUTCOMES:
        model = run_regression(df, outcome)
        for treat in TREATMENTS:
            snap_rows.append({
                "outcome":  outcome,
                "treatment": treat,
                "coef":     model.params[treat],
                "std_err":  model.bse[treat],
                "p_value":  model.pvalues[treat],
                "sig_5pct": model.pvalues[treat] < 0.05,
            })
    snap = pd.DataFrame(snap_rows).round(4)
    print(snap.to_string(index=False))

    # 5D
    print("\n── 5D. Robustness: outlier check on HHI ──")
    out = outlier_flags(df)
    print(f"  IQR-based upper threshold: {out['threshold']:.4f}")
    print(f"  Outlier runs: {out['n_outliers']}")
    if out["n_outliers"]:
        print(f"  IDs: {out['outlier_run_ids']}")

    print("\nDone. Files written:")
    print(f"  {DESC_CSV}")
    print(f"  {EFFECTS_CSV}")
    print(f"  {REG_TXT}")


if __name__ == "__main__":
    main()
