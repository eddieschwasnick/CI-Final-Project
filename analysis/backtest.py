"""
backtest.py
-----------
"For-fun" backtest section: how would each LLM-built portfolio have performed
if it had been held over the trailing 12 months?

We use the trailing 12-month return that already lives in the master dataset
(`return_12m`, computed in pull_financial_data.py). For each successfully-parsed
portfolio in experiment_results.csv we compute:

    portfolio_return_12m  =  Σ  w_i · return_12m_i        (weights as fractions)

This is a hindsight backtest: we are asking "if the LLM had picked this
allocation a year ago and held to today, what return would it have earned?"
It is purely descriptive — we are not re-running the experiment with year-old
prompts (which would burn API credits) — but it gives a meaningful read on
which information condition produced portfolios that line up with what
actually performed well over the last year.

Inputs
  outputs/experiment_results.csv
  data/csv_files/master_dataset.csv

Outputs
  outputs/portfolio_returns.csv
  reports/figures/fig8_portfolio_returns.png   (PNAS-style two-panel plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV  = os.path.join(BASE_DIR, "outputs", "experiment_results.csv")
MASTER_CSV   = os.path.join(BASE_DIR, "data", "csv_files", "master_dataset.csv")
RETURNS_CSV  = os.path.join(BASE_DIR, "outputs", "portfolio_returns.csv")
FIG_DIR      = os.path.join(BASE_DIR, "reports", "figures")

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "JNJ", "UNH", "PFE", "JPM", "GS",
    "XOM", "CVX", "PG", "KO", "CAT", "HON", "AMZN", "NVDA", "V",
    "LLY", "BA", "MMM",
]
CONDITION_ORDER = ["control", "fundamental", "technical", "combined"]
PALETTE = {
    "control":     "#6c757d",
    "fundamental": "#1f77b4",
    "technical":   "#2ca02c",
    "combined":    "#d62728",
}

sns.set_context("talk", font_scale=1.05)
sns.set_style("whitegrid")


def compute_portfolio_returns() -> pd.DataFrame:
    results = pd.read_csv(RESULTS_CSV)
    results = results[results["parse_success"] == True].copy()

    master = pd.read_csv(MASTER_CSV).set_index("ticker")
    returns_12m = master.loc[STOCK_UNIVERSE, "return_12m"].astype(float)  # already in %

    weights = results[STOCK_UNIVERSE].astype(float) / 100.0  # fractions
    results["portfolio_return_12m"] = weights.values @ returns_12m.values

    keep = ["run_id", "condition_id", "condition_name",
            "model", "portfolio_return_12m"]
    return results[keep].copy()


def pnas_returns_plot(returns_df: pd.DataFrame, fname: str):
    grouped = returns_df.groupby("condition_name")["portfolio_return_12m"] \
                        .agg(["mean", "std", "count"]) \
                        .reindex(CONDITION_ORDER)
    sem = grouped["std"] / np.sqrt(grouped["count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    y_lo = returns_df["portfolio_return_12m"].min()
    y_hi = returns_df["portfolio_return_12m"].max()
    pad  = 0.08 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    ylim = (y_lo - pad, y_hi + pad)
    x_pos = np.arange(len(CONDITION_ORDER))

    ax = axes[0]
    ax.errorbar(x_pos, grouped["mean"], yerr=sem, fmt="o",
                color="black", ecolor="black",
                capsize=4, markersize=8, linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Trailing 12-month return (%)")
    ax.set_title("Inferential uncertainty\n(SE only)", fontsize=14)

    ax = axes[1]
    rng = np.random.default_rng(1)
    for i, cond in enumerate(CONDITION_ORDER):
        sub = returns_df.loc[returns_df["condition_name"] == cond,
                             "portfolio_return_12m"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(np.full_like(sub, i, dtype=float) + jitter, sub,
                   color=PALETTE[cond], alpha=0.45, s=45,
                   edgecolor="white", linewidth=0.4, zorder=2)
    ax.errorbar(x_pos, grouped["mean"], yerr=sem, fmt="o",
                color="black", ecolor="black",
                capsize=4, markersize=8, linewidth=2, zorder=3)
    ax.axhline(grouped["mean"].max(), color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axhline(grouped["mean"].min(), color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_title("Outcome variability\n(SE + points)", fontsize=14)

    fig.suptitle("Trailing 12-month portfolio return by condition",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    print("Computing trailing 12-month portfolio returns...")
    returns_df = compute_portfolio_returns()

    os.makedirs(os.path.dirname(RETURNS_CSV), exist_ok=True)
    returns_df.to_csv(RETURNS_CSV, index=False)
    print(f"  Saved {RETURNS_CSV}  ({len(returns_df)} portfolios)")

    print("\n── Mean trailing 12-month return by condition ──")
    summary = returns_df.groupby("condition_name")["portfolio_return_12m"] \
                        .agg(["mean", "std", "count"]) \
                        .reindex(CONDITION_ORDER) \
                        .round(2)
    print(summary.to_string())

    pnas_returns_plot(returns_df, "fig8_portfolio_returns.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
