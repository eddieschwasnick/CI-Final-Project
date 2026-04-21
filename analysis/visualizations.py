"""
visualizations.py
-----------------
Phase 6: Generate all figures for the report.

Inputs:
  outputs/experiment_results.csv
  outputs/portfolio_metrics.csv
  outputs/sector_weights.csv
  outputs/treatment_effects.csv

Outputs (saved to reports/figures/):
  fig1_hhi_by_condition.png           (bar chart w/ error bars)
  fig2_beta_by_condition.png          (bar chart w/ error bars)
  fig3_breadth_by_condition.png       (bar chart w/ error bars)
  fig4_sector_allocation_stacked.png  (stacked bar by condition)
  fig5_hhi_boxplot.png                (box plot of HHI)
  fig6_weight_heatmap.png             (20 stocks × 4 conditions heatmap)
  fig7_treatment_effects.png          (coefficient plot w/ 95% CIs)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV    = os.path.join(BASE_DIR, "outputs", "experiment_results.csv")
METRICS_CSV    = os.path.join(BASE_DIR, "outputs", "portfolio_metrics.csv")
SECTOR_CSV     = os.path.join(BASE_DIR, "outputs", "sector_weights.csv")
EFFECTS_CSV    = os.path.join(BASE_DIR, "outputs", "treatment_effects.csv")
FIG_DIR        = os.path.join(BASE_DIR, "reports", "figures")

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

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_style("whitegrid")


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def bar_with_error(metrics: pd.DataFrame, col: str, ylabel: str, title: str, fname: str):
    """Mean ± 95% CI bar chart across conditions."""
    grouped = metrics.groupby("condition_name")[col].agg(["mean", "std", "count"])
    grouped = grouped.reindex(CONDITION_ORDER)
    ci95 = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE[c] for c in grouped.index]
    ax.bar(grouped.index, grouped["mean"], yerr=ci95, capsize=6,
           color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("Condition")
    fig.tight_layout()
    _save(fig, fname)


def sector_stacked_bar(sector_df: pd.DataFrame):
    sector_cols = [c for c in sector_df.columns
                   if c not in ("run_id", "condition_id", "condition_name")]
    means = sector_df.groupby("condition_name")[sector_cols].mean()
    means = means.reindex(CONDITION_ORDER) * 100  # to percent

    fig, ax = plt.subplots(figsize=(8, 6))
    means.plot(kind="bar", stacked=True, ax=ax,
               colormap="tab20", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean allocation (%)")
    ax.set_xlabel("Condition")
    ax.set_title("Average sector allocation by condition")
    ax.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    fig.tight_layout()
    _save(fig, "fig4_sector_allocation_stacked.png")


def hhi_boxplot(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=metrics, x="condition_name", y="hhi",
                order=CONDITION_ORDER,
                palette=[PALETTE[c] for c in CONDITION_ORDER], ax=ax)
    sns.stripplot(data=metrics, x="condition_name", y="hhi",
                  order=CONDITION_ORDER, color="black", alpha=0.4, size=3, ax=ax)
    ax.set_ylabel("HHI (concentration)")
    ax.set_xlabel("Condition")
    ax.set_title("Distribution of portfolio HHI across conditions")
    fig.tight_layout()
    _save(fig, "fig5_hhi_boxplot.png")


def weight_heatmap(results: pd.DataFrame):
    """Mean weight per stock per condition — 20 × 4 heatmap."""
    means = results.groupby("condition_name")[STOCK_UNIVERSE].mean()
    means = means.reindex(CONDITION_ORDER)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(means, annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={"label": "Mean weight (%)"}, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.set_title("Mean portfolio weight per stock per condition")
    ax.set_xlabel("")
    ax.set_ylabel("Condition")
    fig.tight_layout()
    _save(fig, "fig6_weight_heatmap.png")


def treatment_effect_plot(effects: pd.DataFrame):
    """Coefficient plot — diff-in-means vs control with 95% CIs."""
    plot_outcomes = ["hhi", "sector_hhi", "portfolio_beta", "portfolio_vol"]
    eff = effects[effects["outcome"].isin(plot_outcomes)].copy()

    fig, axes = plt.subplots(1, len(plot_outcomes),
                             figsize=(4 * len(plot_outcomes), 4), sharey=True)
    for ax, outcome in zip(axes, plot_outcomes):
        sub = eff[eff["outcome"] == outcome].set_index("treatment").reindex(
            ["fundamental", "technical", "combined"])

        y_pos = np.arange(len(sub))
        ax.errorbar(
            sub["diff_in_means"], y_pos,
            xerr=[sub["diff_in_means"] - sub["ci95_low"],
                  sub["ci95_high"] - sub["diff_in_means"]],
            fmt="o", color="#1f77b4", capsize=5, markersize=8,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub.index)
        ax.set_title(outcome)
        ax.set_xlabel("Difference vs. control")
    axes[0].set_ylabel("Treatment")
    fig.suptitle("Treatment effects (diff-in-means vs. control, 95% CI)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_treatment_effects.png")


def main():
    print("Loading data...")
    results  = pd.read_csv(RESULTS_CSV)
    results  = results[results["parse_success"] == True]
    metrics  = pd.read_csv(METRICS_CSV)
    sector   = pd.read_csv(SECTOR_CSV)
    effects  = pd.read_csv(EFFECTS_CSV)
    print(f"  {len(metrics)} runs, writing figures to {FIG_DIR}\n")

    bar_with_error(metrics, "hhi",
                   "Mean HHI (± 95% CI)",
                   "Mean portfolio concentration (HHI) by condition",
                   "fig1_hhi_by_condition.png")

    bar_with_error(metrics, "portfolio_beta",
                   "Mean portfolio beta (± 95% CI)",
                   "Mean portfolio beta by condition",
                   "fig2_beta_by_condition.png")

    bar_with_error(metrics, "breadth",
                   "Mean breadth (# non-zero holdings)",
                   "Mean portfolio breadth by condition",
                   "fig3_breadth_by_condition.png")

    sector_stacked_bar(sector)
    hhi_boxplot(metrics)
    weight_heatmap(results)
    treatment_effect_plot(effects)

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
