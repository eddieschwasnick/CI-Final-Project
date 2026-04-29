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

# ── Presentation styling ──────────────────────────────────────────────────────
# `set_context("talk")` enlarges fonts, ticks, and line widths so the charts
# read cleanly when embedded in a slide deck (vs the default "notebook" sizing).
sns.set_context("talk", font_scale=1.25)
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

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [PALETTE[c] for c in grouped.index]
    ax.bar(grouped.index, grouped["mean"], yerr=ci95, capsize=6,
           color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlabel("Condition")
    fig.tight_layout()
    _save(fig, fname)


def pnas_style_two_panel(metrics: pd.DataFrame, col: str, ylabel: str,
                         title: str, fname: str):
    """
    Two-panel point plot in the style of Pritschet et al. (PNAS 2023):
        Left panel  — "Inferential uncertainty (SE only)"
                      mean ± standard error of the mean per condition
        Right panel — "Outcome variability (SE + points)"
                      same point + SE, plus a jittered cloud of individual runs
    The two panels share a y-axis range so the eye sees that more data shrinks
    the error bars but does NOT shrink the cloud of underlying outcomes.
    """
    grouped = metrics.groupby("condition_name")[col].agg(["mean", "std", "count"])
    grouped = grouped.reindex(CONDITION_ORDER)
    sem = grouped["std"] / np.sqrt(grouped["count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # y-range driven by the right panel (which shows individual points)
    y_lo = metrics[col].min()
    y_hi = metrics[col].max()
    pad  = 0.08 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    ylim = (y_lo - pad, y_hi + pad)

    # ── Left: SE only ────────────────────────────────────────────────────────
    ax = axes[0]
    x_pos = np.arange(len(CONDITION_ORDER))
    ax.errorbar(x_pos, grouped["mean"], yerr=sem,
                fmt="o", color="black", ecolor="black",
                capsize=4, markersize=8, linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title("Inferential uncertainty\n(SE only)", fontsize=14)

    # ── Right: SE + jittered points ─────────────────────────────────────────
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, cond in enumerate(CONDITION_ORDER):
        sub = metrics.loc[metrics["condition_name"] == cond, col].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(np.full_like(sub, i, dtype=float) + jitter, sub,
                   color=PALETTE[cond], alpha=0.45, s=45,
                   edgecolor="white", linewidth=0.4, zorder=2)
    ax.errorbar(x_pos, grouped["mean"], yerr=sem,
                fmt="o", color="black", ecolor="black",
                capsize=4, markersize=8, linewidth=2, zorder=3)

    # Reference dashed lines spanning the full mean range (visual anchor like the PNAS figure)
    ax.axhline(grouped["mean"].max(), color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axhline(grouped["mean"].min(), color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_title("Outcome variability\n(SE + points)", fontsize=14)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fname)


def sector_stacked_bar(sector_df: pd.DataFrame):
    sector_cols = [c for c in sector_df.columns
                   if c not in ("run_id", "condition_id", "condition_name")]
    means = sector_df.groupby("condition_name")[sector_cols].mean()
    means = means.reindex(CONDITION_ORDER) * 100  # to percent

    fig, ax = plt.subplots(figsize=(11, 6.5))
    means.plot(kind="bar", stacked=True, ax=ax,
               colormap="tab20", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean allocation (%)")
    ax.set_xlabel("Condition")
    ax.set_title("Average sector allocation by condition")
    ax.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
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

    # Use a much wider canvas and smaller annotation font so the per-cell
    # numbers don't crash into the ticker labels. With set_context("talk")
    # active globally, the default annotation size is too large for a
    # 20-column heatmap.
    fig, ax = plt.subplots(figsize=(18, 5.5))
    sns.heatmap(means, annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={"label": "Mean weight (%)"}, ax=ax,
                linewidths=0.3, linecolor="white",
                annot_kws={"size": 11})
    ax.set_title("Mean portfolio weight per stock per condition (Claude)",
                 fontsize=15)
    ax.set_xlabel("")
    ax.set_ylabel("Condition")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
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

    # Replaces the old bar-chart fig1 with a two-panel point plot in the style
    # of Pritschet et al. (PNAS 2023): inferential uncertainty next to outcome
    # variability, so the audience can read both the SE on the mean and the
    # spread of individual portfolios.
    pnas_style_two_panel(metrics, "hhi",
                         "Portfolio HHI",
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
