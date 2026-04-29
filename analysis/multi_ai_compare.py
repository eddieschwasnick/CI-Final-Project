"""
multi_ai_compare.py
-------------------
Compute portfolio metrics + trailing 12-month returns for the multi-AI
experiment and produce comparison figures across {claude, openai, gemini}.

Inputs
  outputs/multi_ai_results.csv       (from run_multi_ai_experiment.py)
  data/csv_files/master_dataset.csv

Outputs
  outputs/multi_ai_metrics.csv
  reports/figures/fig9_multi_ai_hhi.png       — HHI by condition × model
  reports/figures/fig10_multi_ai_returns.png  — 12m return by condition × model
  reports/figures/fig11_multi_ai_sectors.png  — sector allocation per model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE_DIR, "outputs", "multi_ai_results.csv")
MASTER_CSV  = os.path.join(BASE_DIR, "data", "csv_files", "master_dataset.csv")
METRICS_CSV = os.path.join(BASE_DIR, "outputs", "multi_ai_metrics.csv")
FIG_DIR     = os.path.join(BASE_DIR, "reports", "figures")

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "JNJ", "UNH", "PFE", "JPM", "GS",
    "XOM", "CVX", "PG", "KO", "CAT", "HON", "AMZN", "NVDA", "V",
    "LLY", "BA", "MMM",
]
CONDITION_ORDER = ["control", "fundamental", "technical", "combined"]
MODEL_ORDER     = ["claude", "openai", "gemini"]
MODEL_LABELS    = {
    "claude": "Claude Sonnet 4.6",
    "openai": "OpenAI GPT-4o",
    "gemini": "Gemini 2.5 Pro",
}
SECTOR_ORDER = ["Technology", "Healthcare", "Financials",
                "Energy", "Consumer Goods", "Industrials"]
MODEL_PALETTE   = {
    "claude": "#d97706",   # Anthropic-ish amber
    "openai": "#10a37f",   # OpenAI green
    "gemini": "#4285f4",   # Google blue
}

sns.set_context("talk", font_scale=1.25)
sns.set_style("whitegrid")
os.makedirs(FIG_DIR, exist_ok=True)


def compute_metrics() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["parse_success"] == True].copy()

    master = pd.read_csv(MASTER_CSV).set_index("ticker")
    meta = master.loc[STOCK_UNIVERSE, ["sector", "beta", "volatility", "return_12m"]]

    weights_pct = df[STOCK_UNIVERSE].astype(float)
    weights_dec = weights_pct / 100.0

    df["hhi"]            = (weights_dec ** 2).sum(axis=1)
    df["portfolio_beta"] = weights_dec.values @ meta["beta"].values
    df["portfolio_vol"]  = weights_dec.values @ meta["volatility"].values
    df["return_12m"]     = weights_dec.values @ meta["return_12m"].values
    df["breadth"]        = (weights_pct > 0.01).sum(axis=1)

    sector_map = meta["sector"]
    sector_weights = weights_dec.T.groupby(sector_map).sum().T  # rows = runs
    df["sector_hhi"] = (sector_weights ** 2).sum(axis=1)

    keep = ["run_id", "condition_id", "condition_name", "model", "model_label",
            "hhi", "sector_hhi", "portfolio_beta", "portfolio_vol",
            "return_12m", "breadth"]
    out = df[keep].copy()

    sec_pct = sector_weights * 100
    sec_pct.columns = [f"sector_{c}" for c in sec_pct.columns]
    out = pd.concat([out.reset_index(drop=True),
                     sec_pct.reset_index(drop=True)], axis=1)

    out.to_csv(METRICS_CSV, index=False)
    print(f"  Saved {METRICS_CSV}  ({len(out)} runs)")
    return out


def grouped_point_plot(metrics: pd.DataFrame, col: str, ylabel: str,
                       title: str, fname: str):
    """Mean ± SEM per (condition, model). Models grouped within each condition."""
    g = metrics.groupby(["condition_name", "model_label"])[col] \
               .agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"])

    fig, ax = plt.subplots(figsize=(12, 6.5))
    n_models = len(MODEL_ORDER)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for i, m in enumerate(MODEL_ORDER):
        sub = g[g["model_label"] == m].set_index("condition_name").reindex(CONDITION_ORDER)
        if sub["mean"].isna().all():
            continue
        x = np.arange(len(CONDITION_ORDER)) + offsets[i]
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o",
                    color=MODEL_PALETTE[m], ecolor=MODEL_PALETTE[m],
                    capsize=4, markersize=10, linewidth=2,
                    label=MODEL_LABELS[m])

    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Model", loc="best", frameon=True)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def sector_compare_plot(metrics: pd.DataFrame, fname: str):
    """
    Per-model × per-condition stacked bar chart.

    One subpanel per model. Each subpanel shows 4 stacked bars (one per
    condition), so the viewer can read off how each model's sector mix
    shifts with the prompt — crucially without ever pooling the
    technical-Claude run together with technical-OpenAI etc.
    """
    # Real sector columns only — exclude `sector_hhi`, which is a metric, not
    # a sector. (Earlier versions of this code globbed `sector_*` and the
    # legend ended up containing an "hhi" entry.)
    sector_cols = [c for c in metrics.columns
                   if c.startswith("sector_") and c != "sector_hhi"]
    if not sector_cols:
        print("  (no sector columns; skipping sector comparison)")
        return

    sector_names = [c.replace("sector_", "") for c in sector_cols]
    # Stable color mapping — each sector gets the same color in every panel.
    palette = sns.color_palette("tab10", n_colors=len(SECTOR_ORDER))
    color_map = dict(zip(SECTOR_ORDER, palette))
    ordered_sectors = [s for s in SECTOR_ORDER if s in sector_names] + \
                      [s for s in sector_names if s not in SECTOR_ORDER]

    models_present = [m for m in MODEL_ORDER
                      if m in metrics["model_label"].unique()]
    fig, axes = plt.subplots(1, len(models_present),
                             figsize=(6.0 * len(models_present), 6.5),
                             sharey=True)
    if len(models_present) == 1:
        axes = [axes]

    for ax, model_label in zip(axes, models_present):
        sub = metrics[metrics["model_label"] == model_label]
        means = sub.groupby("condition_name")[sector_cols].mean()
        means.columns = [c.replace("sector_", "") for c in means.columns]
        means = means.reindex(CONDITION_ORDER)
        means = means[ordered_sectors]

        bottom = np.zeros(len(means))
        for sector in ordered_sectors:
            ax.bar(means.index, means[sector], bottom=bottom,
                   color=color_map[sector], edgecolor="black",
                   linewidth=0.4, label=sector)
            bottom = bottom + means[sector].values

        ax.set_title(MODEL_LABELS.get(model_label, model_label),
                     fontweight="bold", fontsize=14)
        ax.set_xlabel("Condition")
        if ax is axes[0]:
            ax.set_ylabel("Mean allocation (%)")
        ax.set_ylim(0, 100)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")

    # One legend for the figure
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Sector",
               bbox_to_anchor=(1.005, 0.5), loc="center left", frameon=True)

    fig.suptitle("Sector allocation by model × condition",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def cross_model_forest(metrics: pd.DataFrame, fname: str):
    """
    Forest plot of treatment effects (diff-in-means vs control with 95% CIs)
    for every (model, treatment, outcome) cell. 4 outcomes × 3 treatments
    × 3 models = 36 effect cells displayed as a 4-panel figure (one panel
    per outcome), with rows = treatments and color = model.
    """
    outcomes = [
        ("hhi",            "HHI"),
        ("sector_hhi",     "Sector HHI"),
        ("portfolio_beta", "Portfolio β"),
        ("portfolio_vol",  "Portfolio vol (%)"),
    ]
    treatments = ["fundamental", "technical", "combined"]

    rows = []
    for outcome, _ in outcomes:
        for model in MODEL_ORDER:
            sub = metrics[metrics["model_label"] == model]
            if sub.empty:
                continue
            ctrl_vals = sub.loc[sub["condition_name"] == "control", outcome].dropna().values
            for tr in treatments:
                tr_vals = sub.loc[sub["condition_name"] == tr, outcome].dropna().values
                if len(ctrl_vals) < 2 or len(tr_vals) < 2:
                    continue
                m_diff = tr_vals.mean() - ctrl_vals.mean()
                # Welch SE
                se = np.sqrt(tr_vals.var(ddof=1) / len(tr_vals)
                             + ctrl_vals.var(ddof=1) / len(ctrl_vals))
                rows.append({
                    "outcome":  outcome,
                    "model":    model,
                    "treatment": tr,
                    "diff":     m_diff,
                    "ci_lo":    m_diff - 1.96 * se,
                    "ci_hi":    m_diff + 1.96 * se,
                })
    eff = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(outcomes),
                             figsize=(5.0 * len(outcomes), 6.0),
                             sharey=True)
    for ax, (outcome, label) in zip(axes, outcomes):
        sub = eff[eff["outcome"] == outcome]
        # y positions: each treatment occupies a band; within band, three
        # offset rows (one per model)
        y_per_tr = {tr: i for i, tr in enumerate(treatments)}
        offsets = {m: o for m, o in zip(MODEL_ORDER, np.linspace(-0.25, 0.25, len(MODEL_ORDER)))}
        for _, r in sub.iterrows():
            y = y_per_tr[r["treatment"]] + offsets[r["model"]]
            ax.errorbar(r["diff"], y,
                        xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]],
                        fmt="o", color=MODEL_PALETTE[r["model"]],
                        capsize=3, markersize=8, linewidth=2)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(treatments)))
        ax.set_yticklabels(treatments)
        ax.set_xlabel("Difference vs control")
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(-0.5, len(treatments) - 0.5)

    # Shared legend
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                          color=MODEL_PALETTE[m], label=MODEL_LABELS[m])
               for m in MODEL_ORDER if m in eff["model"].unique()]
    fig.legend(handles=handles, title="Model", bbox_to_anchor=(1.005, 0.5),
               loc="center left", frameon=True)

    fig.suptitle("Treatment effects vs control by model (95% CI)",
                 fontweight="bold", fontsize=15, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    print("Computing multi-AI metrics...")
    metrics = compute_metrics()

    print("\n── Mean by (model, condition) ──")
    summary = metrics.groupby(["model_label", "condition_name"]).agg(
        n=("run_id", "count"),
        hhi=("hhi", "mean"),
        beta=("portfolio_beta", "mean"),
        ret_12m=("return_12m", "mean"),
    ).round(3)
    print(summary.to_string())

    grouped_point_plot(metrics, "hhi",
                       "Portfolio HHI",
                       "Concentration by condition × model (mean ± SEM)",
                       "fig9_multi_ai_hhi.png")
    # Note: we deliberately do NOT write fig10 here. The trailing 12-month
    # return (return_12m from master_dataset) is what each LLM SAW in its
    # technical prompt — it's an input, not a backtest. The genuine
    # forward-looking realized return (April 2025 → April 2026) is computed
    # in analysis/realized_backtest.py, which writes fig10_multi_ai_returns.png
    # using forward_prices.csv.
    sector_compare_plot(metrics, "fig11_multi_ai_sectors.png")
    cross_model_forest(metrics, "fig12_multi_ai_treatment_effects.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
