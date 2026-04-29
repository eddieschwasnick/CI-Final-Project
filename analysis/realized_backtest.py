"""
realized_backtest.py
--------------------
Real forward-looking backtest: each portfolio was built from April 2025 prompts
and we now compute its actual realized return from April 27, 2025 to today
using forward_prices.csv.

For every parsed portfolio in outputs/multi_ai_results.csv:
    realized_return = Σ  w_i (fraction) × per-ticker return_pct from forward_prices

Inputs
  outputs/multi_ai_results.csv         from run_multi_ai_experiment.py
  data/csv_files/forward_prices.csv    from `pull_financial_data.py --as-of …`

Outputs
  outputs/realized_returns.csv
  reports/figures/fig8_portfolio_returns.png       PNAS two-panel by condition
  reports/figures/fig10_multi_ai_returns.png       grouped points by condition × model
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE_DIR, "outputs", "multi_ai_results.csv")
LEGACY_CSV  = os.path.join(BASE_DIR, "outputs", "experiment_results.csv")
FORWARD_CSV = os.path.join(BASE_DIR, "data", "csv_files", "forward_prices.csv")
OUT_CSV     = os.path.join(BASE_DIR, "outputs", "realized_returns.csv")
FIG_DIR     = os.path.join(BASE_DIR, "reports", "figures")

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
MODEL_ORDER  = ["claude", "openai", "gemini"]
MODEL_LABELS = {
    "claude": "Claude Sonnet 4.6",
    "openai": "OpenAI GPT-4o",
    "gemini": "Gemini 2.5 Pro",
}
MODEL_PALETTE = {
    "claude": "#d97706",
    "openai": "#10a37f",
    "gemini": "#4285f4",
}

sns.set_context("talk", font_scale=1.25)
sns.set_style("whitegrid")
os.makedirs(FIG_DIR, exist_ok=True)


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["parse_success"] == True].copy()
    if "model_label" not in df.columns:
        df["model_label"] = "claude"
    return df


def compute_realized(results: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    fwd = fwd.set_index("ticker")
    if fwd["return_pct"].isna().any():
        bad = fwd[fwd["return_pct"].isna()].index.tolist()
        raise SystemExit(f"forward_prices.csv missing return_pct for: {bad}. "
                         f"Re-run pull_financial_data.py --as-of …")
    rets = fwd.loc[STOCK_UNIVERSE, "return_pct"].astype(float)

    weights = results[STOCK_UNIVERSE].astype(float) / 100.0
    realized = weights.values @ rets.values
    out = results[["run_id", "condition_id", "condition_name",
                   "model", "model_label"]].copy()
    out["realized_return_pct"] = realized
    return out


def pnas_two_panel(returns_df: pd.DataFrame, fname: str, title: str):
    g = returns_df.groupby("condition_name")["realized_return_pct"] \
                  .agg(["mean", "std", "count"]).reindex(CONDITION_ORDER)
    sem = g["std"] / np.sqrt(g["count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    y_lo = returns_df["realized_return_pct"].min()
    y_hi = returns_df["realized_return_pct"].max()
    pad  = 0.08 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    ylim = (y_lo - pad, y_hi + pad)
    x_pos = np.arange(len(CONDITION_ORDER))

    ax = axes[0]
    ax.errorbar(x_pos, g["mean"], yerr=sem, fmt="o", color="black",
                ecolor="black", capsize=4, markersize=8, linewidth=2)
    ax.set_xticks(x_pos); ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5); ax.set_ylim(*ylim)
    ax.set_ylabel("Realized 12-mo return (%)")
    ax.set_title("Inferential uncertainty\n(SE only)", fontsize=14)

    ax = axes[1]
    rng = np.random.default_rng(2)
    for i, c in enumerate(CONDITION_ORDER):
        s = returns_df.loc[returns_df["condition_name"] == c,
                           "realized_return_pct"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(s))
        ax.scatter(np.full_like(s, i, dtype=float) + jitter, s,
                   color=PALETTE[c], alpha=0.45, s=45,
                   edgecolor="white", linewidth=0.4, zorder=2)
    ax.errorbar(x_pos, g["mean"], yerr=sem, fmt="o", color="black",
                ecolor="black", capsize=4, markersize=8, linewidth=2, zorder=3)
    ax.axhline(g["mean"].max(), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(g["mean"].min(), color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_pos); ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5); ax.set_ylim(*ylim)
    ax.set_title("Outcome variability\n(SE + points)", fontsize=14)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def grouped_by_model(returns_df: pd.DataFrame, fname: str, title: str):
    g = returns_df.groupby(["condition_name", "model_label"])["realized_return_pct"] \
                  .agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"])

    models_present = [m for m in MODEL_ORDER if m in g["model_label"].unique()]
    if len(models_present) <= 1:
        print(f"  only {len(models_present)} model(s); skipping cross-model figure")
        return

    fig, ax = plt.subplots(figsize=(12, 6.5))
    n = len(models_present)
    width = 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    for i, m in enumerate(models_present):
        sub = g[g["model_label"] == m].set_index("condition_name").reindex(CONDITION_ORDER)
        x = np.arange(len(CONDITION_ORDER)) + offsets[i]
        ax.errorbar(x, sub["mean"], yerr=sub["sem"], fmt="o",
                    color=MODEL_PALETTE[m], ecolor=MODEL_PALETTE[m],
                    capsize=4, markersize=10, linewidth=2,
                    label=MODEL_LABELS[m])
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Realized 12-mo return (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Model", loc="best", frameon=True)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", default=None,
                    help="Override results CSV (defaults to multi_ai_results.csv "
                         "if it exists, else experiment_results.csv).")
    args = ap.parse_args()

    src = args.results_csv
    if src is None:
        src = RESULTS_CSV if os.path.exists(RESULTS_CSV) else LEGACY_CSV
    if not os.path.exists(src):
        raise SystemExit(f"Missing results CSV: {src}")
    if not os.path.exists(FORWARD_CSV):
        raise SystemExit(f"Missing {FORWARD_CSV}. "
                         f"Run: python data/pull_financial_data.py --as-of last-year")

    print(f"Results CSV : {src}")
    print(f"Forward CSV : {FORWARD_CSV}")

    results = load_results(src)
    fwd     = pd.read_csv(FORWARD_CSV)

    realized = compute_realized(results, fwd)
    realized.to_csv(OUT_CSV, index=False)
    print(f"  Saved {OUT_CSV}  ({len(realized)} portfolios)\n")

    print("── Mean realized return by condition (across all models) ──")
    print(realized.groupby("condition_name")["realized_return_pct"]
                  .agg(["mean", "std", "count"])
                  .reindex(CONDITION_ORDER).round(2).to_string())

    if realized["model_label"].nunique() > 1:
        print("\n── Mean realized return by (model, condition) ──")
        print(realized.groupby(["model_label", "condition_name"])["realized_return_pct"]
                      .agg(["mean", "std", "count"])
                      .round(2).to_string())

    asof = pd.read_csv(FORWARD_CSV)["asof_date"].iloc[0]
    today = pd.read_csv(FORWARD_CSV)["today_date"].iloc[0]
    title_suffix = f" ({asof} → {today})"

    pnas_two_panel(realized, "fig8_portfolio_returns.png",
                   "Realized 12-mo portfolio return by condition" + title_suffix)
    grouped_by_model(realized, "fig10_multi_ai_returns.png",
                     "Realized 12-mo return by condition × model" + title_suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
