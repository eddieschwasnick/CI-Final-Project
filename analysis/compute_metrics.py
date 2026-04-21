"""
compute_metrics.py
------------------
Phase 4: Compute portfolio-level metrics for every run in experiment_results.csv.

Metrics:
  - HHI             : sum of squared weights (decimal form) — concentration
  - sector_HHI      : HHI applied to sector-aggregated weights
  - portfolio_beta  : weighted average of stock betas
  - portfolio_vol   : weighted average of stock volatilities
  - breadth         : number of stocks with weight > 0%

Output: outputs/portfolio_metrics.csv
Also:   outputs/sector_weights.csv (per-run sector allocation, for Phase 5/6)
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV       = os.path.join(BASE_DIR, "outputs", "experiment_results.csv")
MASTER_CSV        = os.path.join(BASE_DIR, "data", "csv_files", "master_dataset.csv")
METRICS_CSV       = os.path.join(BASE_DIR, "outputs", "portfolio_metrics.csv")
SECTOR_WEIGHTS_CSV = os.path.join(BASE_DIR, "outputs", "sector_weights.csv")

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "JNJ", "UNH", "PFE", "JPM", "GS",
    "XOM", "CVX", "PG", "KO", "CAT", "HON", "AMZN", "NVDA", "V",
    "LLY", "BA", "MMM",
]


def load_stock_metadata() -> pd.DataFrame:
    """Load ticker-level metadata (sector, beta, volatility) from master_dataset."""
    master = pd.read_csv(MASTER_CSV)
    cols = ["ticker", "sector", "beta", "volatility"]
    return master[cols].set_index("ticker")


def compute_run_metrics(row: pd.Series, meta: pd.DataFrame) -> dict:
    """
    Given one row of experiment_results.csv (one portfolio),
    return a dict of computed portfolio metrics.
    """
    # Convert weights to decimals (weights in CSV are percentages, sum ~100)
    weights_pct = row[STOCK_UNIVERSE].astype(float)
    weights_dec = weights_pct / 100.0  # fractions summing to 1.0

    # 1. HHI on stock-level weights (decimal)
    hhi = float((weights_dec ** 2).sum())

    # 2. Sector-level HHI: sum weights by sector, then square and sum
    sector_weights = weights_dec.groupby(meta.loc[STOCK_UNIVERSE, "sector"]).sum()
    sector_hhi = float((sector_weights ** 2).sum())

    # 3. Portfolio beta = sum(weight_i * beta_i)
    portfolio_beta = float((weights_dec * meta.loc[STOCK_UNIVERSE, "beta"]).sum())

    # 4. Portfolio volatility (simplified weighted average)
    portfolio_vol = float((weights_dec * meta.loc[STOCK_UNIVERSE, "volatility"]).sum())

    # 5. Breadth — count of non-zero holdings (tolerance for floating-point zeros)
    breadth = int((weights_pct > 0.01).sum())

    return {
        "hhi":            hhi,
        "sector_hhi":     sector_hhi,
        "portfolio_beta": portfolio_beta,
        "portfolio_vol":  portfolio_vol,
        "breadth":        breadth,
        "sector_weights": sector_weights.to_dict(),
    }


def main():
    print("Loading experiment_results.csv...")
    df = pd.read_csv(RESULTS_CSV)
    print(f"  {len(df)} runs loaded.")

    # Filter to only parsed-successfully runs
    df = df[df["parse_success"] == True].copy()
    print(f"  {len(df)} runs with parse_success=True retained.")

    meta = load_stock_metadata()
    print(f"  Loaded metadata for {len(meta)} tickers.")

    metric_rows = []
    sector_rows = []

    for _, row in df.iterrows():
        m = compute_run_metrics(row, meta)

        metric_rows.append({
            "run_id":         row["run_id"],
            "condition_id":   row["condition_id"],
            "condition_name": row["condition_name"],
            "hhi":            m["hhi"],
            "sector_hhi":     m["sector_hhi"],
            "portfolio_beta": m["portfolio_beta"],
            "portfolio_vol":  m["portfolio_vol"],
            "breadth":        m["breadth"],
        })

        sec_row = {
            "run_id":         row["run_id"],
            "condition_id":   row["condition_id"],
            "condition_name": row["condition_name"],
        }
        sec_row.update(m["sector_weights"])
        sector_rows.append(sec_row)

    metrics_df = pd.DataFrame(metric_rows)
    sector_df  = pd.DataFrame(sector_rows).fillna(0.0)

    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    metrics_df.to_csv(METRICS_CSV, index=False)
    sector_df.to_csv(SECTOR_WEIGHTS_CSV, index=False)

    print(f"\nSaved {METRICS_CSV}")
    print(f"Saved {SECTOR_WEIGHTS_CSV}")

    print("\n── Summary by condition ──")
    summary = metrics_df.groupby("condition_name").agg(
        n=("run_id", "count"),
        hhi_mean=("hhi", "mean"),
        hhi_std=("hhi", "std"),
        sector_hhi_mean=("sector_hhi", "mean"),
        beta_mean=("portfolio_beta", "mean"),
        vol_mean=("portfolio_vol", "mean"),
        breadth_mean=("breadth", "mean"),
    ).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
