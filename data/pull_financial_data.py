"""
pull_financial_data.py
======================
Sections 1B & 1C of the LLM Portfolio Causal Inference Project Checklist.

1B: Pull Fundamental Data (snapshot metrics + trailing growth rates)
1C: Pull Technical Data (rolling/time-series derived metrics)

Design decisions (informed by LLM-finance literature):
------------------------------------------------------
- FUNDAMENTALS are presented as *snapshot* values (market cap, P/E, D/E, total debt)
  plus *trailing growth rates* (YoY revenue growth, YoY earnings growth).
  This mirrors how a human fundamental analyst would brief someone: "here's where the
  company stands today, and here's the recent trajectory."  LLMs interpret these well
  because the numbers are self-contained and don't require the model to compute trends.

- TECHNICALS are presented as *derived rolling metrics* (returns over multiple horizons,
  annualized volatility, beta, SMA levels, and MA-crossover signals).
  A rolling window is the natural representation for technical analysis — it captures
  momentum, trend, and risk in a compact form the LLM can reason about directly.

Why NOT all-rolling for fundamentals?
  Fundamental metrics like P/E and debt-to-equity are inherently point-in-time ratios.
  Rolling them would be noisy and semantically confusing to an LLM (e.g., a "rolling P/E"
  mixes quarterly earnings with daily prices in a misleading way). The literature (e.g.,
  StockBench, MarketSenseAI) presents fundamentals as current snapshots + growth rates.

Output files:
  - data/stock_universe.csv
  - data/fundamental_data.csv
  - data/technical_data.csv
  - data/master_dataset.csv
"""

import os
import warnings
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# yfinance imported conditionally in main block when --live is used

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stock universe: 20 large-cap U.S. stocks across 7 sectors
STOCK_UNIVERSE = {
    # Technology
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology"},
    # Healthcare
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth Group", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare"},
    # Financials
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "GS": {"name": "Goldman Sachs", "sector": "Financials"},
    "V": {"name": "Visa Inc.", "sector": "Financials"},
    # Energy
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
    "CVX": {"name": "Chevron Corp.", "sector": "Energy"},
    # Consumer Goods
    "PG": {"name": "Procter & Gamble", "sector": "Consumer Goods"},
    "KO": {"name": "Coca-Cola Co.", "sector": "Consumer Goods"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Goods"},
    # Industrials
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrials"},
    "HON": {"name": "Honeywell International", "sector": "Industrials"},
    "BA": {"name": "Boeing Co.", "sector": "Industrials"},
    # Additional diversification
    "LLY": {"name": "Eli Lilly & Co.", "sector": "Healthcare"},
    "MMM": {"name": "3M Company", "sector": "Industrials"},
}

TICKERS = list(STOCK_UNIVERSE.keys())


# ============================================================================
# PHASE 1A: Save Stock Universe
# ============================================================================

def save_stock_universe():
    """Create and save the stock universe reference table."""
    rows = [
        {"ticker": t, "company_name": info["name"], "sector": info["sector"]}
        for t, info in STOCK_UNIVERSE.items()
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "stock_universe.csv")
    df.to_csv(path, index=False)
    print(f"[1A] Saved stock universe ({len(df)} stocks) -> {path}")
    print(f"     Sectors: {df['sector'].value_counts().to_dict()}")
    return df


# ============================================================================
# PHASE 1B: Pull Fundamental Data
# ============================================================================

def pull_fundamental_data():
    """
    Pull fundamental indicators for each stock.

    Metrics collected (snapshot + trailing growth):
    -----------------------------------------------
    Snapshot (current values):
      - market_cap: Total market capitalization in USD
      - pe_ratio: Trailing P/E ratio (price / trailing 12-month EPS)
      - debt_to_equity: Total debt / total equity ratio
      - total_debt: Total debt in USD

    Trailing Growth Rates:
      - revenue_growth_yoy: Year-over-year quarterly revenue growth (%)
        Computed as: (most recent quarter revenue - same quarter last year) / same quarter last year
      - earnings_growth_yoy: Year-over-year quarterly earnings growth (%)
        Computed similarly using net income

    Why these metrics?
      These are the core fundamentals that analysts use to assess valuation (P/E),
      financial health (leverage), scale (market cap), and trajectory (growth rates).
      The LLM receives a compact "fundamental profile" for each stock that captures
      both where the company stands and where it's heading.
    """
    print("\n[1B] Pulling fundamental data...")
    records = []

    for ticker_str in TICKERS:
        print(f"  Fetching fundamentals for {ticker_str}...", end=" ")
        try:
            ticker = yf.Ticker(ticker_str)
            info = ticker.info

            # --- Snapshot metrics ---
            market_cap = info.get("marketCap", None)
            pe_ratio = info.get("trailingPE", None)
            debt_to_equity = info.get("debtToEquity", None)
            total_debt = info.get("totalDebt", None)

            # --- Trailing growth rates ---
            # Revenue growth: use yfinance's reported trailing value, or compute from financials
            revenue_growth_yoy = info.get("revenueGrowth", None)
            if revenue_growth_yoy is not None:
                revenue_growth_yoy = round(revenue_growth_yoy * 100, 2)  # Convert to %

            # Earnings growth: use yfinance's reported trailing value, or compute from financials
            earnings_growth_yoy = info.get("earningsGrowth", None)
            if earnings_growth_yoy is not None:
                earnings_growth_yoy = round(earnings_growth_yoy * 100, 2)  # Convert to %

            # If yfinance didn't provide growth, try computing from quarterly financials
            if revenue_growth_yoy is None or earnings_growth_yoy is None:
                try:
                    q_fin = ticker.quarterly_financials
                    if q_fin is not None and not q_fin.empty:
                        # Revenue growth from quarterly financials
                        if revenue_growth_yoy is None:
                            rev_row = None
                            for label in ["Total Revenue", "Revenue"]:
                                if label in q_fin.index:
                                    rev_row = q_fin.loc[label]
                                    break
                            if rev_row is not None and len(rev_row) >= 5:
                                # Compare most recent quarter to same quarter last year
                                recent = rev_row.iloc[0]
                                year_ago = rev_row.iloc[4]
                                if pd.notna(recent) and pd.notna(year_ago) and year_ago != 0:
                                    revenue_growth_yoy = round(
                                        ((recent - year_ago) / abs(year_ago)) * 100, 2
                                    )

                        # Earnings growth from quarterly financials
                        if earnings_growth_yoy is None:
                            earn_row = None
                            for label in ["Net Income", "Net Income Common Stockholders"]:
                                if label in q_fin.index:
                                    earn_row = q_fin.loc[label]
                                    break
                            if earn_row is not None and len(earn_row) >= 5:
                                recent = earn_row.iloc[0]
                                year_ago = earn_row.iloc[4]
                                if pd.notna(recent) and pd.notna(year_ago) and year_ago != 0:
                                    earnings_growth_yoy = round(
                                        ((recent - year_ago) / abs(year_ago)) * 100, 2
                                    )
                except Exception:
                    pass  # Growth stays None if financials aren't available

            record = {
                "ticker": ticker_str,
                "market_cap": market_cap,
                "pe_ratio": round(pe_ratio, 2) if pe_ratio is not None else None,
                "revenue_growth_yoy": revenue_growth_yoy,
                "earnings_growth_yoy": earnings_growth_yoy,
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
                "total_debt": total_debt,
            }
            records.append(record)
            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")
            records.append({
                "ticker": ticker_str,
                "market_cap": None, "pe_ratio": None,
                "revenue_growth_yoy": None, "earnings_growth_yoy": None,
                "debt_to_equity": None, "total_debt": None,
            })

        time.sleep(0.5)  # Rate limiting

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "fundamental_data.csv")
    df.to_csv(path, index=False)

    # Report on data quality
    print(f"\n[1B] Saved fundamental data -> {path}")
    print(f"     Records: {len(df)}")
    missing = df.isnull().sum()
    if missing.any():
        print(f"     Missing values:\n{missing[missing > 0].to_string()}")
    else:
        print("     No missing values!")

    return df


# ============================================================================
# PHASE 1C: Pull Technical Data
# ============================================================================

def pull_technical_data():
    """
    Pull technical indicators derived from daily price history.

    Metrics collected (all rolling/derived):
    ----------------------------------------
    Returns (momentum signals):
      - return_1m:  1-month (21 trading days) cumulative return
      - return_3m:  3-month (63 trading days) cumulative return
      - return_6m:  6-month (126 trading days) cumulative return
      - return_12m: 12-month (252 trading days) cumulative return

    Risk metrics:
      - volatility: Annualized historical volatility (std of daily log returns × √252)
        over the trailing 252 trading days
      - beta: Slope of regression of stock daily returns vs S&P 500 daily returns
        over the trailing 252 trading days. Measures systematic risk.

    Moving averages (trend signals):
      - sma_20: 20-day simple moving average (current value)
      - sma_50: 50-day simple moving average
      - sma_200: 200-day simple moving average
      - price_vs_sma50: Current price as % above/below 50-day SMA
        (positive = price above SMA = bullish signal)
      - ma_signal: Categorical trend signal derived from MA crossovers:
        "strong_bullish" = price > SMA50 > SMA200 (golden cross territory)
        "bullish" = price > SMA50 but SMA50 < SMA200
        "bearish" = price < SMA50 but SMA50 > SMA200
        "strong_bearish" = price < SMA50 < SMA200 (death cross territory)

    Why these metrics?
      Returns capture momentum at multiple horizons — LLMs can see whether a stock
      has short-term vs long-term momentum divergence. Volatility and beta give the
      LLM a sense of risk. Moving averages and their crossover signals are the most
      widely used technical indicators and are easily interpretable in text form.
    """
    print("\n[1C] Pulling technical data...")

    # First, get S&P 500 data for beta calculation
    print("  Fetching S&P 500 benchmark data...")
    sp500 = yf.Ticker("^GSPC")
    sp500_hist = sp500.history(period="2y")

    if sp500_hist.empty:
        raise ValueError("Could not fetch S&P 500 data. Check yfinance connection.")

    sp500_returns = np.log(sp500_hist["Close"] / sp500_hist["Close"].shift(1)).dropna()
    print(f"  S&P 500 data: {len(sp500_hist)} trading days")

    records = []

    for ticker_str in TICKERS:
        print(f"  Fetching technicals for {ticker_str}...", end=" ")
        try:
            ticker = yf.Ticker(ticker_str)
            hist = ticker.history(period="2y")  # Need 2 years for 252-day calculations

            if hist.empty or len(hist) < 252:
                print(f"WARN: insufficient history ({len(hist)} days)")
                records.append({"ticker": ticker_str})
                continue

            close = hist["Close"]
            current_price = close.iloc[-1]

            # --- Returns at multiple horizons ---
            def calc_return(series, days):
                if len(series) >= days:
                    return round(((series.iloc[-1] / series.iloc[-days]) - 1) * 100, 2)
                return None

            return_1m = calc_return(close, 21)
            return_3m = calc_return(close, 63)
            return_6m = calc_return(close, 126)
            return_12m = calc_return(close, 252)

            # --- Historical Volatility (annualized) ---
            log_returns = np.log(close / close.shift(1)).dropna()
            # Use trailing 252 days
            trailing_log_returns = log_returns.tail(252)
            volatility = round(trailing_log_returns.std() * np.sqrt(252) * 100, 2)

            # --- Beta (vs S&P 500) ---
            # Align dates between stock and S&P 500
            stock_returns = log_returns.tail(252)
            common_dates = stock_returns.index.intersection(sp500_returns.index)

            if len(common_dates) >= 100:
                sr = stock_returns.loc[common_dates].values
                mr = sp500_returns.loc[common_dates].values
                # Beta = Cov(stock, market) / Var(market)
                cov_matrix = np.cov(sr, mr)
                beta = round(cov_matrix[0, 1] / cov_matrix[1, 1], 2)
            else:
                # Fallback to yfinance info beta
                beta_info = ticker.info.get("beta", None)
                beta = round(beta_info, 2) if beta_info is not None else None

            # --- Moving Averages ---
            sma_20 = round(close.tail(20).mean(), 2)
            sma_50 = round(close.tail(50).mean(), 2)
            sma_200 = round(close.tail(200).mean(), 2)

            # Price relative to SMA50 (% above/below)
            price_vs_sma50 = round(((current_price / sma_50) - 1) * 100, 2)

            # MA crossover signal
            if current_price > sma_50 and sma_50 > sma_200:
                ma_signal = "strong_bullish"
            elif current_price > sma_50 and sma_50 <= sma_200:
                ma_signal = "bullish"
            elif current_price <= sma_50 and sma_50 > sma_200:
                ma_signal = "bearish"
            else:
                ma_signal = "strong_bearish"

            record = {
                "ticker": ticker_str,
                "current_price": round(current_price, 2),
                "return_1m": return_1m,
                "return_3m": return_3m,
                "return_6m": return_6m,
                "return_12m": return_12m,
                "volatility": volatility,
                "beta": beta,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "price_vs_sma50": price_vs_sma50,
                "ma_signal": ma_signal,
            }
            records.append(record)
            print("OK")

        except Exception as e:
            print(f"ERROR: {e}")
            records.append({"ticker": ticker_str})

        time.sleep(0.5)  # Rate limiting

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "technical_data.csv")
    df.to_csv(path, index=False)

    print(f"\n[1C] Saved technical data -> {path}")
    print(f"     Records: {len(df)}")
    missing = df.isnull().sum()
    if missing.any():
        print(f"     Missing values:\n{missing[missing > 0].to_string()}")
    else:
        print("     No missing values!")

    return df


# ============================================================================
# PHASE 1D: Merge & Validate
# ============================================================================

def merge_and_validate(universe_df, fundamental_df, technical_df):
    """Merge all data into a master dataset and run sanity checks."""
    print("\n[1D] Merging datasets...")

    master = universe_df.merge(fundamental_df, on="ticker", how="left")
    master = master.merge(technical_df, on="ticker", how="left")

    path = os.path.join(OUTPUT_DIR, "master_dataset.csv")
    master.to_csv(path, index=False)
    print(f"     Saved master dataset -> {path}")
    print(f"     Shape: {master.shape}")

    # --- Sanity checks ---
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check P/E ratios
    pe = master["pe_ratio"].dropna()
    if len(pe) > 0:
        print(f"\n  P/E Ratio range: {pe.min():.1f} to {pe.max():.1f}")
        outliers = pe[(pe < 0) | (pe > 200)]
        if len(outliers) > 0:
            tickers_with_outliers = master.loc[outliers.index, "ticker"].tolist()
            print(f"  ⚠ P/E outliers (negative or >200): {tickers_with_outliers}")

    # Check beta
    beta = master["beta"].dropna()
    if len(beta) > 0:
        print(f"  Beta range: {beta.min():.2f} to {beta.max():.2f}")
        outside = beta[(beta < 0) | (beta > 3)]
        if len(outside) > 0:
            tickers_outside = master.loc[outside.index, "ticker"].tolist()
            print(f"  ⚠ Beta outside typical range (0-3): {tickers_outside}")

    # Check volatility
    vol = master["volatility"].dropna()
    if len(vol) > 0:
        print(f"  Volatility range: {vol.min():.1f}% to {vol.max():.1f}%")

    # Overall missing data report
    print(f"\n  Missing values per column:")
    missing = master.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"    {col}: {count}/{len(master)} missing")
    if missing.sum() == 0:
        print("    None! All data complete.")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    numeric_cols = master.select_dtypes(include=[np.number]).columns
    print(master[numeric_cols].describe().round(2).to_string())

    return master


# ============================================================================
# DEMO MODE: Generate realistic sample data when yfinance is unavailable
# ============================================================================

def generate_sample_fundamental_data():
    """
    Generate realistic sample fundamental data for testing/demo purposes.
    Values are based on approximate real ranges for these companies.
    Replace with real data by running in live mode (python pull_financial_data.py --live).
    """
    print("\n[1B] Generating sample fundamental data (demo mode)...")
    np.random.seed(42)

    # Realistic approximate values by sector
    sector_profiles = {
        "Technology": {"pe_range": (25, 60), "de_range": (50, 200), "rev_growth": (5, 25), "earn_growth": (5, 30)},
        "Healthcare": {"pe_range": (15, 45), "de_range": (30, 150), "rev_growth": (-5, 20), "earn_growth": (-10, 25)},
        "Financials": {"pe_range": (10, 20), "de_range": (100, 400), "rev_growth": (2, 15), "earn_growth": (3, 20)},
        "Energy": {"pe_range": (8, 18), "de_range": (20, 80), "rev_growth": (-10, 15), "earn_growth": (-15, 20)},
        "Consumer Goods": {"pe_range": (20, 40), "de_range": (50, 200), "rev_growth": (2, 15), "earn_growth": (3, 18)},
        "Industrials": {"pe_range": (15, 30), "de_range": (60, 250), "rev_growth": (-3, 12), "earn_growth": (-5, 15)},
    }

    # Approximate market caps (in billions USD)
    market_caps = {
        "AAPL": 3200, "MSFT": 3100, "GOOGL": 2100, "NVDA": 2800,
        "JNJ": 380, "UNH": 520, "PFE": 150, "JPM": 650,
        "GS": 180, "V": 580, "XOM": 480, "CVX": 280,
        "PG": 390, "KO": 260, "AMZN": 2000, "CAT": 180,
        "HON": 140, "BA": 130, "LLY": 750, "MMM": 70,
    }

    records = []
    for ticker_str, info in STOCK_UNIVERSE.items():
        sector = info["sector"]
        profile = sector_profiles[sector]

        pe = round(np.random.uniform(*profile["pe_range"]), 2)
        de = round(np.random.uniform(*profile["de_range"]), 2)
        rev_g = round(np.random.uniform(*profile["rev_growth"]), 2)
        earn_g = round(np.random.uniform(*profile["earn_growth"]), 2)
        mcap = market_caps.get(ticker_str, 200) * 1e9
        total_debt = mcap * (de / 100) * np.random.uniform(0.3, 0.6)

        records.append({
            "ticker": ticker_str,
            "market_cap": int(mcap),
            "pe_ratio": pe,
            "revenue_growth_yoy": rev_g,
            "earnings_growth_yoy": earn_g,
            "debt_to_equity": de,
            "total_debt": int(total_debt),
        })

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "fundamental_data.csv")
    df.to_csv(path, index=False)
    print(f"[1B] Saved sample fundamental data -> {path}")
    return df


def generate_sample_technical_data():
    """
    Generate realistic sample technical data for testing/demo purposes.
    Replace with real data by running in live mode.
    """
    print("\n[1C] Generating sample technical data (demo mode)...")
    np.random.seed(123)

    records = []
    for ticker_str, info in STOCK_UNIVERSE.items():
        sector = info["sector"]

        # Generate a realistic current price
        base_prices = {
            "AAPL": 195, "MSFT": 420, "GOOGL": 175, "NVDA": 880,
            "JNJ": 155, "UNH": 520, "PFE": 28, "JPM": 200,
            "GS": 470, "V": 280, "XOM": 110, "CVX": 155,
            "PG": 165, "KO": 60, "AMZN": 185, "CAT": 360,
            "HON": 200, "BA": 180, "LLY": 780, "MMM": 105,
        }
        price = base_prices.get(ticker_str, 150) * np.random.uniform(0.95, 1.05)

        # Returns with some correlation structure
        base_return = np.random.normal(0, 8)
        return_1m = round(base_return * 0.3 + np.random.normal(0, 3), 2)
        return_3m = round(base_return * 0.6 + np.random.normal(0, 5), 2)
        return_6m = round(base_return * 0.8 + np.random.normal(0, 8), 2)
        return_12m = round(base_return * 1.2 + np.random.normal(0, 12), 2)

        # Volatility varies by sector
        vol_base = {"Technology": 30, "Healthcare": 25, "Financials": 22,
                     "Energy": 28, "Consumer Goods": 18, "Industrials": 24}
        volatility = round(vol_base.get(sector, 25) + np.random.normal(0, 5), 2)

        # Beta
        beta_base = {"Technology": 1.2, "Healthcare": 0.8, "Financials": 1.1,
                      "Energy": 1.0, "Consumer Goods": 0.7, "Industrials": 1.0}
        beta = round(beta_base.get(sector, 1.0) + np.random.normal(0, 0.2), 2)

        # Moving averages
        sma_20 = round(price * np.random.uniform(0.97, 1.03), 2)
        sma_50 = round(price * np.random.uniform(0.94, 1.06), 2)
        sma_200 = round(price * np.random.uniform(0.88, 1.12), 2)

        price_vs_sma50 = round(((price / sma_50) - 1) * 100, 2)

        # MA signal
        if price > sma_50 and sma_50 > sma_200:
            ma_signal = "strong_bullish"
        elif price > sma_50 and sma_50 <= sma_200:
            ma_signal = "bullish"
        elif price <= sma_50 and sma_50 > sma_200:
            ma_signal = "bearish"
        else:
            ma_signal = "strong_bearish"

        records.append({
            "ticker": ticker_str,
            "current_price": round(price, 2),
            "return_1m": return_1m,
            "return_3m": return_3m,
            "return_6m": return_6m,
            "return_12m": return_12m,
            "volatility": volatility,
            "beta": beta,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "price_vs_sma50": price_vs_sma50,
            "ma_signal": ma_signal,
        })

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "technical_data.csv")
    df.to_csv(path, index=False)
    print(f"[1C] Saved sample technical data -> {path}")
    return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    live_mode = "--live" in sys.argv

    print("=" * 60)
    print("LLM PORTFOLIO PROJECT — DATA COLLECTION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {'LIVE (yfinance)' if live_mode else 'DEMO (sample data)'}")
    print("=" * 60)

    if not live_mode:
        print("\n⚠ Running in DEMO mode with realistic sample data.")
        print("  To pull real data from Yahoo Finance, run:")
        print("  python pull_financial_data.py --live\n")

    universe_df = save_stock_universe()

    if live_mode:
        try:
            import yfinance as yf
        except ImportError:
            print("\nERROR: yfinance not installed. Run: pip install yfinance")
            print("Falling back to demo mode.\n")
            live_mode = False

    if live_mode:
        fundamental_df = pull_fundamental_data()
        technical_df = pull_technical_data()
    else:
        fundamental_df = generate_sample_fundamental_data()
        technical_df = generate_sample_technical_data()

    master_df = merge_and_validate(universe_df, fundamental_df, technical_df)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nFiles created in '{OUTPUT_DIR}/':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")

    print("\n✅ Ready for Phase 2: Prompt Engineering & LLM Pipeline")