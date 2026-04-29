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

# Always write into data/csv_files (where the rest of the pipeline reads from),
# regardless of the cwd from which this script is launched.
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_THIS_DIR, "csv_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# When set by the CLI (--as-of YYYY-MM-DD) all `--live` data pulls compute
# metrics as of that historical date instead of today. This is what the backtest
# section needs: the LLM should see the world as it looked at T0, and we will
# compute realized returns from T0 → today separately.
AS_OF_DATE = None  # type: pd.Timestamp | None


def _strip_tz(idx):
    try:
        return idx.tz_localize(None)
    except (AttributeError, TypeError):
        return idx


def _close_on_or_before(close_series: pd.Series, target: pd.Timestamp):
    """(date, close) for the most recent trading day ≤ target."""
    s = close_series.copy()
    s.index = _strip_tz(s.index)
    s = s[s.index <= target]
    if s.empty:
        raise ValueError(f"No price data on or before {target.date()}")
    return s.index[-1], float(s.iloc[-1])

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

def _historical_fundamentals(ticker, ticker_str: str, yf) -> dict:
    """
    Reconstruct fundamental fields AS OF AS_OF_DATE using:
        - quarterly_financials       (revenue, net income → growth, TTM EPS)
        - quarterly_balance_sheet    (total debt, equity → debt-to-equity)
        - get_shares_full            (shares outstanding at T0 → market cap)
        - history                    (price at T0)

    Limitations:
        yfinance only returns ~5 quarters of statements, so for AS_OF_DATE that
        is more than ~12 months in the past some fields fall back to NaN.
        Those rows will print as "N/A" when the prompt builder embeds the table.
    """
    out = {
        "ticker":              ticker_str,
        "market_cap":          None,
        "pe_ratio":            None,
        "revenue_growth_yoy":  None,
        "earnings_growth_yoy": None,
        "debt_to_equity":      None,
        "total_debt":          None,
    }

    # Price at AS_OF_DATE (and shares for market cap)
    hist = ticker.history(period="3y")
    if hist.empty:
        return out
    close = hist["Close"].copy()
    close.index = _strip_tz(close.index)
    try:
        _, price_t0 = _close_on_or_before(close, AS_OF_DATE)
    except ValueError:
        return out

    shares_t0 = None
    try:
        shares_series = ticker.get_shares_full(
            start=(AS_OF_DATE - pd.Timedelta(days=180)).strftime("%Y-%m-%d"),
            end=(AS_OF_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if shares_series is not None and len(shares_series) > 0:
            shares_series.index = _strip_tz(shares_series.index)
            shares_series = shares_series[shares_series.index <= AS_OF_DATE]
            if not shares_series.empty:
                shares_t0 = float(shares_series.iloc[-1])
                out["market_cap"] = int(shares_t0 * price_t0)
    except Exception:
        pass

    # Quarterly income statement → revenue / earnings growth, TTM EPS
    # Quarterly data only goes ~5 quarters back, so for a 1y-old AS-OF this
    # frequently fails — we fall back to annual statements below.
    try:
        qf = ticker.quarterly_financials
        if qf is not None and not qf.empty:
            qf = qf.copy()
            qf.columns = pd.to_datetime(qf.columns)
            qf = qf.loc[:, qf.columns <= AS_OF_DATE].sort_index(axis=1)

            def find_row(labels):
                for L in labels:
                    if L in qf.index:
                        return qf.loc[L]
                return None

            def yoy_q(series):
                if series is None or len(series) < 5:
                    return None
                latest = series.iloc[-1]
                year_ago = series.iloc[-5]
                if pd.isna(latest) or pd.isna(year_ago) or year_ago == 0:
                    return None
                return round(((latest - year_ago) / abs(year_ago)) * 100, 2)

            rev  = find_row(["Total Revenue", "Revenue", "TotalRevenue"])
            earn = find_row(["Net Income", "Net Income Common Stockholders",
                             "NetIncome", "Net Income Continuous Operations"])
            out["revenue_growth_yoy"]  = yoy_q(rev)
            out["earnings_growth_yoy"] = yoy_q(earn)

            if earn is not None and len(earn) >= 4 and shares_t0 and shares_t0 > 0:
                ttm_ni = float(earn.iloc[-4:].sum())
                eps = ttm_ni / shares_t0
                if eps > 0:
                    out["pe_ratio"] = round(price_t0 / eps, 2)
    except Exception:
        pass

    # Annual income statement fallback for any of the 3 fields still missing.
    # yfinance keeps ~4 years of annual data, which is enough for AS-OF dates
    # up to a few years back. We only consider annual periods whose period-end
    # is at least ~60 days before AS_OF_DATE (typical earnings-reporting lag).
    needs_growth_rev  = out["revenue_growth_yoy"] is None
    needs_growth_earn = out["earnings_growth_yoy"] is None
    needs_pe          = out["pe_ratio"] is None
    if needs_growth_rev or needs_growth_earn or needs_pe:
        try:
            af = ticker.income_stmt if hasattr(ticker, "income_stmt") else ticker.financials
            if af is not None and not af.empty:
                af = af.copy()
                af.columns = pd.to_datetime(af.columns)
                cutoff = AS_OF_DATE - pd.Timedelta(days=60)
                af = af.loc[:, af.columns <= cutoff].sort_index(axis=1)

                def find_a(labels):
                    for L in labels:
                        if L in af.index:
                            return af.loc[L]
                    return None

                def yoy_a(series):
                    if series is None or len(series) < 2:
                        return None
                    latest = series.iloc[-1]
                    prior  = series.iloc[-2]
                    if pd.isna(latest) or pd.isna(prior) or prior == 0:
                        return None
                    return round(((latest - prior) / abs(prior)) * 100, 2)

                rev_a  = find_a(["Total Revenue", "Revenue", "TotalRevenue"])
                earn_a = find_a(["Net Income", "Net Income Common Stockholders",
                                 "NetIncome", "Net Income Continuous Operations"])
                if needs_growth_rev:
                    out["revenue_growth_yoy"]  = yoy_a(rev_a)
                if needs_growth_earn:
                    out["earnings_growth_yoy"] = yoy_a(earn_a)
                if needs_pe and earn_a is not None and len(earn_a) >= 1 \
                        and shares_t0 and shares_t0 > 0:
                    annual_ni = float(earn_a.iloc[-1])
                    eps = annual_ni / shares_t0
                    if eps > 0:
                        out["pe_ratio"] = round(price_t0 / eps, 2)
        except Exception:
            pass

    # Quarterly balance sheet → debt-to-equity, total debt
    try:
        bs = ticker.quarterly_balance_sheet
        if bs is not None and not bs.empty:
            bs = bs.copy()
            bs.columns = pd.to_datetime(bs.columns)
            bs = bs.loc[:, bs.columns <= AS_OF_DATE].sort_index(axis=1)

            def bs_val(labels):
                for L in labels:
                    if L in bs.index:
                        v = bs.loc[L].dropna()
                        if not v.empty:
                            return float(v.iloc[-1])
                return None

            total_debt = bs_val(["Total Debt", "TotalDebt", "Long Term Debt"])
            equity     = bs_val(["Stockholders Equity", "Total Stockholder Equity",
                                 "Common Stock Equity", "StockholdersEquity"])
            if total_debt is not None:
                out["total_debt"] = int(total_debt)
            if total_debt is not None and equity not in (None, 0):
                out["debt_to_equity"] = round((total_debt / equity) * 100, 2)
    except Exception:
        pass

    return out


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
    if AS_OF_DATE is not None:
        print(f"     (reconstructing fundamentals AS-OF {AS_OF_DATE.date()})")
    records = []

    for ticker_str in TICKERS:
        print(f"  Fetching fundamentals for {ticker_str}...", end=" ")
        try:
            ticker = yf.Ticker(ticker_str)

            if AS_OF_DATE is not None:
                rec = _historical_fundamentals(ticker, ticker_str, yf)
                records.append(rec)
                print("OK (historical)")
                time.sleep(0.5)
                continue

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
    if AS_OF_DATE is not None:
        print(f"     (computing metrics AS-OF {AS_OF_DATE.date()})")

    # First, get S&P 500 data for beta calculation. Pull 3y so the trailing-252
    # window still has data when AS_OF_DATE is up to ~2y in the past.
    print("  Fetching S&P 500 benchmark data...")
    sp500 = yf.Ticker("^GSPC")
    sp500_period = "3y" if AS_OF_DATE is not None else "2y"
    sp500_hist = sp500.history(period=sp500_period)

    if sp500_hist.empty:
        raise ValueError("Could not fetch S&P 500 data. Check yfinance connection.")

    sp500_close = sp500_hist["Close"].copy()
    sp500_close.index = _strip_tz(sp500_close.index)
    if AS_OF_DATE is not None:
        sp500_close = sp500_close[sp500_close.index <= AS_OF_DATE]
    sp500_returns = np.log(sp500_close / sp500_close.shift(1)).dropna()
    print(f"  S&P 500 data: {len(sp500_close)} trading days through "
          f"{sp500_close.index.max().date() if not sp500_close.empty else 'n/a'}")

    records = []

    for ticker_str in TICKERS:
        print(f"  Fetching technicals for {ticker_str}...", end=" ")
        try:
            ticker = yf.Ticker(ticker_str)
            hist_period = "3y" if AS_OF_DATE is not None else "2y"
            hist = ticker.history(period=hist_period)

            if hist.empty:
                print(f"WARN: no history")
                records.append({"ticker": ticker_str})
                continue

            close = hist["Close"].copy()
            close.index = _strip_tz(close.index)
            if AS_OF_DATE is not None:
                close = close[close.index <= AS_OF_DATE]
            if len(close) < 252:
                print(f"WARN: insufficient history ({len(close)} days)")
                records.append({"ticker": ticker_str})
                continue
            current_price = close.iloc[-1]

            # --- Returns at multiple horizons (anchored at AS_OF_DATE if set) ---
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
            # Use trailing 252 days ending at the anchor date
            trailing_log_returns = log_returns.tail(252)
            volatility = round(trailing_log_returns.std() * np.sqrt(252) * 100, 2)

            # --- Beta (vs S&P 500), trailing 252 days ending at the anchor ---
            stock_returns = log_returns.tail(252)
            common_dates = stock_returns.index.intersection(sp500_returns.index)

            if len(common_dates) >= 100:
                sr = stock_returns.loc[common_dates].values
                mr = sp500_returns.loc[common_dates].values
                # Beta = Cov(stock, market) / Var(market)
                cov_matrix = np.cov(sr, mr)
                beta = round(cov_matrix[0, 1] / cov_matrix[1, 1], 2)
            else:
                # Fallback to yfinance info beta only when no anchor date is set
                # (the .info value is "current", which would mismatch a historical pull).
                if AS_OF_DATE is None:
                    beta_info = ticker.info.get("beta", None)
                    beta = round(beta_info, 2) if beta_info is not None else None
                else:
                    beta = None

            # --- Moving Averages (computed on the AS_OF_DATE-truncated window) ---
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

def _emit_forward_prices(yf, as_of: pd.Timestamp):
    """
    Save the price at AS_OF and the price today for every ticker, so the
    realized-return backtest can compute  (price_today / price_asof - 1) * 100
    per stock and weight them by each portfolio's allocations.
    """
    today = pd.Timestamp(datetime.utcnow().date())
    rows = []
    for tk in TICKERS:
        try:
            h = yf.Ticker(tk).history(period="3y")["Close"].copy()
            h.index = _strip_tz(h.index)
            _, p0 = _close_on_or_before(h, as_of)
            _, p1 = _close_on_or_before(h, today)
            rows.append({
                "ticker":      tk,
                "asof_date":   as_of.date().isoformat(),
                "asof_price":  round(p0, 2),
                "today_date":  today.date().isoformat(),
                "today_price": round(p1, 2),
                "return_pct":  round((p1 / p0 - 1) * 100, 2) if p0 else None,
            })
        except Exception as e:
            print(f"    forward-price ERROR for {tk}: {e}")
            rows.append({"ticker": tk, "asof_date": as_of.date().isoformat()})
    out = os.path.join(OUTPUT_DIR, "forward_prices.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"     Saved forward prices -> {out}")


if __name__ == "__main__":
    import sys
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Pull real data from yfinance (otherwise demo mode).")
    ap.add_argument("--as-of", default=None,
                    help="Snapshot date YYYY-MM-DD. Implies --live. "
                         "Default 'last-year' shortcut: --as-of last-year "
                         "→ today minus 365 days.")
    args = ap.parse_args()

    live_mode = args.live or args.as_of is not None

    if args.as_of:
        if args.as_of.lower() == "last-year":
            AS_OF_DATE = pd.Timestamp(datetime.utcnow().date()) - pd.Timedelta(days=365)
        else:
            AS_OF_DATE = pd.Timestamp(args.as_of).tz_localize(None) \
                if pd.Timestamp(args.as_of).tzinfo is None \
                else pd.Timestamp(args.as_of).tz_convert(None)

    print("=" * 60)
    print("LLM PORTFOLIO PROJECT — DATA COLLECTION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {'LIVE (yfinance)' if live_mode else 'DEMO (sample data)'}")
    if AS_OF_DATE is not None:
        print(f"AS-OF: {AS_OF_DATE.date()}  (forward-prices file will be written)")
    print("=" * 60)

    if not live_mode:
        print("\n⚠ Running in DEMO mode with realistic sample data.")
        print("  To pull real data from Yahoo Finance, run:")
        print("  python pull_financial_data.py --live")
        print("  To pull a one-year-ago snapshot:")
        print("  python pull_financial_data.py --as-of last-year\n")

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

    if AS_OF_DATE is not None and live_mode:
        print("\n[1E] Emitting forward prices for realized-return backtest...")
        _emit_forward_prices(yf, AS_OF_DATE)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nFiles created in '{OUTPUT_DIR}/':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")

    print("\n✅ Ready for Phase 2: Prompt Engineering & LLM Pipeline")