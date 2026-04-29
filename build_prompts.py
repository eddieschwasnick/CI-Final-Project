"""
build_prompts.py
----------------
Rebuild the four prompt files (control / fundamental / technical / combined)
from data/csv_files/master_dataset.csv so prompts and data stay in sync.

The as-of date is auto-detected from data/csv_files/forward_prices.csv (if
present), otherwise pass --as-of YYYY-MM-DD or --as-of last-year.

Usage:
  python build_prompts.py
  python build_prompts.py --as-of 2025-04-27
  python build_prompts.py --out-dir prompts/historical    # leave originals untouched
"""

import os
import argparse
import pandas as pd
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV  = os.path.join(BASE_DIR, "data", "csv_files", "master_dataset.csv")
FORWARD_CSV = os.path.join(BASE_DIR, "data", "csv_files", "forward_prices.csv")
DEFAULT_OUT = os.path.join(BASE_DIR, "prompts")

# Display ordering and sector labels used in the table — these are intentionally
# different from master_dataset's "sector" field (AMZN listed under Tech,
# PG/KO under Consumer Staples) to match how a human analyst would group them.
DISPLAY_GROUPS = [
    ("Technology",       ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]),
    ("Healthcare",       ["JNJ", "UNH", "PFE", "LLY"]),
    ("Financials",       ["JPM", "GS", "V"]),
    ("Energy",           ["XOM", "CVX"]),
    ("Consumer Staples", ["PG", "KO"]),
    ("Industrials",      ["CAT", "HON", "BA", "MMM"]),
]
TICKER_TO_DISPLAY_SECTOR = {tk: g for g, ts in DISPLAY_GROUPS for tk in ts}
ORDERED_TICKERS = [tk for _, ts in DISPLAY_GROUPS for tk in ts]
UNIVERSE_LINE   = ", ".join(ORDERED_TICKERS)

EXAMPLE_JSON = (
    '{"AAPL": 7.5, "MSFT": 6.0, "GOOGL": 5.5, "JNJ": 4.0, "UNH": 4.5, '
    '"PFE": 3.5, "JPM": 6.0, "GS": 4.0, "XOM": 5.0, "CVX": 4.5, '
    '"PG": 4.5, "KO": 4.0, "CAT": 5.5, "HON": 4.5, "AMZN": 6.5, '
    '"NVDA": 8.0, "V": 5.0, "LLY": 6.0, "BA": 3.5, "MMM": 2.5}'
)


# ── Cell formatters ───────────────────────────────────────────────────────

def fmt_market_cap_b(val) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val / 1e9:,.0f}"


def fmt_signed_pct(val) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:+.1f}" if abs(val) < 100 else f"{val:+.2f}"


def fmt_unsigned(val, dp=2) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:,.{dp}f}"


def fmt_signal(val) -> str:
    return "N/A" if pd.isna(val) or val is None else str(val)


# ── Table builders ────────────────────────────────────────────────────────

def fundamental_table(df: pd.DataFrame) -> str:
    header = (
        "| Ticker | Sector           | Market Cap ($B) | P/E Ratio | "
        "Revenue Growth YoY (%) | Earnings Growth YoY (%) | Debt/Equity |"
    )
    sep = (
        "|--------|------------------|-----------------|-----------|"
        "------------------------|--------------------------|-------------|"
    )
    rows = [header, sep]
    for tk in ORDERED_TICKERS:
        r = df.loc[tk]
        rows.append(
            f"| {tk:<6} | {TICKER_TO_DISPLAY_SECTOR[tk]:<16} | "
            f"{fmt_market_cap_b(r['market_cap']):>15} | "
            f"{fmt_unsigned(r['pe_ratio'], 2):>9} | "
            f"{fmt_signed_pct(r['revenue_growth_yoy']):>22} | "
            f"{fmt_signed_pct(r['earnings_growth_yoy']):>24} | "
            f"{fmt_unsigned(r['debt_to_equity'], 2):>11} |"
        )
    return "\n".join(rows)


def technical_table(df: pd.DataFrame) -> str:
    header = (
        "| Ticker | Sector           | Price ($) | 1M Return (%) | "
        "3M Return (%) | 6M Return (%) | 12M Return (%) | "
        "Volatility (%, Ann.) | Beta  | MA Signal      |"
    )
    sep = (
        "|--------|------------------|-----------|---------------|"
        "---------------|---------------|----------------|"
        "----------------------|-------|----------------|"
    )
    rows = [header, sep]
    for tk in ORDERED_TICKERS:
        r = df.loc[tk]
        rows.append(
            f"| {tk:<6} | {TICKER_TO_DISPLAY_SECTOR[tk]:<16} | "
            f"{fmt_unsigned(r['current_price'], 2):>9} | "
            f"{fmt_signed_pct(r['return_1m']):>13} | "
            f"{fmt_signed_pct(r['return_3m']):>13} | "
            f"{fmt_signed_pct(r['return_6m']):>13} | "
            f"{fmt_signed_pct(r['return_12m']):>14} | "
            f"{fmt_unsigned(r['volatility'], 2):>20} | "
            f"{fmt_unsigned(r['beta'], 2):>5} | "
            f"{fmt_signal(r['ma_signal']):<14} |"
        )
    return "\n".join(rows)


# ── Prompt assembly ───────────────────────────────────────────────────────

PREFIX = (
    "You are a portfolio manager constructing an equity portfolio. "
    "You must allocate capital across the following 20 U.S. large-cap stocks "
    "and no others:\n\n"
    f"STOCK UNIVERSE (20 tickers):\n{UNIVERSE_LINE}\n"
)

INSTR_BLOCK = (
    "- Every ticker in the universe must receive a weight greater than 0%.\n"
    "- Weights must sum to exactly 100%.\n"
    "- Do not concentrate more than 20% in any single stock."
)

OUTPUT_BLOCK = (
    "BEFORE OUTPUTTING:\n"
    "Mentally sum your 20 weights. If they do not add up to exactly 100, "
    "rescale each weight proportionally (divide every weight by the total and "
    "multiply by 100) so that they do. Only output the final rescaled values.\n\n"
    "OUTPUT FORMAT:\n"
    "Respond ONLY with a single raw JSON object. No markdown, no code fences, "
    "no explanation, no commentary. The JSON must map each of the 20 tickers "
    "exactly as written above to a numeric percentage weight.\n\n"
    f"Example of the required format:\n{EXAMPLE_JSON}\n"
)

MA_KEY = (
    "MA SIGNAL KEY:\n"
    "- strong_bullish: price well above 50-day SMA, golden cross present\n"
    "- bullish: price above 50-day SMA\n"
    "- bearish: price below 50-day SMA\n"
    "- strong_bearish: price well below 50-day SMA, death cross present\n"
)


def build_control() -> str:
    return (
        f"{PREFIX}\n"
        "INSTRUCTIONS:\n"
        + INSTR_BLOCK + "\n"
        "- Construct the portfolio you believe offers the best risk-adjusted "
        "return given your knowledge of these companies.\n\n"
        + OUTPUT_BLOCK
    )


def build_fundamental(df: pd.DataFrame, date_label: str) -> str:
    return (
        f"{PREFIX}\n"
        f"FUNDAMENTAL DATA (current):\n"
        "Use the table below to inform your allocation. These are the only "
        "inputs you should use — do not rely on any other knowledge about "
        "these companies.\n\n"
        f"{fundamental_table(df)}\n\n"
        "ALLOCATION INSTRUCTIONS:\n"
        "- Use the fundamental data above to determine each stock's weight.\n"
        + INSTR_BLOCK + "\n\n"
        + OUTPUT_BLOCK
    )


def build_technical(df: pd.DataFrame, date_label: str) -> str:
    return (
        f"{PREFIX}\n"
        f"TECHNICAL DATA (current):\n"
        "Use the table below to inform your allocation. These are the only "
        "inputs you should use — do not rely on any other knowledge about "
        "these companies.\n\n"
        f"{technical_table(df)}\n\n"
        f"{MA_KEY}\n"
        "ALLOCATION INSTRUCTIONS:\n"
        "- Use the technical data above to determine each stock's weight.\n"
        + INSTR_BLOCK + "\n\n"
        + OUTPUT_BLOCK
    )


def build_combined(df: pd.DataFrame, date_label: str) -> str:
    return (
        f"{PREFIX}\n"
        f"FUNDAMENTAL DATA (current):\n\n"
        f"{fundamental_table(df)}\n\n"
        f"TECHNICAL DATA (current):\n\n"
        f"{technical_table(df)}\n\n"
        f"{MA_KEY}\n"
        "ALLOCATION INSTRUCTIONS:\n"
        "- Use both the fundamental data and the technical data above to "
        "determine each stock's weight.\n"
        + INSTR_BLOCK + "\n\n"
        + OUTPUT_BLOCK
    )


# ── Date resolution ───────────────────────────────────────────────────────

def resolve_date(cli_value: str | None) -> tuple[str, str]:
    """Return (date_label_for_prompt, iso_date)."""
    iso = None
    if cli_value:
        if cli_value.lower() == "last-year":
            iso = (pd.Timestamp(datetime.utcnow().date())
                   - pd.Timedelta(days=365)).date().isoformat()
        else:
            iso = pd.Timestamp(cli_value).date().isoformat()
    elif os.path.exists(FORWARD_CSV):
        fwd = pd.read_csv(FORWARD_CSV)
        if not fwd.empty and "asof_date" in fwd.columns:
            iso = str(fwd["asof_date"].iloc[0])

    if iso is None:
        iso = datetime.utcnow().date().isoformat()

    label = pd.Timestamp(iso).strftime("%B %-d, %Y")  # e.g. "April 27, 2025"
    return label, iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None,
                    help="Date label for the prompt headers. Default: read "
                         "asof_date from forward_prices.csv.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help=f"Where to write the four prompt files (default: {DEFAULT_OUT}).")
    args = ap.parse_args()

    if not os.path.exists(MASTER_CSV):
        raise SystemExit(f"Missing {MASTER_CSV} — run pull_financial_data.py first.")

    df = pd.read_csv(MASTER_CSV).set_index("ticker")
    missing = [t for t in ORDERED_TICKERS if t not in df.index]
    if missing:
        raise SystemExit(f"master_dataset.csv is missing tickers: {missing}")

    date_label, iso = resolve_date(args.as_of)
    print(f"Building prompts with date label: {date_label}")

    os.makedirs(args.out_dir, exist_ok=True)
    files = {
        "control_prompt.txt":     build_control(),
        "fundamental_prompt.txt": build_fundamental(df, date_label),
        "technical_prompt.txt":   build_technical(df, date_label),
        "combined_prompt.txt":    build_combined(df, date_label),
    }
    for name, body in files.items():
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            f.write(body)
        print(f"  wrote {path}  ({len(body):,} chars)")

    nan_cells = df.isna().sum().sum()
    if nan_cells:
        print(f"\nNote: {nan_cells} NaN cell(s) rendered as 'N/A' in the prompts.")
    print("\nDone.")


if __name__ == "__main__":
    main()
