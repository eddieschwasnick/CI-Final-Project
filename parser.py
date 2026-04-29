import json
import re
import logging

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "JNJ", "UNH", "PFE", "JPM", "GS",
    "XOM", "CVX", "PG", "KO", "CAT", "HON", "AMZN", "NVDA", "V",
    "LLY", "BA", "MMM"
]

WEIGHT_TOLERANCE = 15.0  # weights must sum to 100 ± this value before normalization


def parse_portfolio(raw_response: str | None) -> dict | None:
    """
    Extract and validate a portfolio JSON object from a raw LLM response string.

    Returns a normalized dict {ticker: weight} where weights sum to exactly 100.0,
    or None if parsing fails.
    """
    if raw_response is None or not str(raw_response).strip():
        logging.warning("Empty/None response from model.")
        return None

    # Step 1: extract JSON from the response
    portfolio = _extract_json(raw_response)
    if portfolio is None:
        logging.warning("JSON extraction failed. Raw response:\n%s", raw_response)
        return None

    # Step 2: validate tickers
    missing = [t for t in STOCK_UNIVERSE if t not in portfolio]
    extra   = [t for t in portfolio if t not in STOCK_UNIVERSE]
    if missing:
        logging.warning("Portfolio is missing tickers: %s", missing)
        return None
    if extra:
        logging.warning("Portfolio contains unexpected tickers: %s", extra)
        return None

    # Step 3: validate all weights are numeric and non-negative
    for ticker, weight in portfolio.items():
        if not isinstance(weight, (int, float)):
            logging.warning("Non-numeric weight for %s: %r", ticker, weight)
            return None
        if weight < 0:
            logging.warning("Negative weight for %s: %s", ticker, weight)
            return None

    # Step 4: check weight sum is within tolerance
    total = sum(portfolio.values())
    if abs(total - 100.0) > WEIGHT_TOLERANCE:
        logging.warning("Weights sum to %.4f, outside tolerance of ±%.1f", total, WEIGHT_TOLERANCE)
        return None

    # Step 5: normalize to exactly 100.0
    portfolio = {ticker: round((w / total) * 100, 6) for ticker, w in portfolio.items()}

    return portfolio


def _extract_json(text: str) -> dict | None:
    """
    Try multiple strategies to pull a JSON object out of the LLM response.
    Returns a parsed dict or None.
    """
    # Strategy 1: the whole response is already valid JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: grab the first {...} block in the response
    braces = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if braces:
        try:
            return json.loads(braces.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_batch(results: list[dict]) -> dict:
    """
    Given a list of parse result dicts (each with keys: run_id, condition,
    raw_response, portfolio), return a summary of parse success rate.
    """
    total   = len(results)
    success = sum(1 for r in results if r.get("portfolio") is not None)
    failed  = [r["run_id"] for r in results if r.get("portfolio") is None]

    summary = {
        "total_runs":     total,
        "parse_success":  success,
        "parse_failures": total - success,
        "success_rate":   round(success / total * 100, 2) if total else 0,
        "failed_run_ids": failed,
    }
    return summary


# ---------------------------------------------------------------------------
# Quick smoke-test — run with: python parser.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    test_cases = [
        # (label, raw_response)
        (
            "clean JSON",
            '{"AAPL": 7.5, "MSFT": 6.0, "GOOGL": 5.5, "JNJ": 4.0, "UNH": 4.5, '
            '"PFE": 3.5, "JPM": 6.0, "GS": 4.0, "XOM": 5.0, "CVX": 4.5, '
            '"PG": 4.5, "KO": 4.0, "CAT": 5.5, "HON": 4.5, "AMZN": 6.5, '
            '"NVDA": 8.0, "V": 5.0, "LLY": 6.0, "BA": 3.5, "MMM": 2.5}',
        ),
        (
            "markdown code fence",
            '```json\n{"AAPL": 7.5, "MSFT": 6.0, "GOOGL": 5.5, "JNJ": 4.0, "UNH": 4.5, '
            '"PFE": 3.5, "JPM": 6.0, "GS": 4.0, "XOM": 5.0, "CVX": 4.5, '
            '"PG": 4.5, "KO": 4.0, "CAT": 5.5, "HON": 4.5, "AMZN": 6.5, '
            '"NVDA": 8.0, "V": 5.0, "LLY": 6.0, "BA": 3.5, "MMM": 2.5}\n```',
        ),
        (
            "JSON embedded in prose",
            'Here is my portfolio allocation:\n'
            '{"AAPL": 7.5, "MSFT": 6.0, "GOOGL": 5.5, "JNJ": 4.0, "UNH": 4.5, '
            '"PFE": 3.5, "JPM": 6.0, "GS": 4.0, "XOM": 5.0, "CVX": 4.5, '
            '"PG": 4.5, "KO": 4.0, "CAT": 5.5, "HON": 4.5, "AMZN": 6.5, '
            '"NVDA": 8.0, "V": 5.0, "LLY": 6.0, "BA": 3.5, "MMM": 2.5}\n'
            'These weights sum to 100%.',
        ),
        (
            "weights sum to 99.9 (within tolerance)",
            '{"AAPL": 7.4, "MSFT": 6.0, "GOOGL": 5.5, "JNJ": 4.0, "UNH": 4.5, '
            '"PFE": 3.5, "JPM": 6.0, "GS": 4.0, "XOM": 5.0, "CVX": 4.5, '
            '"PG": 4.5, "KO": 4.0, "CAT": 5.5, "HON": 4.5, "AMZN": 6.5, '
            '"NVDA": 8.0, "V": 5.0, "LLY": 6.0, "BA": 3.5, "MMM": 2.5}',
        ),
        (
            "FAIL: missing tickers",
            '{"AAPL": 50.0, "MSFT": 50.0}',
        ),
        (
            "FAIL: weights too far off 100",
            '{"AAPL": 7.5, "MSFT": 6.0, "GOOGL": 5.5, "JNJ": 4.0, "UNH": 4.5, '
            '"PFE": 3.5, "JPM": 6.0, "GS": 4.0, "XOM": 5.0, "CVX": 4.5, '
            '"PG": 4.5, "KO": 4.0, "CAT": 5.5, "HON": 4.5, "AMZN": 6.5, '
            '"NVDA": 8.0, "V": 5.0, "LLY": 6.0, "BA": 3.5, "MMM": 99.0}',
        ),
        (
            "FAIL: pure prose, no JSON",
            "I would allocate most of the portfolio to Apple and Microsoft.",
        ),
    ]

    print(f"{'Test Case':<40} {'Result'}")
    print("-" * 60)
    for label, raw in test_cases:
        result = parse_portfolio(raw)
        if result is not None:
            total = round(sum(result.values()), 4)
            status = f"PASS  (sum={total})"
        else:
            status = "FAIL  (returned None)"
        print(f"{label:<40} {status}")


