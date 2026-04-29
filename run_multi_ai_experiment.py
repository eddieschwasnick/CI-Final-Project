"""
run_multi_ai_experiment.py
--------------------------
Run the same 4-arm portfolio experiment across three frontier-tier LLMs:
    • Anthropic    — claude-sonnet-4-6
    • OpenAI       — gpt-4o
    • Google       — gemini/gemini-1.5-pro

All three are comparable-tier models (rough peers of claude-sonnet-4-6).
LiteLLM dispatches by provider prefix and reads the relevant key from .env:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.

The Anthropic run is identical to what `run_experiment.py` already produced,
so by default we *re-use* the existing outputs/experiment_results.csv for
Claude rather than re-running it. To force a fresh Claude run, set
    --rerun-claude

Outputs:
  outputs/raw_responses_multi_ai/<model_label>/...    one JSON per call
  outputs/multi_ai_results.csv                         combined parsed weights
"""

import os
import csv
import json
import time
import random
import logging
import datetime
import argparse
import yaml
from dotenv import load_dotenv
import litellm
from parser import parse_portfolio, validate_batch, STOCK_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)
load_dotenv(CFG["api_key_path"])

CONDITIONS = {
    0: "control",
    1: "fundamental",
    2: "technical",
    3: "combined",
}
PROMPT_FILES = {
    0: "control_prompt.txt",
    1: "fundamental_prompt.txt",
    2: "technical_prompt.txt",
    3: "combined_prompt.txt",
}

# Comparable-tier frontier models, one per provider.
# Note: gemini-1.5-pro was retired off the v1beta endpoint in early 2025;
# gemini-2.5-pro is the current flagship and comparable to claude-sonnet-4-6.
# Gemini 2.5 uses "thinking" tokens that count against max_output_tokens, so we
# (a) bump its token cap and (b) ask it to keep reasoning short.
MODELS = [
    {"label": "claude",  "model": "claude-sonnet-4-6"},
    {"label": "openai",  "model": "gpt-4o"},
    {"label": "gemini",  "model": "gemini/gemini-2.5-pro",
     "extra_params": {"reasoning_effort": "low", "max_tokens": 8192}},
]

OUTPUTS_DIR    = CFG["outputs_dir"]
PROMPTS_DIR    = CFG["prompts_dir"]
RAW_BASE       = os.path.join(OUTPUTS_DIR, "raw_responses_multi_ai")
COMBINED_CSV   = os.path.join(OUTPUTS_DIR, "multi_ai_results.csv")
EXISTING_CSV   = os.path.join(OUTPUTS_DIR, "experiment_results.csv")


def load_prompts() -> dict[int, str]:
    prompts = {}
    for cid, fname in PROMPT_FILES.items():
        with open(os.path.join(PROMPTS_DIR, fname)) as f:
            prompts[cid] = f.read()
    return prompts


def call_llm(prompt: str, model: str, temperature: float, max_tokens: int,
             extra_params: dict | None = None) -> str | None:
    """Send a prompt to LiteLLM. Returns the response text, or None if the
    provider returned an empty / filtered response."""
    kwargs = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if extra_params:
        # Per-model overrides (e.g., Gemini reasoning_effort, larger max_tokens)
        kwargs.update(extra_params)
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content


def save_raw(label: str, run_id: str, record: dict):
    raw_dir = os.path.join(RAW_BASE, label)
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f"{run_id}.json"), "w") as f:
        json.dump(record, f, indent=2)


def run_one_model(label: str, model_id: str, runs_per_condition: int,
                  prompts: dict[int, str], temperature: float,
                  max_tokens: int, max_retries: int, sleep_sec: float,
                  extra_params: dict | None = None):
    log.info("=" * 60)
    log.info("Model: %s   (label=%s)", model_id, label)
    if extra_params:
        log.info("  extra_params: %s", extra_params)
    log.info("=" * 60)

    trial_order = [(cid, i) for cid in CONDITIONS for i in range(runs_per_condition)]
    random.shuffle(trial_order)

    results = []
    total = len(trial_order)
    for idx, (cid, trial_i) in enumerate(trial_order, start=1):
        cond_name = CONDITIONS[cid]
        run_id = (f"{label}_{cond_name}_{trial_i:03d}_"
                  f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}")
        log.info("[%s %d/%d] condition=%s trial=%d", label, idx, total, cond_name, trial_i)

        prompt    = prompts[cid]
        portfolio = None
        raw_resp  = None
        attempts  = 0

        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                raw_resp = call_llm(prompt, model_id, temperature, max_tokens,
                                    extra_params=extra_params)
            except Exception as e:
                log.warning("  Attempt %d API error: %s", attempt, e)
                time.sleep(sleep_sec)
                continue
            if raw_resp is None or not str(raw_resp).strip():
                log.warning("  Attempt %d empty/None response from provider, "
                            "retrying…", attempt)
                time.sleep(sleep_sec)
                continue
            portfolio = parse_portfolio(raw_resp)
            if portfolio is not None:
                break
            log.warning("  Attempt %d parse failed, retrying…", attempt)
            time.sleep(sleep_sec)

        record = {
            "run_id":         run_id,
            "condition_id":   cid,
            "condition_name": cond_name,
            "model":          model_id,
            "model_label":    label,
            "temperature":    temperature,
            "timestamp":      datetime.datetime.utcnow().isoformat(),
            "attempts":       attempts,
            "prompt":         prompt,
            "raw_response":   raw_resp,
            "portfolio":      portfolio,
        }
        save_raw(label, run_id, record)
        results.append(record)
        if idx < total:
            time.sleep(sleep_sec)

    summary = validate_batch(results)
    log.info("%s done — %d/%d parsed (%.1f%%)",
             label, summary["parse_success"], summary["total_runs"],
             summary["success_rate"])
    return results


def write_combined_csv(all_results: list[dict]):
    fieldnames = ["run_id", "condition_id", "condition_name", "model",
                  "model_label", "temperature", "timestamp", "attempts",
                  "parse_success"] + STOCK_UNIVERSE
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(COMBINED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "run_id":         r["run_id"],
                "condition_id":   r["condition_id"],
                "condition_name": r["condition_name"],
                "model":          r["model"],
                "model_label":    r["model_label"],
                "temperature":    r["temperature"],
                "timestamp":      r["timestamp"],
                "attempts":       r["attempts"],
                "parse_success":  r["portfolio"] is not None,
            }
            row.update(r["portfolio"] if r["portfolio"]
                       else {t: None for t in STOCK_UNIVERSE})
            writer.writerow(row)
    log.info("Wrote %s  (%d rows)", COMBINED_CSV, len(all_results))


def import_existing_claude_runs() -> list[dict]:
    """Load the existing 120-row claude-sonnet-4-6 results into the same shape."""
    import pandas as pd
    df = pd.read_csv(EXISTING_CSV)
    df = df[df["parse_success"] == True].copy()
    out = []
    for _, row in df.iterrows():
        portfolio = {t: float(row[t]) for t in STOCK_UNIVERSE}
        out.append({
            "run_id":         row["run_id"],
            "condition_id":   int(row["condition_id"]),
            "condition_name": row["condition_name"],
            "model":          row["model"],
            "model_label":    "claude",
            "temperature":    float(row["temperature"]),
            "timestamp":      row["timestamp"],
            "attempts":       int(row["attempts"]),
            "prompt":         None,
            "raw_response":   None,
            "portfolio":      portfolio,
        })
    log.info("Imported %d existing Claude runs from %s", len(out), EXISTING_CSV)
    return out


def load_runs_from_combined_csv(label_filter=None,
                                exclude_label=None) -> list[dict]:
    """
    Load rows from outputs/multi_ai_results.csv back into the runner's record
    shape. If `label_filter` is set, keep only that model_label. If
    `exclude_label` is set, drop that model_label. Used to preserve previous
    runs when re-running a single provider with --only.
    """
    import pandas as pd
    if not os.path.exists(COMBINED_CSV):
        return []
    df = pd.read_csv(COMBINED_CSV)
    if label_filter is not None:
        df = df[df["model_label"] == label_filter]
    if exclude_label is not None:
        df = df[df["model_label"] != exclude_label]
    out = []
    for _, row in df.iterrows():
        if row.get("parse_success", False):
            portfolio = {t: float(row[t]) for t in STOCK_UNIVERSE}
        else:
            portfolio = None
        out.append({
            "run_id":         row["run_id"],
            "condition_id":   int(row["condition_id"]),
            "condition_name": row["condition_name"],
            "model":          row["model"],
            "model_label":    row["model_label"],
            "temperature":    float(row["temperature"]),
            "timestamp":      row["timestamp"],
            "attempts":       int(row["attempts"]),
            "prompt":         None,
            "raw_response":   None,
            "portfolio":      portfolio,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30,
                    help="Runs per condition per model (default 30 → 120/model).")
    ap.add_argument("--rerun-claude", action="store_true",
                    help="Re-run Claude instead of importing existing results.")
    ap.add_argument("--only", choices=["claude", "openai", "gemini"], default=None,
                    help="Run just one provider.")
    args = ap.parse_args()

    random.seed(CFG["random_seed"])
    temperature = CFG["temperature"]
    max_tokens  = CFG["max_tokens"]
    max_retries = CFG["max_retries"]
    sleep_sec   = CFG["sleep_between_calls"]
    prompts     = load_prompts()

    runs = args.runs
    log.info("Multi-AI experiment: runs/condition=%d, models=%s",
             runs, [m["label"] for m in MODELS])

    all_results: list[dict] = []

    # If we're only re-running one provider, preserve the rows for the others
    # so we don't clobber Claude/OpenAI work the user already paid for.
    if args.only and os.path.exists(COMBINED_CSV):
        preserved = load_runs_from_combined_csv(exclude_label=args.only)
        if preserved:
            log.info("Preserving %d existing rows for models other than %s",
                     len(preserved), args.only)
            all_results.extend(preserved)

    for spec in MODELS:
        if args.only and spec["label"] != args.only:
            continue
        if spec["label"] == "claude" and not args.rerun_claude and os.path.exists(EXISTING_CSV):
            all_results.extend(import_existing_claude_runs())
            continue
        all_results.extend(
            run_one_model(spec["label"], spec["model"], runs, prompts,
                          temperature, max_tokens, max_retries, sleep_sec,
                          extra_params=spec.get("extra_params"))
        )

    write_combined_csv(all_results)

    # Also write a Claude-only slice to outputs/experiment_results.csv so the
    # existing analysis pipeline (compute_metrics / statistical_analysis /
    # placebo_tests / visualizations) keeps working unchanged.
    claude_only = [r for r in all_results if r["model_label"] == "claude"]
    if claude_only:
        legacy = os.path.join(OUTPUTS_DIR, "experiment_results.csv")
        legacy_fields = ["run_id", "condition_id", "condition_name", "model",
                         "temperature", "timestamp", "attempts",
                         "parse_success"] + STOCK_UNIVERSE
        with open(legacy, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=legacy_fields)
            writer.writeheader()
            for r in claude_only:
                row = {
                    "run_id":         r["run_id"],
                    "condition_id":   r["condition_id"],
                    "condition_name": r["condition_name"],
                    "model":          r["model"],
                    "temperature":    r["temperature"],
                    "timestamp":      r["timestamp"],
                    "attempts":       r["attempts"],
                    "parse_success":  r["portfolio"] is not None,
                }
                row.update(r["portfolio"] if r["portfolio"]
                           else {t: None for t in STOCK_UNIVERSE})
                writer.writerow(row)
        log.info("Wrote %s  (%d Claude rows for legacy pipeline)",
                 legacy, len(claude_only))


if __name__ == "__main__":
    main()
