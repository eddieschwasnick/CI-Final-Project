"""
run_experiment.py
-----------------
Runs the LLM portfolio allocation experiment across 4 conditions:
    0 — control      (no financial data)
    1 — fundamental  (fundamental indicators only)
    2 — technical    (technical indicators only)
    3 — combined     (fundamental + technical)

Results are saved to:
    outputs/raw_responses/   — one JSON file per run
    outputs/experiment_results.csv — parsed weights for all runs
"""

import os
import csv
import json
import time
import random
import logging
import datetime
import yaml
from dotenv import load_dotenv
import litellm
from parser import parse_portfolio, validate_batch, STOCK_UNIVERSE

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

load_dotenv(CFG["api_key_path"])

# LiteLLM picks up provider keys automatically from the environment.
# Supported env vars:  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, etc.
# If you have a LiteLLM proxy, set LITELLM_API_BASE in your .env instead.

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_prompts(prompts_dir: str) -> dict[int, str]:
    prompts = {}
    for condition_id, filename in PROMPT_FILES.items():
        path = os.path.join(prompts_dir, filename)
        with open(path) as f:
            prompts[condition_id] = f.read()
    log.info("Loaded %d prompt templates from %s", len(prompts), prompts_dir)
    return prompts


def call_llm(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Send a prompt to the LLM via LiteLLM and return the raw text response."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def save_raw(outputs_dir: str, run_id: str, record: dict):
    """Persist one run's full record as a JSON file in outputs/raw_responses/."""
    raw_dir = os.path.join(outputs_dir, "raw_responses")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def save_results_csv(outputs_dir: str, results: list[dict]):
    """Write experiment_results.csv — one row per successful run."""
    os.makedirs(outputs_dir, exist_ok=True)
    path = os.path.join(outputs_dir, "experiment_results.csv")

    fieldnames = ["run_id", "condition_id", "condition_name", "model",
                  "temperature", "timestamp", "attempts", "parse_success"] + STOCK_UNIVERSE

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
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
            if r["portfolio"]:
                row.update(r["portfolio"])
            else:
                row.update({t: None for t in STOCK_UNIVERSE})
            writer.writerow(row)

    log.info("Saved experiment_results.csv (%d rows) → %s", len(results), path)


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_experiment():
    random.seed(CFG["random_seed"])

    model       = CFG["model"]
    temperature = CFG["temperature"]
    max_tokens  = CFG["max_tokens"]
    runs        = CFG["runs_per_condition"]
    max_retries = CFG["max_retries"]
    sleep_sec   = CFG["sleep_between_calls"]
    outputs_dir = CFG["outputs_dir"]
    prompts_dir = CFG["prompts_dir"]

    prompts = load_prompts(prompts_dir)

    # Build a randomised trial order: each condition appears `runs` times,
    # shuffled so conditions are interleaved (reduces ordering effects).
    trial_order = [(cid, trial_i) for cid in CONDITIONS for trial_i in range(runs)]
    random.shuffle(trial_order)

    results = []
    total   = len(trial_order)

    log.info("Starting experiment: %d conditions × %d runs = %d total API calls",
             len(CONDITIONS), runs, total)
    log.info("Model: %s  |  Temperature: %s  |  Max retries: %d",
             model, temperature, max_retries)

    for idx, (condition_id, trial_i) in enumerate(trial_order, start=1):
        condition_name = CONDITIONS[condition_id]
        run_id = f"{condition_name}_{trial_i:03d}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"

        log.info("[%d/%d] condition=%s  trial=%d  run_id=%s",
                 idx, total, condition_name, trial_i, run_id)

        prompt    = prompts[condition_id]
        portfolio = None
        raw_resp  = None
        attempts  = 0

        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                raw_resp = call_llm(prompt, model, temperature, max_tokens)
            except Exception as e:
                log.warning("  Attempt %d — API error: %s", attempt, e)
                time.sleep(sleep_sec)
                continue

            portfolio = parse_portfolio(raw_resp)
            if portfolio is not None:
                log.info("  Attempt %d — parse OK", attempt)
                break
            else:
                log.warning("  Attempt %d — parse failed, retrying…", attempt)
                time.sleep(sleep_sec)

        record = {
            "run_id":         run_id,
            "condition_id":   condition_id,
            "condition_name": condition_name,
            "model":          model,
            "temperature":    temperature,
            "timestamp":      datetime.datetime.utcnow().isoformat(),
            "attempts":       attempts,
            "prompt":         prompt,
            "raw_response":   raw_resp,
            "portfolio":      portfolio,
        }

        save_raw(outputs_dir, run_id, record)
        results.append(record)

        # Rate-limit pause between calls
        if idx < total:
            time.sleep(sleep_sec)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = validate_batch(results)
    log.info("=" * 60)
    log.info("Experiment complete.")
    log.info("  Total runs    : %d", summary["total_runs"])
    log.info("  Parse success : %d (%.1f%%)", summary["parse_success"], summary["success_rate"])
    log.info("  Parse failures: %d", summary["parse_failures"])
    if summary["failed_run_ids"]:
        log.warning("  Failed run IDs: %s", summary["failed_run_ids"])

    save_results_csv(outputs_dir, results)
    return results


if __name__ == "__main__":
    run_experiment()
