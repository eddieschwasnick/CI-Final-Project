# CI-Final-Project

**Does the Information You Give an LLM Change the Portfolio It Builds?**
A Randomized Experiment on Information Conditioning and LLM Allocation Behavior.

**Author:** Edward Schwasnick
**Course:** STAT 6990 — Causal Inference, Spring 2026 (Prof. Zhang)

---

## Overview

This project runs a 4-arm randomized experiment on Anthropic's
`claude-sonnet-4-6`. In each trial, the model is asked to allocate capital
across 20 U.S. large-cap stocks; the only thing that varies across
conditions is the information block injected into the prompt:

| Condition | Prompt contains |
|---|---|
| `control` | ticker list only |
| `fundamental` | + fundamentals table (P/E, growth, leverage) |
| `technical` | + technicals table (returns, vol, beta, MA signals) |
| `combined` | + both tables |

Each condition is run 30 times (120 total API calls). Portfolio-level
outcomes (HHI, sector HHI, weighted beta, weighted volatility, breadth)
are then regressed on the treatment indicators and evaluated with
permutation placebos and sensitivity checks.

**Headline finding:** information framing is a large, highly significant,
and robust causal driver of LLM portfolio behavior. See
[`reports/report.tex`](reports/report.tex) for the full write-up.

---

## Repository layout

```
.
├── config.yaml                         # experiment configuration
├── prompts/                            # 4 prompt templates
│   ├── control_prompt.txt
│   ├── fundamental_prompt.txt
│   ├── technical_prompt.txt
│   └── combined_prompt.txt
├── data/
│   ├── pull_financial_data.py          # scraper for yfinance
│   └── csv_files/master_dataset.csv    # merged stock-level dataset
├── run_experiment.py                   # runs all 120 API calls
├── parser.py                           # JSON extraction + validation
├── analysis/
│   ├── compute_metrics.py              # Phase 4 — portfolio metrics
│   ├── statistical_analysis.py         # Phase 5 — ATE, t-tests, OLS
│   ├── placebo_tests.py                # placebo + sensitivity diagnostics
│   └── visualizations.py               # Phase 6 — figures
├── outputs/
│   ├── raw_responses/                  # one JSON per API call (n=120)
│   ├── experiment_results.csv          # parsed weights per trial
│   ├── portfolio_metrics.csv           # HHI, β, vol, breadth per trial
│   ├── sector_weights.csv              # per-trial sector allocation
│   ├── descriptive_stats.csv           # mean/median/std by condition
│   ├── treatment_effects.csv           # ATE + CI + p-values
│   ├── regression_results.txt          # full OLS summaries
│   ├── placebo_permutation.csv         # 5,000-iter permutation placebo
│   ├── placebo_within_control.csv      # 1,000-split null calibration
│   └── sensitivity_trimmed.csv         # trimmed-sample ATE
├── reports/
│   ├── figures/                        # 7 publication-ready PNGs
│   ├── report.tex                      # final paper (LaTeX)
│   ├── build_presentation.py           # slide generator
│   └── presentation.pptx               # 10-minute solo deck
├── LICENSE
└── README.md
```

---

## Reproducing the results

### Environment

```bash
python -m venv venv && source venv/bin/activate
pip install pandas numpy yfinance matplotlib seaborn scipy statsmodels \
            python-pptx litellm python-dotenv pyyaml
```

Put your Anthropic key in a `.env` file at repo root:

```
ANTHROPIC_API_KEY=sk-...
```

### Pipeline

```bash
# 1. Collect data (already done; the 120 raw_responses are committed).
python run_experiment.py

# 2. Compute portfolio metrics.
python analysis/compute_metrics.py

# 3. Run statistical analysis.
python analysis/statistical_analysis.py

# 4. Run placebo + sensitivity diagnostics.
python analysis/placebo_tests.py

# 5. Generate figures.
python analysis/visualizations.py

# 6. Build the presentation.
python reports/build_presentation.py

# 7. Compile the report.
cd reports && pdflatex report.tex
```

Every step after #1 is deterministic and uses cached raw API responses,
so the full analysis pipeline can be re-run offline in under a minute.

---

## Key results (summary)

| Outcome | Control | Fundamental | Technical | Combined |
|---|---|---|---|---|
| HHI | 0.060 | **0.103** | 0.076 | 0.059 |
| Sector HHI | 0.209 | **0.366** | 0.173 | 0.182 |
| Portfolio β | 0.92 | **1.05** | 0.78 | 0.85 |
| Portfolio vol (%) | 29.9 | 30.3 | 27.7 | 28.5 |

All treatment–outcome contrasts except breadth are significant at
p < 0.001. Permutation placebos, within-control placebos, and
trimmed-sample sensitivity checks all corroborate the main findings.

---

## License

MIT — see [LICENSE](LICENSE).
