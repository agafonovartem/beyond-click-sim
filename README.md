# Beyond Click Sim

*Research code for evaluating LLM user simulators in recommender systems beyond click prediction.*

![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue?style=flat-square&logo=python)
![uv](https://img.shields.io/badge/uv-managed-5c3ee8?style=flat-square)
![Status](https://img.shields.io/badge/status-research%20prototype-orange?style=flat-square)

[Overview](#overview) - [Research scope](#research-scope) - [Getting started](#getting-started) - [Data](#data) - [Run experiments](#run-experiments) - [Outputs](#outputs)

## Overview

This repository contains the clean implementation for the research project currently titled:

**Beyond Click Prediction: A Multi-Task Evaluation of LLM User Simulators for Recommendation**

The project asks whether LLM-based user simulators are useful for recommender-system evaluation, and under which evaluation setups. The core premise is that simulator quality is not a single global property: it depends on the target, split, candidate construction, negative sampling strategy, metadata visibility, model/prompt configuration, and metric.

> [!IMPORTANT]
> Treat this repository as active scientific code. Results are only meaningful together with their task manifest, candidate construction, decision rule, seed, model configuration, and known caveats.

The first implementation stage focuses on offline one-step response prediction: given user history and explicit user-item candidate rows, a scorer predicts how the user would respond. Full simulator loops are reserved for trajectory-dependent questions such as fatigue, trust degradation, filter bubbles, session exit, or same-items-different-order effects.

## Research scope

The paper scope is broader than one click-prediction benchmark:

- **Memorization tests** - check whether LLMs know recommendation datasets instead of inferring user preferences.
- **In-distribution multi-target prediction** - evaluate interaction, positive preference, rating/playtime, and other intensity targets.
- **Pointwise vs ranking evaluation** - distinguish binary outcome prediction from ranking candidates within candidate groups.
- **Policy-ranking agreement** - test whether simulated responses rank recommender policies like real held-out data.
- **Offline-constructed OOD shifts** - evaluate cold-start, temporal, semantic, feature-based, or domain-like shifts.
- **Behavioral extrapolation** - study effects such as choice overload, anchoring, social proof, repeated exposure, fatigue, novelty seeking, trust degradation, and session exit.

Current code primarily supports the first in-distribution interaction-prediction benchmark.

## What is inside

| Area | Purpose |
| --- | --- |
| `src/beyond_click_sim/data/` | Canonical dataset descriptors, manifests, and MovieLens/Steam adapters. |
| `src/beyond_click_sim/tasks/` | Filters, splitters, candidate samplers, and task builders. |
| `src/beyond_click_sim/scorers/` | Response scorers, including popularity and LLM yes/no scorers. |
| `src/beyond_click_sim/evaluation/` | Metric and decision-rule helpers. |
| `runners/` | Experiment runners that write manifests, metrics, predictions, and error logs. |
| `tests/` | Unit tests for adapters, task contracts, scorers, splitters, and metrics. |
| `notebooks/` | Exploratory inspection notebooks and reports. |

Key design distinction:

- A **scorer/simulator** predicts user response for provided candidates.
- A **recommender/policy** decides which candidates to show.

Do not mix these roles when interpreting experiments.

## Getting started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Raw recommendation datasets for experiments
- Optional: an OpenAI-compatible LLM endpoint for LLM scorers

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

## Data

Raw datasets are not included in this repository. Canonicalized data is expected under:

```text
data/canonical/{dataset}/v1/
  manifest.json
  users.parquet
  items.parquet
  interactions.parquet
```

The current runners look for `data/canonical` in this repository or one of its parent directories.

Materialize MovieLens-1M:

```bash
uv run python -m beyond_click_sim.data.canonicalize movielens \
  --raw-dir /path/to/ml-1m \
  --out-dir data/canonical/ml-1m/v1
```

Materialize Steam:

```bash
uv run python -m beyond_click_sim.data.canonicalize steam \
  --raw-dir /path/to/steam \
  --out-dir data/canonical/steam/v1
```

Supported canonical datasets:

| Dataset | Adapter | Standard targets |
| --- | --- | --- |
| MovieLens-1M | `MovieLens1MAdapter` | interaction, rating >= 4, raw rating |
| Steam | `SteamAdapter` | ownership interaction, playtime >= 120 minutes, raw playtime |

## Run experiments

Run one popularity-baseline interaction task:

```bash
uv run python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_cap20_eval_users1000_m1_seed0 \
  --methods popularity_f1_threshold
```

Run an Ollama LLM smoke test:

```bash
ollama pull llama3.1:8b

uv run python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_cap20_eval_users1000_m1_seed0 \
  --methods llm_yes_no_ollama_llama31_8b_smoke
```

Run a vLLM-backed Llama 3.3 70B smoke test:

```bash
uv run python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_cap20_eval_users1000_m1_seed0 \
  --methods llm_yes_no_vllm_llama33_70b_smoke
```

The default interaction tasks use `eval_users1000`: classic scorers are trained on
the full train split, while validation/test candidate construction is capped to a
deterministic 1000 held-out users per split. Full-scale tasks remain available by
omitting the `eval_users1000` part, e.g. `ml-1m_cap20_m1_seed0`.

Available interaction-prediction methods:

| Method | Description |
| --- | --- |
| `popularity_f1_threshold` | Item-popularity scorer with a validation-selected `macro_by_user_group_mean_f1` threshold. |
| `popularity_ranking` | Item-popularity scorer evaluated as raw-score candidate ranking; headline metric `test.macro_by_user_group_mean.ndcg@5` (raw scores ranked directly, no validation selection). |
| `llm_yes_no_ollama_llama31_8b_smoke` | Local Ollama Llama 3.1 8B yes/no scorer on a small candidate-group subset. |
| `llm_yes_no_ollama_llama31_8b_full` | Local Ollama Llama 3.1 8B yes/no scorer on the full selected task. |
| `llm_yes_no_vllm_llama33_70b_smoke` | vLLM Llama 3.3 70B yes/no scorer on a small candidate-group subset. |
| `llm_yes_no_vllm_llama33_70b_full` | vLLM Llama 3.3 70B yes/no scorer on the full selected task. |

> [!NOTE]
> LLM methods use OpenAI-compatible clients. Ollama is expected at `http://localhost:11434/v1`; vLLM is expected at `http://127.0.0.1:8000/v1`.

## Outputs

Experiment runs are written under:

```text
outputs/in_distribution/interaction_prediction/
  {timestamp}_{task}_{method}/
    manifest.json
    metrics.json
    metrics_ranking.json
    predictions.parquet
    llm_errors.jsonl
```

`metrics.json` is produced by pointwise methods and LLM yes/no methods. `metrics_ranking.json`
is produced by ranking methods and LLM yes/no methods. `llm_errors.jsonl` is produced by LLM
methods and may be empty.

For grouped pointwise interaction runs, the headline metric is
`test.macro_by_user_group_mean.f1`: compute the metric per candidate group, average groups
within each user, then average users equally. `macro_by_group` and `micro` remain diagnostic
metrics in `metrics.json`.

Ranking interaction runs use raw row-level `score` values and write `metrics_ranking.json`
instead of mixing protocols into `metrics.json`. The ranking headline is
`test.macro_by_user_group_mean.ndcg@5`; HR@1/3/5/10 and NDCG@1/3/5/10 are also logged.
LLM yes/no runs write both files because ranking can be computed from the same fixed scores.

Old fixed-prediction LLM runs can be migrated to the current metric schema without new LLM
calls:

```bash
uv run python runners/in_distribution/interaction_prediction/evaluate_llm_predictions.py RUN_DIR
```

Score-based pointwise methods with validation-selected thresholds should be rerun instead of
migrated, because threshold selection is part of their method protocol. Raw-score ranking
methods should be run under an explicit ranking method such as `popularity_ranking`.

Before trusting a result, inspect:

- `manifest.json` for dataset, split, target, sampler, scorer, prompt/model config, and git commit;
- `metrics.json` for pointwise validation/test metrics, user-level headline metrics, and candidate-group diagnostics;
- `metrics_ranking.json` for raw-score ranking metrics when present;
- `predictions.parquet` for row-level scores and, for pointwise runs, thresholded or fixed predictions;
- `issues.md` for known current validity issues.

## Documentation

- [`AGENTS.md`](AGENTS.md) - research framing and working principles for coding agents.
- [`CLAUDE.md`](CLAUDE.md) - critical code-review instructions for Claude-style agent reviews.
- [`architecture_note.md`](architecture_note.md) - offline-first architecture and extension plan.
- [`in_distribution_scenarios.md`](in_distribution_scenarios.md) - task formulation notes for in-distribution evaluation.
- [`issues.md`](issues.md) - known defects in the current experimental pipeline.
- [`notes.md`](notes.md) - active research and implementation notes.

## Development notes

- Keep pointwise prediction, candidate ranking, static policy evaluation, and trajectory simulation separate.
- Do not tune thresholds, prompts, or hyperparameters on test data.
- Report setup details with every experiment output: dataset, split, target definition, candidate construction, seed, scorer/model config, metadata visibility, sample sizes, metrics, and caveats.
- Prefer small, explicit research code over a large simulator framework until the experiment requires one.
