# Beyond Click Sim

*Research code for evaluating LLM user simulators in recommender systems beyond click prediction.*

![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue?style=flat-square&logo=python)
![uv](https://img.shields.io/badge/uv-managed-5c3ee8?style=flat-square)
![Status](https://img.shields.io/badge/status-research%20prototype-orange?style=flat-square)

[Overview](#overview) - [Research scope](#research-scope) - [Getting started](#getting-started) - [Git workflow](#git-workflow) - [Data](#data) - [Run experiments](#run-experiments) - [Outputs](#outputs)

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

Current code primarily supports the first in-distribution interaction-prediction benchmark and a small MovieLens-1M rating-regression protocol.

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

## Git workflow

The repository is mirrored to two remotes:

| Remote | Purpose |
| --- | --- |
| `origin` | Public GitHub remote: `git@github.com:agafonovartem/beyond-click-sim.git`. |
| `corp` | Corporate GitLab remote: `git@gitlab.corp.mail.ru:ai/research-department/beyond-click-sim.git`. |

Keep both remotes up to date for shared work on `master`. The local `master`
branch may track `origin/master`, but every committed `master` update should be
pushed to both remotes:

```bash
git push origin master
git push corp master
```

Before starting work, fetch both remotes and inspect divergence:

```bash
git fetch --all --prune
git status --short --branch
git log --oneline --decorate --graph --all -n 20
```

Do not rename `origin` to the corporate remote. Do not force-push `master` unless
the branch protection and overwrite implications are explicit and intentional.
If GitLab rejects a push to protected `master`, fix the remote branch state or
branch protection in GitLab rather than repeatedly retrying `git push -f`.

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

By default, MovieLens canonicalization adds the vendored Agent4Rec movie
summaries to `items.parquet` as the optional `summary` column. Override the
source with `--movies-augmentation PATH`, or use `--without-movie-summaries`
to build the original canonical tables without this enrichment. The join is
strict: summary IDs must match the canonical MovieLens item IDs exactly.
Regression, interaction, preference, cold-start, and policy-ranking task
builders expose this canonical field as `item_summary` and copy its source
SHA256 into the task manifest. Runners that request summaries reject tasks
with missing values or missing canonical provenance.

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
  --tasks ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --methods popularity_f1_threshold
```

Run one MovieLens-1M rating-regression mode baseline:

```bash
uv run python -m runners.in_distribution.regression_prediction.run \
  --tasks ml-1m_rating_eval_users1000_rows_per_user5_seed0 \
  --methods mode_regressor
```

Run an Ollama LLM smoke test:

```bash
ollama pull llama3.1:8b

uv run python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --methods llm_yes_no_ollama_llama31_8b_smoke
```

Run a vLLM-backed Llama 3.3 70B smoke test:

```bash
uv run python -m runners.in_distribution.interaction_prediction.run \
  --tasks ml-1m_cap20_eval_users1000_cg5_m1_seed0 \
  --methods llm_yes_no_vllm_llama33_70b_smoke
```

Run the Cold Start experiment:

```bash
uv run python -m runners.in_distribution.cold_start.run \
  --methods popularity_f1_threshold,popularity_ranking,item_knn_cold_start,item_knn_cold_start_ranking,llm_yes_no_vllm_qwen36_27b_full \
  --output-dir outputs/in_distribution/cold_start
```

The default interaction tasks use the reduced `eval_users1000_cg5` protocol:
classic scorers are trained on the full train split, while validation/test
candidate construction is capped to a deterministic 1000 held-out users per
split and up to 5 candidate groups per selected user. Reduced defaults cover
seeds 0, 1, and 2. Older `eval_users1000` tasks without the per-user
candidate-group cap remain available by explicit name, and full-scale tasks
remain available by omitting the `eval_users1000` part, e.g.
`ml-1m_cap20_m1_seed0`.

Explicit item-stats task variants are also available, but are not defaults.
They add train-only `item_rating_mean` and `item_rating_count` aggregate item
features, e.g.
`ml-1m_item_stats_cap20_eval_users1000_cg5_m1_seed0` for reduced interaction
prediction and `ml-1m_rating_item_stats_eval_users1000_seed0` for rating
regression, plus
`ml-1m_rating_item_stats_eval_users1000_rows_per_user5_seed0` for the reduced
rating-regression protocol. On ML-1M the mean is computed from train ratings;
on Steam it is computed from train `playtime_forever`, despite the historical
column name. Use item-stats interaction tasks with LLM `*_with_item_stats_*`
methods when the
prompt should expose train-only item aggregates. ML-1M Agent4Rec yes/no methods
also require ML-1M item-stats interaction tasks because their movie candidate
prompt includes train-only `History ratings` (`item_rating_mean`).

Smaller debug interaction tasks with `eval_users100_cg5` are available for
prompt and scorer iteration, e.g.
`ml-1m_item_stats_cap20_eval_users100_cg5_m1_seed0`. They use the same split
and train-only feature rules as the reduced `eval_users1000_cg5` tasks, but
are not defaults and should not be treated as final reporting runs.

The current reduced interaction reporting protocol is to run seeds 0, 1, and 2
and report mean plus standard deviation over the seed-level headline test
metric. All methods compared within one seed should use the same materialized
task so model differences are paired on the same users and candidate groups.

Available interaction-prediction methods:

| Method | Description |
| --- | --- |
| `popularity_f1_threshold` | Item-popularity scorer with a validation-selected `macro_by_user_group_mean_f1` threshold. |
| `popularity_ranking` | Item-popularity scorer evaluated as raw-score candidate ranking; headline metric `test.macro_by_user_group_mean.ndcg@5` (raw scores ranked directly, no validation selection). |
| `llm_yes_no_ollama_llama31_8b_smoke` | Local Ollama Llama 3.1 8B yes/no scorer on a small candidate-group subset. |
| `llm_yes_no_ollama_llama31_8b_full` | Local Ollama Llama 3.1 8B yes/no scorer on the full selected task. |
| `llm_yes_no_vllm_llama33_70b_smoke` | vLLM Llama 3.3 70B yes/no scorer on a small candidate-group subset. |
| `llm_yes_no_vllm_llama33_70b_full` | vLLM Llama 3.3 70B yes/no scorer on the full selected task. |
| `llm_yes_no_vllm_qwen36_27b_smoke` | vLLM Qwen3.6 27B yes/no scorer on a small candidate-group subset, with Qwen thinking disabled. |
| `llm_yes_no_vllm_qwen36_27b_full` | vLLM Qwen3.6 27B yes/no scorer on the full selected task, with Qwen thinking disabled. |
| `llm_yes_no_openai_vk_gpt54_mini_smoke` | VK AI proxy `gpt-5.4-mini` yes/no scorer on a small candidate-group subset. |
| `llm_yes_no_openai_vk_gpt54_mini_full` | VK AI proxy `gpt-5.4-mini` yes/no scorer on the full selected task. |
| `llm_yes_no_openai_vk_gpt55_smoke` | VK AI proxy `gpt-5.5` yes/no scorer on a small candidate-group subset. |
| `llm_yes_no_openai_vk_gpt55_full` | VK AI proxy `gpt-5.5` yes/no scorer on the full selected task. |
| `agent4rec_yes_no_ollama_llama31_8b_smoke` | Local Ollama Llama 3.1 8B Agent4Rec-style yes/no scorer on a small candidate-group subset. |
| `agent4rec_yes_no_ollama_llama31_8b_full` | Local Ollama Llama 3.1 8B Agent4Rec-style yes/no scorer on the full selected task. |
| `agent4rec_yes_no_vllm_llama33_70b_smoke` | vLLM Llama 3.3 70B Agent4Rec-style yes/no scorer on a small candidate-group subset. |
| `agent4rec_yes_no_vllm_llama33_70b_full` | vLLM Llama 3.3 70B Agent4Rec-style yes/no scorer on the full selected task. |
| `agent4rec_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_smoke` | LiteLLM Qwen3-8B Agent4Rec-style yes/no scorer with traits, cached `gpt-4o-mini` taste, and candidate summaries on a small candidate-group subset. |
| `agent4rec_yes_no_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_full` | LiteLLM Qwen3-8B Agent4Rec-style yes/no scorer with traits, cached `gpt-4o-mini` taste, and candidate summaries on the full selected task. |
| `agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_smoke` | vLLM Qwen3.6 27B Agent4Rec-style yes/no scorer with train-derived traits and cached `gpt-4o-mini` taste profiles on a small candidate-group subset. |
| `agent4rec_yes_no_vllm_qwen36_27b_traits_taste_gpt4o_mini_full` | vLLM Qwen3.6 27B Agent4Rec-style yes/no scorer with train-derived traits and cached `gpt-4o-mini` taste profiles on the full selected task. |
| `agent4rec_yes_no_vllm_qwen36_27b_port8001_smoke` | vLLM Qwen3.6 27B Agent4Rec-style yes/no scorer using the `127.0.0.1:8001` endpoint on a small candidate-group subset. |
| `agent4rec_yes_no_vllm_qwen36_27b_port8001_full` | vLLM Qwen3.6 27B Agent4Rec-style yes/no scorer using the `127.0.0.1:8001` endpoint on the full selected task. |
| `agent4rec_yes_no_vllm_qwen36_27b_port8002_smoke` | vLLM Qwen3.6 27B Agent4Rec-style yes/no scorer using the `127.0.0.1:8002` endpoint on a small candidate-group subset. |
| `agent4rec_yes_no_vllm_qwen36_27b_port8002_full` | vLLM Qwen3.6 27B Agent4Rec-style yes/no scorer using the `127.0.0.1:8002` endpoint on the full selected task. |

The corresponding Qwen3-8B Agent4Rec `traits+taste` candidate-summary
configuration is also registered for listwise ranking as
`agent4rec_listwise_ranking_litellm_qwen3_8b_traits_taste_gpt4o_mini_candidate_summary_{smoke,full}`.

Each LLM yes/no method also has an explicit item-stats variant with `_with_item_stats`
before `_smoke` / `_full`, for example
`llm_yes_no_ollama_llama31_8b_with_item_stats_smoke`. Use these with item-stats
task variants when the prompt should expose average rating and number of prior reviews.
The two Qwen backbones have matched LiteLLM/vLLM methods for both output
contracts:
`llm_{yes_no,listwise_ranking}_litellm_qwen{3_8b,36_27b}_with_item_stats_{smoke,full}`.
Matched neutral ranker-prompt variants use
`llm_{yes_no,listwise_ranking}_openp5_style_litellm_qwen{3_8b,36_27b}_with_item_stats_{smoke,full}`.
They keep the same interaction histories, item statistics, candidate rows,
models, and output contracts while removing the simulator role-play framing.

The current Agent4Rec yes/no scorer has dataset-specific prompt/profile configs
for ML-1M and Steam. ML-1M methods build train-only traits (`activity`,
`conformity`, `diversity`) and use an Agent4Rec-style movie prompt with labeled
candidate parsing; Steam methods use game title/genre/tag candidate fields and a
playtime-oriented profile prompt. The `traits_taste_gpt4o_mini` Qwen methods add
a separate pre-scoring taste stage: they generate cached `gpt-4o-mini` taste
profiles from the same 20-item train-history window used by the history LLM
baseline. For named interaction/preference methods, taste generation uses
`make_llm_client("openai_vk_proxy")`, which loads `OPENAI_VK_PROXY_API_KEY`
from the environment or `.env`. Taste outputs are cached as append-only JSONL
under `outputs/agent4rec_taste_cache/`; cache identity depends on the taste
model and prompt version rather than the OpenAI-compatible transport. The
generic ML-1M Agent4Rec runner accepts
`summary_usage=none|profile|candidate|both`; `profile` controls only the
taste-generation history, while `candidate` controls the final movie rows.
Canonical named Agent4Rec methods use candidate summaries whenever the task
carries canonical movie-summary enrichment, matching the original Agent4Rec
item profile; profile generation remains summary-free. They fall back to
`none` for an explicit canonical opt-out and for datasets without a configured
summary source, including Steam. Treat direct comparisons against
`llm_yes_no_*` as exploratory unless temperature, token budget, visible item
fields, and history/profile representation are controlled.

For matched Steam runs, use the explicit
`agent4rec[_preference]_{yes_no,listwise_ranking}_litellm_qwen{3_8b,36_27b}_traits_taste_gpt4o_mini_no_summary_{smoke,full}`
methods. They expose game title, genres, and tags, generate playtime-oriented
taste profiles through the VK `gpt-4o-mini` client, and record
`summary_usage=none` in the manifest.

> [!NOTE]
> LLM methods use OpenAI-compatible clients. Ollama is expected at `http://localhost:11434/v1`; the default vLLM client is expected at `http://127.0.0.1:8000/v1`. VK AI proxy methods use `https://ai-proxy.vk.team/v1` and load `OPENAI_VK_PROXY_API_KEY` from the environment or `.env`. Set `BEYOND_CLICK_SIM_OPENAI_TIMEOUT_SECONDS` to override the 60-second OpenAI/VK request timeout and `BEYOND_CLICK_SIM_OPENAI_MAX_RETRIES` to override the default of zero SDK-level retries. Some Agent4Rec Qwen methods intentionally target additional local vLLM ports `8001` and `8002`.

Available preference-prediction LLM methods include Qwen3-8B and Qwen3.6-27B
`_smoke` / `_full` variants. Their ML-1M `_summary_full` variants expose
canonical movie summaries in both the history and candidate rows. The generic
preference runner also accepts `summary_visibility=none|history|candidate|both`;
movie-summary modes are not available for Steam. ML-1M Agent4Rec preference
methods use candidate-only summaries under the same canonical-enrichment rule.
Named Qwen3-8B Agent4Rec `traits+taste` methods are available for both grouped
yes/no and listwise ranking; they use cached `gpt-4o-mini` taste profiles and
candidate summaries.
Neutral ranker-prompt controls are registered as
`llm_preference_{yes_no,listwise_ranking}_openp5_style_litellm_qwen{3_8b,36_27b}_{smoke,full}`.
The matched ML-1M/Steam launcher is documented in `scripts/README.md`.

Available regression-prediction methods:

| Method | Description |
| --- | --- |
| `mode_regressor` | Constant discrete baseline that predicts the most frequent train target; default regression method and headline metric `test.macro_by_user_mean.mae`. |
| `mean_regressor` | Constant continuous baseline that predicts the train target mean; useful MAE/RMSE diagnostic but not a literal discrete rating response simulator. |
| `item_mean_regressor` | Per-item continuous baseline that predicts the train target mean for the candidate item, with global mean fallback for cold items. |
| `item_mode_regressor` | Per-item discrete baseline that predicts the train target mode for the candidate item, with global mode fallback for cold items. |
| `user_mean_regressor` | Per-user continuous baseline that predicts the mean rating from the same train-history window shown to the LLM. |
| `user_mode_regressor` | Per-user discrete baseline that predicts the most frequent rating from the same train-history window shown to the LLM. |
| `llm_regressor_ollama_llama31_8b_smoke` | Local Ollama Llama 3.1 8B integer-rating scorer on the first 25 test rows. |
| `llm_regressor_ollama_llama31_8b_full` | Local Ollama Llama 3.1 8B integer-rating scorer on the full selected test split. |
| `llm_regressor_vllm_llama33_70b_smoke` | vLLM Llama 3.3 70B integer-rating scorer on the first 25 test rows. |
| `llm_regressor_vllm_llama33_70b_full` | vLLM Llama 3.3 70B integer-rating scorer on the full selected test split. |
| `llm_regressor_openai_vk_gpt54_mini_smoke` | VK AI proxy `gpt-5.4-mini` integer-rating scorer on the first 25 test rows. |
| `llm_regressor_openai_vk_gpt54_mini_full` | VK AI proxy `gpt-5.4-mini` integer-rating scorer on the full selected test split. |
| `llm_regressor_openai_vk_gpt55_smoke` | VK AI proxy `gpt-5.5` integer-rating scorer on the first 25 test rows. |
| `llm_regressor_openai_vk_gpt55_full` | VK AI proxy `gpt-5.5` integer-rating scorer on the full selected test split. |

Each LLM regressor also has explicit `_with_item_stats_smoke` and
`_with_item_stats_full` variants, for example
`llm_regressor_vllm_llama33_70b_with_item_stats_full`. These are intended for
`*_item_stats_*` tasks; `item_mean_regressor` is the matched train-only item-mean
comparator for ordinary rating tasks.

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

outputs/in_distribution/regression_prediction/
  {timestamp}_{task}_{method}/
    manifest.json
    metrics.json
    predictions.parquet
    llm_errors.jsonl

outputs/in_distribution/preference_prediction/
  {timestamp}_{task}_{method}/
    manifest.json
    metrics.json
    metrics_ranking.json
    predictions.parquet
    llm_errors.jsonl
```

`metrics.json` is produced by pointwise methods and LLM yes/no methods. `metrics_ranking.json`
is produced by ranking methods and LLM yes/no methods. `llm_errors.jsonl` is produced by LLM
methods and may be empty; constant regression baselines do not write it.

Regression runs use observed held-out rows only: no candidate sampler, no negatives, no
`candidate_group`, and no `sampled` column. The current ML-1M rating protocol uses
`target_rating`, a 70/10/20 per-user random split, full filtered train, and a deterministic
post-split `eval_users1000_rows_per_user5` cap for validation/test by default:
sample up to 1000 users per split, then keep up to 5 observed held-out rows per
selected user with deterministic per-user random sampling. Older full-row
`eval_users1000` rating tasks remain available by explicit name. LLM rating
regression evaluates test rows only and requires a strict bare integer in
`{1, 2, 3, 4, 5}`; invalid or out-of-range responses are logged as LLM errors
rather than clamped.

For grouped pointwise interaction runs, the headline metric is
`test.macro_by_user_group_mean.f1`: compute the metric per candidate group, average groups
within each user, then average users equally. `macro_by_group` and `micro` remain diagnostic
metrics in `metrics.json`.

Ranking interaction runs use raw row-level `score` values and write `metrics_ranking.json`
instead of mixing protocols into `metrics.json`. The ranking headline is
`test.macro_by_user_group_mean.ndcg@5`; HR@1/3/5/10 and NDCG@1/3/5/10 are also logged.
LLM yes/no runs write both files because ranking can be computed from the same fixed scores.

For LLM yes/no runs, ranking is a block ranking over two hard scores: `yes = 1.0` and
`no = 0.0`. All `yes` rows are ranked above all `no` rows. Rows with the same score are
handled with `tie_policy = "average"`: the metric is the expected value under a uniform
random order inside the tied block, not the original row order.

Example candidate group:

| Row | LLM score | Ground truth |
| --- | --- | --- |
| 1 | yes | no |
| 2 | yes | yes |
| 3 | yes | no |
| 4 | no | no |
| 5 | no | yes |
| 6 | no | no |

The top tied block is the three `yes` rows and contains one true positive. Therefore
`HR@1 = 1/3`, `HR@2 = 2/3`, and `HR@3 = 1.0`. This is intentional: hard yes/no scores
can test whether positives are lifted into the `yes` block, but they do not rank items
within that block. Interpret these ranking metrics as tie-aware block-ranking metrics, not
as evidence of fine-grained candidate ordering.

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
- [`vllm_start_mbz_instruction.md`](vllm_start_mbz_instruction.md) - local vLLM startup notes for MBZUAI servers.

## Development notes

- Keep pointwise prediction, candidate ranking, static policy evaluation, and trajectory simulation separate.
- Do not tune thresholds, prompts, or hyperparameters on test data.
- Report setup details with every experiment output: dataset, split, target definition, candidate construction, seed, scorer/model config, metadata visibility, sample sizes, metrics, and caveats.
- Prefer small, explicit research code over a large simulator framework until the experiment requires one.
