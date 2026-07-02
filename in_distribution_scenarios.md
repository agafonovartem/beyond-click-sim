### TaskBuilder for In-Distribution Tasks

Preprocessing after canonization has the following task-construction stages:
1. **Filtering.** I can filter any dataset by any rules I like. It should be simply `DatasetFilter` class with method `filter`. That's it. It does not care about target. For `DatasetFilter` it is just filtering condition.
2. **Splitters:** A splitter creates train/validation/test interaction splits. Examples:
    - RandomFractionSplitter: train/val/test by fractions, e.g. 70/10/20
    - StratifiedRandomSplitter: random split preserving label proportions.
    - GlobalTemporalSplitter: shared time cutoffs across users.
    - leave-n-out variants may be added later.
3. **CandidateSampler.**  It constructs explicit candidate rows when the task needs them. This is independent of whether we later compute pointwise metrics or ranking metrics. The same candidate table can be flattened for binary classification metrics or grouped by candidate group for ranking metrics.
4. **Optional item feature enrichment.** Split-dependent item metadata, such as train-only item rating mean/count, must be fit on the train interactions only. In the current builders it is applied after split construction and candidate/eval row sampling, then joined through the item feature table before the final task dataframes are returned. These features are not canonical dataset metadata, so their source and scorer visibility should be recorded in the task/run manifest.
While canonicalization creates standard target columns, candidate samplers may add rows that are not canonical interactions such as sampled non-interactions; in that case the sampler is responsible for assigning `target` and documenting the assumption.

Task families:

1. **Interaction task.**
    Question: will the user interact with this item?
    - Filter: usually ensure each user has enough observed interactions before splitting.
    - Split: random or temporal split over observed interactions.
    - Sampler: required for explicit negative examples. We sample unobserved items to provide negative candidates for positive observed interactions. For sampled non-interactions, `target=0` is part of the interaction-task assumption.

2. **Preference task.**
    Question: will the user like this interacted item enough?
    - Filter: usually ensure each user has enough observed positives and observed negatives under the selected preference target.
    - Split: random, stratified random, or temporal split over observed interactions.
    - Sampler: optional for observed-only pointwise classification, but required for candidate-set evaluation. When sampling candidates, we should usually sample observed unliked items to provide negative candidates for positive liked items. This keeps preference labels grounded in observed feedback.

3. **Regression task.**
    Question: what numeric intensity will the user produce?
    - Filter: usually ensure enough observed interactions with valid numeric target values.
    - Split: random or temporal split over observed interactions.
    - Sampler: no negative sampling is needed. For standard RMSE/MAE, evaluate on observed held-out rows with known numeric labels. In the SimUSER paper, rating prediction uses an 80/10/10 time-based split without stratification or fixed-size candidate sets.  

### Problem Formulation for Multi-Target In-Distribution User Outcome Prediction
When we simulate users in recommender systems, or build recommender systems in general, we have many datasets and can define many tasks on them:
1. **Interaction:** will the user interact with this item?
2. **Positive preference:** will the user like it enough?
3. **Intensity regression:** how much will the user like or consume this item?

Note, that all of these problems are user-level.

The first two targets can be evaluated from two perspectives: pointwise outcome prediction, similar to alignment-style tables in Agent4Rec and SimUSER, and candidate ranking, similar to AgentRecBench-style recommendation evaluation.

From the model perspective, both settings reduce to scoring. The model assigns a numeric score to each user-item row. Then we can treat them in different ways according to evaluation protocol.

- Pointwise / Alignment / Classification (Tasks 1 and 2):
    - input: test set (candidates list)
    - model: calculates scores. They could be different: LLM just says yes/no, while classical ML methods may give probabilities, ratings, logits.
    - evaluation: convert scores into binary predictions (per candidate group if needed) using a decision rule, such as a validation-selected threshold, then compute accuracy, precision, recall, F1, AUC / PR-AUC, ...

    This setting asks whether the model can distinguish positive and negative user outcomes.

- Ranking / Recommendation (Tasks 1 and 2)
    - input: candidates list
    - model: assigns one score to each candidate row
    - evaluation: sort candidates by score within each candidate group, then compute HR@K, NDCG@K, MRR, and related ranking metrics.

    In this setting, the model does not directly output ranks. It outputs relevance scores; ranks are derived by the evaluation protocol.

- Regression / Intensity Prediction (Task 3)
    - input: test set
    - model: calculates scores
    - evaluation: computes mse, mae

This leads to the core Stage-1 abstraction: every user simulator or response model is a `Scorer`. Conceptually, a scorer estimates `score_target(user, item, ...)`.

The score is not necessarily a calibrated probability. For binary tasks, it is a positive-class score. For ranking tasks, it is a relevance or utility score, and ranks are obtained by sorting scores inside candidate groups. For regression tasks, the score is the predicted numeric value. Therefore the minimal `Scorer` interface should have two methods: `fit()` and `score()`.

### Pointwise Decision Protocol
For pointwise alignment/classification, scorers may output arbitrary real-valued scores, so binary metrics require an explicit decision rule.

For score-based methods (Popularity, ItemKNN, MF/BPR, tabular ML), choose the threshold on validation and keep it fixed on test. Do not tune thresholds on test.

For direct LLM yes/no protocols, no threshold is needed: the prompt/parser already defines binary predictions. Prompt and parser choices should still be fixed before test evaluation.

For pointwise threshold selection, validation should use the same candidate construction as test.

For candidate-grouped pointwise tasks, aggregate the headline metric as
`macro_by_user_group_mean`: compute the metric inside each candidate group, average groups
within the same user, then average users equally. Keep `macro_by_group` and `micro` as
diagnostics, not as the headline. Threshold selection for score-based methods should use
the same user-group aggregation as the reported test metric.

Threshold and hyperparameter selection belongs to the experiment loop, not to metric functions.

Ranking is a separate protocol: sorting candidates within `candidate_group` and reporting HR@K/NDCG@K/MRR should not be silently mixed with pointwise alignment metrics.

---

## Task Family 4: Policy Ranking Agreement (Q3)

### Scientific Question

Do LLM-based user simulators rank recommender policies in the same order as real held-out user data?

A user simulator is useful for offline recommender evaluation only if the policy ordering it produces agrees with the ordering a real A/B test would produce. This task family measures that agreement directly: each policy is scored under both simulated responses and real binary interaction targets, yielding a utility scalar per policy. Policy ranking agreement is then computed between the simulated ranking and the real ranking using Kendall's tau (main metric) and Spearman's rho.

### High-Level Pipeline

```
train interactions  →  fit each Policy  →  recommend top-K items per user
                                                  ↓
test interactions   →  attach binary target (hit/miss) per recommended item
                                                  ↓
                        Score every (user, policy, item) row with a Scorer/Simulator
                                                  ↓
                        Simulated utility per policy = mean simulated score
                        Real utility per policy     = mean hit rate on held-out data
                                                  ↓
                        Kendall's τ / Spearman's ρ between simulated and real rankings
```

No data from the test split is visible to the policies during fitting. The scorer receives the recommendation rows with user history context drawn from the train split only.

### Recommender Policies

Six policies are compared, spanning the main families of collaborative filtering:

| Policy | Type | Key hyperparams |
|---|---|---|
| `RandomPolicy` | Random baseline | `k`, `seed` |
| `PopularityPolicy` | Non-personalized popularity | `k`, `seed` |
| `ItemKNNPolicy` | Memory-based CF (item-item cosine similarity) | `k`, `n_neighbors=20`, `seed` |
| `ALSPolicy` | Matrix factorization via ALS | `k`, `n_factors=64`, `iterations=20`, `seed` |
| `BPRPolicy` | MF via Bayesian Personalized Ranking | `k`, `n_factors=64`, `learning_rate=0.01`, `regularization=0.01`, `iterations=100`, `seed` |
| `LightGCNPolicy` | Graph CF (LightGCN, He et al. 2020) | `k`, `n_factors=64`, `n_layers=3`, `learning_rate=0.001`, `regularization=1e-4`, `iterations=200`, `seed` |

All policy classes implement the `Policy` ABC:
- `fit(train_interactions, *, items, ...)` → `Self` — trains the model on the train split
- `recommend(users, *, train_interactions, items, ...)` → DataFrame with columns `[user_id, item_id, policy, rank]`

Policies exclude items the user already interacted with in the train split from recommendations.

**Implementation:** `src/beyond_click_sim/tasks/policies.py`

### Dataset and Split Setup

**Datasets:** MovieLens-1M (`ml-1m`) and Steam (`steam`)

**Filtering:** `MinUserInteractionsFilter(min_interactions=10)` — retain only users with at least 10 recorded interactions.

**Split:** `RandomFractionSplitter(train_fraction=0.8, val_fraction=0.0, test_fraction=0.2, seed=seed)` — random 80/20 train/test split, no validation split (val is always empty for Q3).

**Eval users:** Two task variants are built:
- `eval_users1000` — evaluate on a random subsample of up to 1 000 users (default for runs); faster iteration.
- `full` — evaluate on all users passing the filter.

**Recommendation list size:** `POLICY_K = 10` — each policy recommends exactly 10 items per user.

**Task name format:**
- Default (eval1000): `{dataset}_policy_ranking_eval_users1000_seed{seed}`
- Full: `{dataset}_policy_ranking_seed{seed}`
- Seeds: 0, 1, 2, 3, 4

### Task Construction

`PolicyRankingTaskBuilder` (`src/beyond_click_sim/tasks/policy_ranking.py`) assembles the task:

1. Applies the dataset filter and splitter to produce train/test interaction splits.
2. Optionally subsamples eval users with `PostSplitUserSampler`.
3. Fits each policy on the **train** split.
4. Calls `policy.recommend(eval_users, ...)` for each policy, producing recommendation rows.
5. Attaches a binary `target` column: `1` if the recommended item appears in the user's **test** interactions, `0` otherwise.
6. Concatenates all policy recommendation rows into a single DataFrame. Each row carries: `user_id`, `item_id`, `policy`, `rank`, `target`, plus user history columns for LLM prompt construction.
7. Sets `task.train` = train interaction rows (used by the scorer's `.fit()`), `task.test` = recommendation rows with targets, `task.val` = empty.

No test-set signal leaks into policy fitting: policies see only the train split. The `target` column is derived from the test split after all policies have generated recommendations.

**Task builder module:** `runners/in_distribution/policy_ranking_agreement/task_builders.py`

### Scoring Protocol

The scorer (`LLMInteractionYesNoScorer`) operates on the recommendation rows in `task.test`. Two scoring modes are supported:

**Batch scoring** (`scoring="batch"`)  
The scorer receives all `k` recommended items for a given `(user, policy)` pair in a single prompt. One LLM call per `(user, policy)` group. This is the default and is cheaper.

**Itemwise scoring** (`scoring="itemwise"`)  
The scorer receives one item per LLM call, eliminating any positional or list-context bias. One LLM call per `(user, policy, item)` triplet. Cross-policy deduplication: if the same `(user, item)` pair appears under multiple policies, only one LLM call is made and the score is broadcast to all policy rows for that pair.

Both modes output a binary score (1.0 = "yes the user would interact", 0.0 = "no").

**Utility aggregation:**
- Simulated utility for policy `p` = mean simulated score over all `(user, item)` rows belonging to `p`
- Real utility for policy `p` = mean binary hit rate (target) over the same rows

**Agreement metrics** (`src/beyond_click_sim/evaluation/policy_ranking.py`):
- Kendall's τ (main metric: `test.kendall_tau`)
- Spearman's ρ (`test.spearman_rho`)

Both are computed between the vector of simulated utilities and the vector of real utilities, one scalar per policy.

### LLM Prompt Context

The scorer constructs prompts using the user's train interactions as history. Dataset-specific columns:

| Dataset | History columns | Candidate columns |
|---|---|---|
| `ml-1m` | `item_title`, `item_genres`, `rating` | `item_title`, `item_genres` |
| `steam` | `item_title`, `item_genres_json`, `item_tags_json`, `playtime_forever` | `item_title`, `item_genres_json`, `item_tags_json` |

History is capped at `MAX_HISTORY_ITEMS = 20` items. LLM inference uses `temperature=0.0`, `max_tokens=256`.

### Output Artifacts

Each run writes to a timestamped output directory under `outputs/in_distribution/policy_ranking_agreement/`:

| File | Contents |
|---|---|
| `manifest.json` | Full run provenance: method, scorer config, prompt columns, dedup strategy, git commit |
| `metrics.json` | Agreement metrics: `test.kendall_tau`, `test.spearman_rho`, per-policy utilities, timing |
| `predictions.parquet` | Row-level scores and targets for every recommendation row |
| `llm_errors.jsonl` | LLM call failures (empty if all calls succeeded) |

### Runner

**Entry point:** `runners/in_distribution/policy_ranking_agreement/run.py`

```
python -m runners.in_distribution.policy_ranking_agreement.run \
    [--tasks TASK1,TASK2,...] \
    [--methods METHOD1,METHOD2,...] \
    [--output-dir PATH]
```

- `--tasks`: comma-separated task names from `TASK_BUILDERS`. Default: all 10 eval1000 tasks (both datasets × 5 seeds).
- `--methods`: comma-separated method names from `METHOD_RUNNERS`. Default: `popularity_scorer`.
- `--output-dir`: root directory for output artifacts. Default: `outputs/in_distribution/policy_ranking_agreement/`.

**Available method names** (from `runners/in_distribution/policy_ranking_agreement/methods/__init__.py`):

| Method name | Backend | Model | Scoring |
|---|---|---|---|
| `popularity_scorer` | — | Popularity heuristic | — |
| `llm_yes_no_ollama_llama31_8b_smoke` | Ollama | LLaMA 3.1 8B | batch, 25 groups |
| `llm_yes_no_ollama_llama31_8b_full` | Ollama | LLaMA 3.1 8B | batch, all groups |
| `llm_yes_no_ollama_llama32_smoke` | Ollama | LLaMA 3.2 | batch, 25 groups |
| `llm_yes_no_ollama_llama32_full` | Ollama | LLaMA 3.2 | batch, all groups |
| `llm_yes_no_vllm_llama33_70b_smoke` | vLLM | LLaMA 3.3 70B | batch, 25 groups |
| `llm_yes_no_vllm_llama33_70b_full` | vLLM | LLaMA 3.3 70B | batch, all groups |
| `llm_yes_no_vllm_qwen3_8b_full` | vLLM | Qwen3-8B | batch, all groups |
| `llm_yes_no_vllm_qwen36_27b_full` | vLLM | Qwen3.6-27B | batch, all groups |
| `llm_yes_no_vllm_qwen36_35b_a3b_full` | vLLM | Qwen3.6-35B-A3B | batch, all groups |
| `llm_yes_no_itemwise_vllm_llama33_70b_full` | vLLM | LLaMA 3.3 70B | itemwise, all |
| *(+ itemwise variants for all models above)* | | | |

Smoke variants cap evaluation at 25 LLM groups per run (proportionally distributed across policies) for fast sanity checks.

### CLI Example: Run Q3 on MovieLens with LLaMA 3.3 70B (vLLM, full)

Run all 5 seeds for `ml-1m` with all 6 policies using the LLaMA 3.3 70B scorer in batch mode:

```bash
cd beyond-click-sim

python -m runners.in_distribution.policy_ranking_agreement.run \
    --tasks ml-1m_policy_ranking_eval_users1000_seed0,ml-1m_policy_ranking_eval_users1000_seed1,ml-1m_policy_ranking_eval_users1000_seed2,ml-1m_policy_ranking_eval_users1000_seed3,ml-1m_policy_ranking_eval_users1000_seed4 \
    --methods llm_yes_no_vllm_llama33_70b_full \
    --output-dir outputs/in_distribution/policy_ranking_agreement
```

For a quick smoke test on seed 0 only (caps at 25 LLM groups per policy):

```bash
python -m runners.in_distribution.policy_ranking_agreement.run \
    --tasks ml-1m_policy_ranking_eval_users1000_seed0 \
    --methods llm_yes_no_vllm_llama33_70b_smoke \
    --output-dir outputs/in_distribution/policy_ranking_agreement
```

The runner builds each task once (fitting all 6 policies on the train split), then runs all requested methods against it. Policy fitting happens inside `build_policy_ranking_task()` — LightGCN and BPR are the most expensive, taking 1–10 minutes per task on MovieLens-1M depending on hardware.

### Key Design Decisions

**No validation split.** Q3 requires no hyperparameter tuning at evaluation time — policies are fixed, and the scorer uses `temperature=0.0`. The val split is always empty.

**L2 regularization on initial embeddings only (LightGCN).** Per LightGCN paper Section 3.3, regularization is applied to `E0` (the initial embedding matrix), not to the propagated `E_final`. This is critical for correct gradient computation and is implemented accordingly in `LightGCNPolicy`.

**Cross-policy deduplication (itemwise mode).** When the same item appears in multiple policies' recommendation lists for the same user, the LLM is called only once per `(user, item)` pair and the score is broadcast. This is correct because the itemwise scorer is context-independent of which policy recommended the item.

**Proportional smoke capping.** In smoke mode, the group cap (`max_llm_groups=25`) is distributed proportionally across policies so every policy stays represented in the result. Without this, a smoke run could accidentally exclude some policies entirely.

---

## Cold Start

TODO: Verify later after exp runs

### Scientific Question

Can an LLM given only a short user profile — items the user interacted with before any training data was collected — predict per-candidate relevance competitively with classical low-data baselines (Popularity, ItemKNN)?

This experiment isolates **Normal User Cold-Start**: users are entirely absent from training. Every candidate scored is for a user the model has never seen. Items remain warm (all items in cold users' profiles and evaluation sets exist in the training item catalog).

### Split Design

Users are partitioned into three non-overlapping groups; none of a cold user's interactions appear in `train`:

| Group | Fraction | Role |
|---|---|---|
| Warm users | 0.7 | All interactions → `train` |
| Val-cold users | 0.1 | Held out; evaluated on val |
| Test-cold users | 0.2 | Held out; evaluated on test |

Within each cold user, interactions are sorted by `timestamp` ascending (ties broken by `item_id`). The first k form `online_session_history` — the scorer's only view of the user. Post-k interactions become the evaluation target. Cold users with ≤ k interactions in total are dropped from evaluation.

The k-axis produces a curve rather than a single point: running for `k ∈ {1, 3, 5}` shows how each additional profile item affects cold-user relevance prediction.

### Dataset and Split Setup

**Datasets:** MovieLens-1M (`ml-1m`) only. Steam lacks wall-clock timestamps and is unsupported.

**Filtering:** `MinUserInteractionsFilter(min_interactions=10)`

**Split:** `ColdUserHoldoutSplitter(k=k, train_fraction=0.7, val_fraction=0.1, test_fraction=0.2, seed=seed)`

**Candidate sampling:** `CappedUserInteractionCandidateSampler(negative_ratio=negative_ratio, total_items=20, seed=seed)`. Exclusion set covers the full cold user history (profile + val + test positives), so no observed item is drawn as a negative.

**Task name format:** `{dataset}_cold_start_k{k}_cap20_m{negative_ratio}_seed{seed}`

Total tasks: 3 k-values × 5 negative ratios × 5 seeds = **75 tasks** (ml-1m only).

`DEFAULT_TASK_NAMES`: `k ∈ {1, 3, 5}`, `negative_ratio=1`, `seed ∈ {0, 1, 2}` — 9 tasks.

**Task builder module:** `runners/in_distribution/cold_start/task_builders.py`

### Scorer Protocol

Each scorer receives a different frame for fitting:

| Scorer | Fit data | Why |
|---|---|---|
| `PopularityScorer` | `task.train` | Item counts from warm users only |
| `LLMInteractionYesNoScorer` | `task.online_session_history` | Cold user's k-item visible profile; `task.train` has no cold-user rows |
| `ColdItemKNNScorer` | `fit_train(task.train)` then `fit(task.online_session_history)` | Sim matrix from warm co-occurrence; aggregation profiles from cold user history |

`task.val` is used for threshold calibration by Popularity and ItemKNN (threshold is fixed on val, not re-tuned on test). `task.test` is the evaluation target for all scorers.

### Output Artifacts

Each run writes to a timestamped directory under `outputs/in_distribution/cold_start/`:

| File | Produced by | Contents |
|---|---|---|
| `manifest.json` | All | Method, scorer config (`fit_on`, `k`, `n_neighbors` for ItemKNN; prompt columns for LLM), decision rule, task manifest, git commit |
| `metrics.json` | Pointwise methods | `val` + `test` metrics: `macro_by_user_group_mean` (headline), `macro_by_group`, `micro`; threshold |
| `metrics_ranking.json` | Ranking methods | `val` + `test` ranking: `NDCG@{1,3,5,10}`, `HR@{1,3,5,10}` |
| `predictions.parquet` | All | Row-level scores and predictions for val and test |
| `llm_errors.jsonl` | LLM methods only | Failed LLM call records (candidate group, attempt count, errors) |

### Runner

**Entry point:** `runners/in_distribution/cold_start/run.py`

```bash
python -m runners.in_distribution.cold_start.run \
    [--tasks TASK1,TASK2,...] \
    [--methods METHOD1,METHOD2,...] \
    [--output-dir PATH]
```

Default: 9 `DEFAULT_TASK_NAMES`, method `popularity_f1_threshold`, output `outputs/in_distribution/cold_start/`.

**Available methods:**

| Method name | Scorer | Description |
|---|---|---|
| `popularity_f1_threshold` | `PopularityScorer` | Val-threshold pointwise; F1 headline |
| `popularity_ranking` | `PopularityScorer` | Raw-score ranking; NDCG headline |
| `item_knn_cold_start` | `ColdItemKNNScorer` | Val-threshold pointwise; F1 headline |
| `item_knn_cold_start_smoke` | `ColdItemKNNScorer` | Same, capped at 25 candidate groups |
| `item_knn_cold_start_ranking` | `ColdItemKNNScorer` | Raw-score ranking; NDCG headline |
| `item_knn_cold_start_ranking_smoke` | `ColdItemKNNScorer` | Same, capped at 25 candidate groups |
| `llm_yes_no_ollama_llama31_8b_smoke` | `LLMInteractionYesNoScorer` | Ollama, LLaMA 3.1 8B, 25 groups |
| `llm_yes_no_ollama_llama31_8b_full` | `LLMInteractionYesNoScorer` | Ollama, LLaMA 3.1 8B, all groups |
| `llm_yes_no_vllm_llama33_70b_smoke` | `LLMInteractionYesNoScorer` | vLLM, LLaMA 3.3 70B, 25 groups |
| `llm_yes_no_vllm_llama33_70b_full` | `LLMInteractionYesNoScorer` | vLLM, LLaMA 3.3 70B, all groups |
| `llm_yes_no_vllm_qwen36_27b_smoke` | `LLMInteractionYesNoScorer` | vLLM, Qwen 3.6 27B, 25 groups |
| `llm_yes_no_vllm_qwen36_27b_full` | `LLMInteractionYesNoScorer` | vLLM, Qwen 3.6 27B, all groups |

LLM prompt columns for ml-1m: history = `item_title`, `item_genres`, `rating`; candidate = `item_title`, `item_genres`.

### Key Design Decisions

**Temporal split within cold users is mandatory.** The k earliest temporal interactions form the profile; post-k interactions are the evaluation target. Random sampling within a cold user's sequence is invalid — it would allow future interactions to contaminate the visible profile.

**`task.train` must not be passed to the LLM scorer.** That frame contains only warm-user rows; the LLM would construct empty per-cold-user histories and ignore the profile entirely. `task.online_session_history` is the only correct fit source for cold-user context.

**`ColdItemKNNScorer` needs two separate fit steps.** Item-item cosine similarity is built from warm train co-occurrence; aggregation profiles come from `online_session_history`. Merging the frames would corrupt the similarity matrix with cold-user counts. Mean aggregation (not sum) keeps raw scores scale-comparable across k values so the k-curve is interpretable.

**Threshold calibration uses val cold users, not test.** Popularity and ItemKNN output real-valued scores. The threshold is selected on val via `find_best_user_group_threshold` using `macro_by_user_group_mean_f1` — the same aggregation as the headline test metric — then fixed before test evaluation.
