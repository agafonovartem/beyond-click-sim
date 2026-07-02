## Cold Start

### Scientific Question

Can an LLM given only a short user profile — items the user interacted with before any training data was collected — predict per-candidate relevance competitively with classical low-data baselines (Popularity, ItemKNN)?

This experiment isolates **Normal User Cold-Start**: users are entirely absent from training. Every candidate scored is for a user the model has never seen. Items remain warm (all items in cold users' profiles and evaluation sets exist in the training item catalog).

### Split Design

The cold-start split is fundamentally different from the in-distribution random fraction split. It partitions **users**, not interactions.

**Step 1 — User partition (random, seed-controlled)**

Users are divided into three non-overlapping groups:

| Group | Fraction | Role |
|---|---|---|
| Warm users | `train_fraction` (default 0.7) | Fully observed; all their interactions go into `train` |
| Val-cold users | `val_fraction` (default 0.1) | Never seen in training; evaluated on val |
| Test-cold users | `test_fraction` (default 0.2) | Never seen in training; evaluated on test |

No cold user — and none of their interactions — appears in `train`. The partition is deterministic via `stable_sample_values` and a seed.

**Step 2 — Temporal split within each cold user (mandatory)**

For every cold user, sort all their interactions by `timestamp` ascending, with ties broken by `item_id` ascending for reproducibility. Cut at position `k`:

```
cold user's full history (sorted by timestamp):
  [t0, t1, …, t_{k-1}]  →  online_session_history  (k earliest interactions)
  [t_k, t_{k+1}, …]     →  evaluation target        (post-k interactions → val or test)
```

The temporal boundary is the only valid cut. No random sampling is applied within a cold user's interaction sequence. Cold users whose total interaction count is ≤ k produce zero post-k rows and are **dropped from evaluation entirely** (drop counts are recorded in the task manifest).

### `online_session_history`

`online_session_history` is a DataFrame of the first `k` temporal interactions per cold user. It is separate from `train`, `val`, and `test` — it is the minimal context window available for cold users and is used exclusively by scorers that rely on user history (LLM prompt construction, ItemKNN profile aggregation). It is **not** part of the candidate evaluation sets.

The `k` axis produces a curve rather than a single point: running the experiment for `k ∈ {1, 3, 5}` reveals how much each additional profile item improves cold-user relevance prediction.

### Classes

**`ColdStartSplitFrames`** (`src/beyond_click_sim/tasks/cold_start.py`)

Frozen dataclass returned by `ColdUserHoldoutSplitter.split()`:

```python
@dataclass(frozen=True)
class ColdStartSplitFrames:
    train: pd.DataFrame                   # warm users' full interactions
    val: pd.DataFrame                     # val-cold users' post-k interactions (raw positives)
    test: pd.DataFrame                    # test-cold users' post-k interactions (raw positives)
    online_session_history: pd.DataFrame  # all cold users' first k interactions
    warm_user_ids: frozenset
    val_cold_user_ids: frozenset
    test_cold_user_ids: frozenset
    dropped_val_cold_count: int
    dropped_test_cold_count: int
```

`val` and `test` at this stage are raw positive interaction rows — no negatives yet. Negative sampling is the task builder's responsibility.

**`ColdUserHoldoutSplitter`** (`src/beyond_click_sim/tasks/cold_start.py`)

Does not subclass `Splitter` (return type is `ColdStartSplitFrames`, not `SplitFrames`).

```python
ColdUserHoldoutSplitter(
    k=3,
    train_fraction=0.7,
    val_fraction=0.1,
    test_fraction=0.2,
    seed=0,
    timestamp_column="timestamp",
)
```

Raises `ValueError` if `timestamp_column` is absent — Steam has no wall-clock timestamps and is unsupported by this splitter.

**`ColdStartTask`** (`src/beyond_click_sim/tasks/cold_start.py`)

Extends `Task` with two additional fields:

```python
@dataclass
class ColdStartTask(Task):
    online_session_history: pd.DataFrame  # cold users' first k interactions with features
    k: int                                # profile size used to build the split
```

Field semantics for scorers:

| Field | Used by | How |
|---|---|---|
| `task.train` | `PopularityScorer`, `ColdItemKNNScorer` | Fit on warm-user interactions (item counts, item-item similarity matrix) |
| `task.online_session_history` | `LLMInteractionYesNoScorer`, `ColdItemKNNScorer` | Cold user's k-item visible profile; passed to both scorers' `fit()` — the same frame object |
| `task.val` | Threshold selection (Popularity, ItemKNN) | Same candidate-group format as in-distribution val; used to calibrate the decision threshold |
| `task.test` | All scorers — evaluation | Same candidate-group format as in-distribution test |

`online_session_history` shares the same column schema as `train` (user/item features, history context columns, target, sampled, candidate_group) so scorers can call `fit()` on either frame.

**`ColdStartTaskBuilder`** (`src/beyond_click_sim/tasks/cold_start.py`)

Does not subclass `AlignmentInteractionTaskBuilder`. Build sequence:

1. Apply `dataset_filter` (e.g. `MinUserInteractionsFilter(10)`) to raw canonical tables.
2. Call `ColdUserHoldoutSplitter.split(filtered_interactions)` → `ColdStartSplitFrames`.
3. Build exclusion set for negative sampling: `concat(split.online_session_history, split.val, split.test)`. This ensures profile items and post-k positive items are never drawn as negatives.
4. Sample candidates from `split.val` → val candidate rows (with negatives).
5. Sample candidates from `split.test`, passing val negative pairs as `excluded_pairs` → test candidate rows.
6. Enrich items from `split.train` only (train-only item statistics remain uncontaminated by cold user activity).
7. Join user/item features.
8. Return `ColdStartTask`.

### Negative Sampling

Reuses `CappedUserInteractionCandidateSampler` with the same `negative_ratio` and `total_items=20` parameters as the in-distribution interaction prediction setup. The exclusion set (`interactions=`) is all cold user interactions — history plus raw val and test positives — so a cold user's negative pool excludes every item they have ever interacted with at any point in their history.

### Scorer Protocol for Cold-Start Runners

The cold-start runner must not pass `task.train` to the LLM scorer. The correct protocol:

```python
# LLM scorer: fit on cold user profiles, not warm interactions
X_history, y_history = split_xy(task.online_session_history, target_column=schema.target_column)
llm_scorer.fit(X_history, y_history)

# Popularity: fit on warm interactions only
X_train, y_train = split_xy(task.train, target_column=schema.target_column)
popularity_scorer.fit(X_train, y_train)

# ItemKNN: two-step fit — warm train builds the similarity matrix; history stores cold user profiles
item_knn_scorer.fit_train(X_train, y_train).fit(X_history, y_history)
```

`fit_train()` and `fit()` are separate because `ColdItemKNNScorer` needs two data sources for two distinct purposes: the warm train set for the item-item cosine similarity matrix, and `online_session_history` for the cold users' aggregation profiles. No scorer code changes are required for LLM or Popularity.

### Dataset Support

| Dataset | Supported | Notes |
|---|---|---|
| `ml-1m` | Yes | `timestamp` column present (Unix epoch seconds) |
| `steam` | No | Steam interactions are a library ownership snapshot with no wall-clock timestamps |

---

### Runner Configuration

**Module:** `runners/in_distribution/cold_start/task_builders.py`

The cold-start runner produces tasks for every combination of `(dataset, k, negative_ratio, seed)`. Constants mirror the in-distribution interaction-prediction setup where the cold-start design leaves them unchanged:

| Constant | Value | Notes |
|---|---|---|
| `K_VALUES` | `(1, 3, 5)` | Profile sizes to sweep — produces a k-curve |
| `DATASETS` | `("ml-1m",)` | Steam excluded (no timestamps) |
| `SEEDS` | `(0, 1, 2, 3, 4)` | Five random seeds for user partition |
| `REDUCED_SEEDS` | `(0, 1, 2)` | Used for `DEFAULT_TASK_NAMES` |
| `MIN_INTERACTIONS` | `10` | Minimum interactions per user before splitting |
| `NEGATIVE_RATIOS` | `(1, 2, 3, 9, 19)` | Same set as in-distribution |
| `TRAIN_FRACTION` | `0.7` | Warm-user fraction |
| `VAL_FRACTION` | `0.1` | Val-cold-user fraction |
| `TEST_FRACTION` | `0.2` | Test-cold-user fraction |
| `TOTAL_CANDIDATE_ITEMS` | `20` | Max candidates per group (positive + negatives) |

**Task name format:** `{dataset}_cold_start_k{k}_cap20_m{negative_ratio}_seed{seed}`

Example: `ml-1m_cold_start_k3_cap20_m1_seed0`

Total task count: 3 k-values × 5 negative ratios × 5 seeds = **75 tasks** (ml-1m only).

`DEFAULT_TASK_NAMES` covers `k ∈ {1, 3, 5}`, `negative_ratio=1`, `seed ∈ {0, 1, 2}` — 9 tasks — and is what the runner executes when no `--tasks` argument is provided.

**History context columns:** `("rating",)` for ml-1m, consistent with the in-distribution interaction-prediction setup.

Each task is built by `build_cold_start_task(dataset_name, k, negative_ratio, seed)`, which instantiates:

```python
ColdStartTaskBuilder(
    name=task_name,
    dataset_filter=MinUserInteractionsFilter(min_interactions=10),
    splitter=ColdUserHoldoutSplitter(
        k=k,
        train_fraction=0.7,
        val_fraction=0.1,
        test_fraction=0.2,
        seed=seed,
    ),
    sampler=CappedUserInteractionCandidateSampler(
        negative_ratio=negative_ratio,
        total_items=20,
        seed=seed,
    ),
    history_context_columns=("rating",),
).build(dataset)
```

`load_canonical_dataset` and `repo_root` are imported from `runners.in_distribution.interaction_prediction.task_builders` — no duplication.

---

### Runner

**Entry point:** `runners/in_distribution/cold_start/run.py`

Structurally identical to `runners/in_distribution/interaction_prediction/run.py`. Builds each requested task once, then runs all requested methods against it. Default output root: `outputs/in_distribution/cold_start/`.

```bash
python -m runners.in_distribution.cold_start.run \
    [--tasks TASK1,TASK2,...] \
    [--methods METHOD1,METHOD2,...] \
    [--output-dir PATH]
```

- `--tasks`: comma-separated names from `TASK_BUILDERS`. Default: the 9 `DEFAULT_TASK_NAMES` (k ∈ {1,3,5}, m=1, seed ∈ {0,1,2}).
- `--methods`: comma-separated names from `METHOD_RUNNERS`. Default: `popularity_f1_threshold`.
- `--output-dir`: root for artifacts. Default: `outputs/in_distribution/cold_start/`.

Each run writes to a timestamped directory: `{output_dir}/{timestamp}_{task_name}_{method_name}/`.

**CLI example — popularity baseline across all three k values, seeds 0–2:**

```bash
python -m runners.in_distribution.cold_start.run \
    --methods popularity_f1_threshold \
    --output-dir outputs/in_distribution/cold_start
```

**CLI example — LLM smoke test for k=3, seed=0:**

```bash
python -m runners.in_distribution.cold_start.run \
    --tasks ml-1m_cold_start_k3_cap20_m1_seed0 \
    --methods llm_yes_no_vllm_llama33_70b_smoke \
    --output-dir outputs/in_distribution/cold_start
```

---

### Methods

**Module:** `runners/in_distribution/cold_start/methods/`

#### Popularity (`methods/popularity.py`)

Re-exports `run` and `run_ranking` from `runners.in_distribution.interaction_prediction.methods.popularity` unchanged. No logic change is needed because `PopularityScorer.fit(X_train, y_train)` receives `task.train`, which in `ColdStartTask` already contains only warm-user interaction rows. Item popularity counts are therefore computed from the training population, not contaminated by cold user activity.

Method names registered in `METHOD_RUNNERS`:

| Method name | Description |
|---|---|
| `popularity_f1_threshold` | Validation-threshold pointwise alignment; reports precision, recall, F1 |
| `popularity_ranking` | Raw-score ranking; reports NDCG@K, HR@K |

#### LLM Yes/No (`methods/llm_yes_no.py`)

Near-copy of `runners/in_distribution/interaction_prediction/methods/llm_yes_no.py` with **one critical change** inside `run_method()`: the scorer is fitted on `task.online_session_history` instead of `task.train`.

```python
# Cold-start: fit the LLM scorer on the cold user's k-item visible profile
X_history, y_history = split_xy(
    task.online_session_history,
    target_column=task.schema.target_column,
)
scorer = LLMInteractionYesNoScorer(...).fit(X_history, y_history)
```

Fitting on `task.train` instead would be a silent but critical bug: that frame contains only warm-user rows; the LLM would construct empty per-cold-user histories and produce responses that ignore the profile entirely.

Everything else — per-model wrapper functions, parallel group scoring, error handling, metrics computation, manifest and metrics JSON writing — is identical to the in-distribution version.

The scorer manifest records `"fit_on": "online_session_history"` and `"k": task.k` to make the cold-start provenance explicit.

`DATASET_PROMPT_COLUMNS` is restricted to `ml-1m` (Steam unsupported):

| Dataset | History columns | Candidate columns |
|---|---|---|
| `ml-1m` | `item_title`, `item_genres`, `rating` | `item_title`, `item_genres` |

Method names registered in `METHOD_RUNNERS`:

| Method name | Backend | Model | Scope |
|---|---|---|---|
| `llm_yes_no_ollama_llama31_8b_smoke` | Ollama | LLaMA 3.1 8B | 25 groups |
| `llm_yes_no_ollama_llama31_8b_full` | Ollama | LLaMA 3.1 8B | all groups |
| `llm_yes_no_vllm_llama33_70b_smoke` | vLLM | LLaMA 3.3 70B | 25 groups |
| `llm_yes_no_vllm_llama33_70b_full` | vLLM | LLaMA 3.3 70B | all groups |
| `llm_yes_no_vllm_qwen36_27b_smoke` | vLLM | Qwen 3.6 27B | 25 groups |
| `llm_yes_no_vllm_qwen36_27b_full` | vLLM | Qwen 3.6 27B | all groups |

Smoke variants cap evaluation at 25 candidate groups for fast sanity checks.

---

#### ItemKNN (`methods/item_knn.py`)

`ColdItemKNNScorer` (`src/beyond_click_sim/scorers/item_knn.py`) scores cold users by aggregating item-item cosine similarities over the user's k-item profile. It requires a **two-step fit** because it draws on two separate data sources for two distinct purposes.

**Step 1 — `fit_train(X_train, y_train)`**: Builds the item-item cosine similarity matrix from warm training interactions. Mirrors `ItemKNNPolicy.fit()` (`src/beyond_click_sim/tasks/policies.py`):

1. Construct a binary item-user sparse matrix (`csr_matrix`, shape `n_items × n_users`).
2. L2-normalize each item row (`sklearn.preprocessing.normalize`).
3. Find the top-`n_neighbors=20` nearest neighbors per item via brute-force cosine kNN (`sklearn.neighbors.NearestNeighbors`).
4. Drop each item's self-match; store `(neighbor_indices, neighbor_similarities)` per item.

`y_train` is accepted for API consistency but not used — similarity is binary co-occurrence only.

**Step 2 — `fit(X_history, y_history)`**: Stores cold user profiles from `task.online_session_history`. The profile for each cold user is exactly the k items in that frame — the same set the LLM scorer receives. Satisfies the `Scorer` ABC.

**`score(X)`**: For each cold user in `X`, accumulates neighbor similarities across all profile items using `np.add.at`, then divides by the number of contributing profile items (`aggregation="mean"`). Mean normalization keeps raw scores scale-comparable across k values: a k=5 user and a k=1 user with identical neighborhood structure receive the same per-item score, making the k-curve interpretable. Without normalization (sum), scores grow with k regardless of neighborhood quality.

Candidate items absent from warm train receive score 0.0; the cold-start spec guarantees all items are warm, so this is only a defensive fallback.

**Threshold calibration on validation.** `ColdItemKNNScorer.score()` produces real-valued similarity scores, not binary labels. The runner selects a threshold on val cold users:

```python
scorer = (
    ColdItemKNNScorer(n_neighbors=20, aggregation="mean")
    .fit_train(X_train, y_train)   # item-item cosine sim from warm train
    .fit(X_history, y_history)     # cold user k-item profiles from online_session_history
)
val_scores = scorer.score(X_val)
threshold_selection = find_best_user_group_threshold(
    y_val, val_scores, X_val[candidate_group_column], X_val["user_id"], metric="f1"
)
threshold = float(threshold_selection["threshold"])
test_predictions = apply_threshold(scorer.score(X_test), threshold)
```

`find_best_user_group_threshold` uses the same `macro_by_user_group_mean_f1` aggregation as the headline test metric, so the decision rule is calibrated consistently. The threshold is never re-tuned on test.

**Same candidate rows as LLM.** Both the LLM and ItemKNN runners use `xy["test"]` (and optionally `limit_candidate_groups` with the same cap). Their test candidate groups are identical, making results directly comparable.

Manifest records `"fit_on": "online_session_history"` and `"k": task.k` alongside `"n_neighbors"` and `"aggregation"` to make provenance explicit.

Method names registered in `METHOD_RUNNERS`:

| Method name | Description |
|---|---|
| `item_knn_cold_start` | Pointwise alignment with val-calibrated F1 threshold; reports precision, recall, F1 |
| `item_knn_cold_start_smoke` | Same, capped at 25 candidate groups |
| `item_knn_cold_start_ranking` | Raw-score ranking; reports NDCG@K, HR@K |
| `item_knn_cold_start_ranking_smoke` | Same, capped at 25 candidate groups |

Smoke variants are useful for fast sanity checks; the fit step always runs on the full train and history data regardless of the group cap.

---

### Output Artifacts

Each method run writes to a timestamped directory under the configured output root:

| File | Produced by | Contents |
|---|---|---|
| `manifest.json` | All methods | Full provenance: method name, scorer config (including `fit_on`, `k`, `n_neighbors` for ItemKNN; prompt columns for LLM), decision rule, limits, candidate group summary, task manifest, git commit |
| `metrics.json` | Pointwise methods (`popularity_f1_threshold`, `item_knn_cold_start*`) | `val` and `test` pointwise alignment metrics: `macro_by_user_group_mean` (headline), `macro_by_group`, `micro`; threshold and threshold metric |
| `metrics.json` | LLM methods | `test` pointwise alignment metrics only (no val threshold); scored/requested row counts |
| `metrics_ranking.json` | Ranking methods (`popularity_ranking`, `item_knn_cold_start_ranking*`, LLM runners) | `val` and `test` ranking metrics: `NDCG@{1,3,5,10}`, `HR@{1,3,5,10}`, computed by sorting raw scores within each candidate group |
| `predictions.parquet` | All methods | Row-level scores and binary predictions (pointwise) or raw scores (ranking) for val and test candidate rows |
| `llm_errors.jsonl` | LLM methods only | One JSON object per failed LLM call (empty if all calls succeeded); records candidate group ID, attempt count, and error messages |
