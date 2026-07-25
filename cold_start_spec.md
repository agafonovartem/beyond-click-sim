## Cold Start

### Scientific Question

Can an LLM given only a short user profile — items the user interacted with before any training data was collected — predict which items the user will **like** next, competitively with classical low-data baselines (Popularity, ItemKNN)?

This experiment isolates **Normal User Cold-Start**: users are entirely absent from training. Every candidate scored is for a user the model has never seen. **Items remain warm by construction**: history profiles, evaluation targets, and sampled negatives are restricted to items that appear in warm `train` (see Warm-Item Coverage below).

**Target definition:** `target_like_ge4` (rating ≥ 4), not raw interaction. Ambiguous items (rating = 3) and disliked consumed items (rating ≤ 2) are excluded from the evaluation frame entirely — they are neither targets nor negative candidates. This is a positive-preference target; do not describe results as "interaction prediction."

### Governing Constraint

Every manipulated axis changes exactly one thing and holds everything else — evaluated users, evaluated targets — fixed:

| Axis | Values | Controls | Holds fixed |
|---|---|---|---|
| `k` | {1, 3, 5} | Profile length — **and unavoidably target recency** (see k-sweep caveat: evaluated users are fixed, evaluated targets are not) | Evaluated users, protocol constants |
| `m` (negative ratio) | {1, 2, 3, 9, 19} | Number of negatives per target | Target frame, users, split |
| negative strategy | {`negu`, `negpop`} | How negatives are drawn | Target frame, users, split |
| `sampling_seed` | {0, 1, 2} | The negative draw only | Everything else — true replicates |
| `split_seed` | 0 (fixed) | User partition + eval-user selection | Fixed across the whole grid |

The constraint is enforced structurally: the target frame is built before and independently of negative sampling, and `target_frame_sha256` in each task manifest makes frame identity across tasks checkable without rebuilding them.

### Split Design

The cold-start split is fundamentally different from the in-distribution random fraction split. It partitions **users**, not interactions.

**Step 1 — User partition (random, `split_seed`-controlled)**

Users are divided into three non-overlapping groups:

| Group | Fraction | Role |
|---|---|---|
| Warm users | `train_fraction` (default 0.7) | Fully observed; all their interactions go into `train` |
| Val-cold users | `val_fraction` (default 0.1) | Never seen in training; evaluated on val |
| Test-cold users | `test_fraction` (default 0.2) | Never seen in training; evaluated on test |

No cold user — and none of their interactions — appears in `train`. The partition is deterministic via `stable_sample_values` and `split_seed` (fixed at 0 for the grid).

**Step 2 — Temporal split within each cold user (mandatory)**

For every cold user, sort all their interactions by `timestamp` ascending, with ties broken by `item_id` ascending for reproducibility. Cut at position `k`:

```
cold user's full history (sorted by timestamp):
  [t0, t1, …, t_{k-1}]  →  online_session_history  (k earliest interactions)
  [t_k, t_{k+1}, …]     →  evaluation pool         (post-k interactions → val or test)
```

The temporal boundary is the only valid cut. No random sampling is applied within a cold user's interaction sequence. Cold users whose total interaction count is ≤ k produce zero post-k rows and are **dropped** (recorded as `dropped_val_cold` / `dropped_test_cold` in the manifest).

### Warm-Item Coverage (strict, enforced)

After the user/temporal split, `ColdStartTaskBuilder` derives `warm_item_ids` from items present in warm `train` and applies a **strict** coverage policy:

| Stage | Rule | Drop / filter accounting |
|---|---|---|
| History | A cold user is eligible only if all `k` `online_session_history` items are in `warm_item_ids` | `users.dropped_nonwarm_history_{val,test}` and `warm_item_coverage.dropped_users_nonwarm_history_{val,test}` |
| Targets | Post-k rows on non-warm items are removed before `select_cold_start_targets` | `warm_item_coverage.dropped_post_k_rows_nonwarm_{val,test}`; users left with zero warm likes still count under `dropped_zero_qualifying_targets_{val,test}` |
| Negatives | The sampler receives only the warm-item catalog (`items` filtered to `warm_item_ids`) | Catalog-only / cold-only items cannot appear as negatives |

This is a protocol-breaking enforcement relative to earlier runs that only *claimed* warm items. Manifest key `warm_item_coverage` records `{enforced: true, policy: "strict", warm_items, ...}`. Always read evaluated-user counts from the manifest after this filter — eligibility pools can shrink.

### Target Frame (built once per `(dataset, k, split_seed)`)

The evaluation frame is constructed **before** any negative sampling, by `select_cold_start_targets` (`src/beyond_click_sim/tasks/cold_start.py`):

- Per cold user, take the first **C = 5** items with `target_like_ge4 == 1` from their **warm-only** post-k eval-pool, in temporal order.
- Users with zero qualifying post-k items (after warm-item filtering) are dropped **here**, recorded separately as `dropped_zero_qualifying_targets_{val,test}` — a distinct drop point from the splitter's `dropped_*_cold_count` and from non-warm history drops.
- The frame never sees `m`, the negative strategy, or `sampling_seed`. Tasks sharing `(dataset, k, split_seed)` therefore share a byte-identical frame by construction; `target_frame_sha256` (per split) in the manifest is the checkable invariant.

**Eval-user pool (fixed across k).** Up to `EVAL_USERS` (1000) evaluated users per split are selected from users **eligible at `pool_k = 5`**: warm history at `pool_k` **and** ≥ 1 qualifying **warm** target in their post-position-5 eval-pool. Val users are sampled with `stable_sample_values(seed=split_seed)`, test users independently with `seed=split_seed + 1` (mirroring the splitter's own partition seeds). When an eligible pool is smaller than `EVAL_USERS`, all eligible users are taken — always read `users.evaluated_{val,test}` from the manifest rather than assuming 1000. Eligibility at k=5 implies eligibility at k ∈ {1, 3} because eval-pools and history prefixes are nested in k — so the same users are evaluated across the whole k-grid. Selection never depends on `sampling_seed`.

**k-sweep caveat.** Holding users fixed across k is exact; holding targets fixed is impossible — the first C liked items after position k move later as k grows. `k` is therefore a joint manipulation of profile length and target recency. Read the k-curve as "profile length + how far the targets are from the profile," not as pure profile-length marginal effects.

### Candidate Groups and Negative Sampling

Each frame row becomes exactly one candidate group: **1 positive (the target) + m negatives**. Group ids are target-keyed: `candidate:user:{user_id}:item:{item_id}`.

The exclusion set for negatives is all cold-user interactions (profile + val + test rows), so any item the user has ever consumed — at any rating — is never drawn as their negative. The **candidate pool** itself is the warm-train item set only (strict coverage).

Two strategies, selected per task (`PerTargetCandidateSampler`, `src/beyond_click_sim/tasks/samplers.py`):

| Strategy | Name segment | Semantics |
|---|---|---|
| `uniform` | `negu` | Negatives drawn uniformly from the **warm** catalog. Known caveat: makes groups partly popularity-solvable (most random items are tail items). |
| `popularity_matched` | `negpop` | Items are binned into up to 10 quantile bins of **warm-train** interaction counts (`pd.qcut` with duplicate edges dropped, so heavy ties at low counts can yield fewer bins); each target's negatives come from the target's own bin. Removes the within-group popularity signal. If a bin lacks enough eligible items, the pool expands to adjacent bins and the group is counted in `sampler_diagnostics.{val,test}.bin_fallback_groups`. |

Popularity bins use warm train interactions only — no cold-user activity contaminates the matching.

**`sampling_seed` drives only the negative draw** (and the within-group row shuffle). Mean ± std over `sampling_seed ∈ {0, 1, 2}` therefore estimates negative-sampling variance of one fixed protocol, which is what seed-averaging claims to measure.

### Group-Id Uniqueness Contract

`candidate_group` is unique within `(dataset, k, m, negative_strategy, sampling_seed, split)` — and only within that key. For a fixed k, the same id legitimately appears in up to **60 task variants** (5 m × 2 strategies × 3 sampling seeds × 2 item-stats variants): same target, different negatives and/or item metadata.

Cold-start runners therefore write `k`, `negative_ratio`, `sampling_seed`, and `negative_strategy` as constant columns in `predictions.parquet` (via `cold_start_metadata`, `runners/in_distribution/interaction_prediction/methods/common.py`). Two things these columns deliberately do **not** capture:

- **item-stats variant** — plain and `item_stats` tasks share ids and all four columns;
- **method** — different methods on the same task produce the same ids with different scores.

Run identity (method, item-stats variant, timestamp) exists only in the output directory name. Any cross-run analysis must therefore carry the run directory's method/task name alongside the parquet rows, and group by `(method, k, negative_ratio, negative_strategy, sampling_seed, split)` — never by `candidate_group` alone.

### `online_session_history`

`online_session_history` is a DataFrame of the first `k` temporal interactions per evaluated cold user. It is separate from `train`, `val`, and `test` — it is the minimal context window available for cold users and is used exclusively by scorers that rely on user history (LLM prompt construction, ItemKNN profile aggregation, Agent4Rec profile generation). It is **not** part of the candidate evaluation sets.

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

**`ColdUserHoldoutSplitter`** (`src/beyond_click_sim/tasks/cold_start.py`)

Does not subclass `Splitter` (return type is `ColdStartSplitFrames`, not `SplitFrames`). Raises `ValueError` if the timestamp column is absent — Steam has no wall-clock timestamps and is unsupported.

**`select_cold_start_targets` / `target_frame_sha256`** (`src/beyond_click_sim/tasks/cold_start.py`)

Frame construction and the frame-identity hash; see Target Frame above.

**`ColdStartTask`** (`src/beyond_click_sim/tasks/cold_start.py`)

Extends `Task` with `online_session_history` and `k`. Field semantics for scorers:

| Field | Used by | How |
|---|---|---|
| `task.train` | `PopularityScorer`, `ColdItemKNNScorer` | Fit on warm-user interactions (liked-item counts, item-item similarity matrix) |
| `task.online_session_history` | `LLMInteractionYesNoScorer`, `LLMPreferenceListwiseRankingScorer`, `ColdItemKNNScorer`, `Agent4RecYesNoScorer`, `Agent4RecPreferenceListwiseRankingScorer` | Cold user's k-item visible profile; passed to history/profile scorers' `fit()` — the same frame object |
| `task.val` | Threshold selection (Popularity, ItemKNN) | Same candidate-group format as test; used to calibrate the decision threshold |
| `task.test` | All scorers — evaluation | Candidate groups of 1 positive + m negatives |

`online_session_history` shares the same column schema as `train` (user/item features, history context columns, target, sampled, candidate_group) so scorers can call `fit()` on either frame.

**`ColdStartTaskBuilder`** (`src/beyond_click_sim/tasks/cold_start.py`)

Build sequence:

1. Apply `dataset_filter` (`MinUserInteractionsFilter(10)`) to raw canonical tables.
2. `ColdUserHoldoutSplitter(k).split(...)` → `ColdStartSplitFrames`.
3. Derive `warm_item_ids` from `split.train`; drop cold users whose k-item history is not fully warm.
4. Select eval users from the `pool_k`-eligibility pool (warm history + ≥1 warm qualifying target; `split_seed` only); restrict val/test/history.
5. Filter post-k rows to warm items; build target frames via `select_cold_start_targets`; count zero-qualifying drops.
6. Build the negative exclusion set: `concat(split.online_session_history, split.val, split.test)`.
7. Sample negatives per frame row from the **warm-item catalog only** (`PerTargetCandidateSampler`; warm train passed for popularity bins).
8. Enrich items from `split.train` only (train-only item statistics remain uncontaminated).
9. Join user/item features; blank history-context columns in val/test candidates.
10. Return `ColdStartTask` with full manifest (see Manifest below).

### Scorer Protocol for Cold-Start Runners

The cold-start runner must not pass `task.train` to the LLM scorer. The correct protocol:

```python
# LLM scorer: fit on cold user profiles, not warm interactions
X_history, y_history = split_xy(task.online_session_history, target_column=schema.target_column)
llm_scorer.fit(X_history, y_history)

# Popularity: fit on warm interactions only. Note: with target_like_ge4 as the
# target source, PopularityScorer sums liked-item counts per item — a
# like-popularity baseline matched to the target definition.
X_train, y_train = split_xy(task.train, target_column=schema.target_column)
popularity_scorer.fit(X_train, y_train)

# ItemKNN: two-step fit — warm train builds the similarity matrix; history stores cold user profiles
item_knn_scorer.fit_train(X_train, y_train).fit(X_history, y_history)
```

**Agent4Rec activity trait.** By default, activity uses **fit-frame row counts** on `online_session_history` (`activity_counts=None` → manifest `profile_generator.activity_source = "fit_frame_row_counts"`). For cold users that means every profile has exactly k rows, so the activity tercile is degenerate. That is intentional: feeding full-dataset totals via `_full_dataset_activity_counts` / `activity_counts={...}` would be a **metadata leakage** — those counts include post-k / evaluation-period interactions, so the activity text in the system prompt would encode held-out interaction volume rather than k-shot information alone. Do not enable full-dataset activity counts unless running an explicit metadata-visibility ablation. Diversity and conformity still come from the k-item profile.

### Metrics Caveats

- **Ranking across m:** group size = 1 + m, so @k metrics with k > group size are degenerate (`k_eff = min(k, group_size)`). `ndcg@5`/`hit_rate@5` are uninterpretable for m < 4 and non-comparable across m. Use **@1** as the cross-m headline; keep @1/3/5/10 for within-m analysis and check `groups_with_size_lte@k`.
- **LLM ranking on binary scores (yes/no families):** `llm_yes_no` and `agent4rec_yes_no` emit 0/1 scores, so within-group ranking is resolved entirely by the tie policy (`average` = expected value under uniform tie-breaking). Ranking metrics for those methods carry no graded discrimination — only the per-group yes/no pattern. Use a direct `llm_preference_listwise_ranking` or `agent4rec_preference_listwise_ranking` method for ranking claims. Direct listwise removes binary ties by eliciting a complete ranking, but still forces a relative order over one-positive sampled groups and remains sensitive to candidate presentation order.
- **ItemKNN zero floor:** under strict warm-item coverage, every history/target/negative item appears in warm train, so the previous "item absent from warm train → score 0.0" case should not arise for candidates. Scores can still be exactly 0.0 for warm items with no usable neighbor mass; report zero-score rates when interpreting KNN-vs-LLM gaps.

### Dataset Support

| Dataset | Supported | Notes |
|---|---|---|
| `ml-1m` | Yes | `timestamp` column present; `target_like_ge4` available |
| `steam` | No | No wall-clock timestamps; splitter raises |

---

### Runner Configuration

**Module:** `runners/in_distribution/cold_start/task_builders.py`

| Constant | Value | Notes |
|---|---|---|
| `K_VALUES` | `(1, 3, 5)` | Profile sizes — produces a k-curve |
| `DATASETS` | `("ml-1m",)` | Steam excluded (no timestamps) |
| `SAMPLING_SEEDS` | `(0, 1, 2)` | Negative-draw replicates |
| `SPLIT_SEED` | `0` | Fixed: partition + eval-user selection |
| `MIN_INTERACTIONS` | `10` | Minimum interactions per user before splitting |
| `NEGATIVE_RATIOS` | `(1, 2, 3, 9, 19)` | m per target |
| `NEGATIVE_STRATEGIES` | `("uniform", "popularity_matched")` | `negu` / `negpop` name segments |
| `C_TARGETS` | `5` | Liked targets per user per split |
| `EVAL_USERS` | `1000` | Eval users per split |
| `POOL_K` | `5` | Eligibility defined at the largest k |
| `TARGET_SOURCE_COLUMN` | `"target_like_ge4"` | Positive-preference target |
| `TRAIN/VAL/TEST_FRACTION` | `0.7 / 0.1 / 0.2` | User partition |

**Task name format:**

```
{dataset}[_item_stats]_cold_like_k{k}_c{C}_eval{U}_m{m}_{negu|negpop}_split{S}_sample{s}
```

Example: `ml-1m_cold_like_k3_c5_eval1000_m9_negpop_split0_sample1`

Total: 3 k × 5 m × 2 strategies × 3 sampling seeds × {plain, item_stats} = **180 task builders** (ml-1m only). Building is cheap; runs are selected explicitly.

`DEFAULT_TASK_NAMES`: k=3, m ∈ {1, 19}, both strategies, sampling_seed=0 — 4 smoke-friendly tasks.

Each task is built by `build_cold_start_task(dataset_name, k, negative_ratio, negative_strategy, sampling_seed, *, use_item_rating_stats=False)`.

**History context columns:** `("rating",)` for ml-1m.

---

### Runner

**Entry point:** `runners/in_distribution/cold_start/run.py` — unchanged CLI:

```bash
python -m runners.in_distribution.cold_start.run \
    [--tasks TASK1,TASK2,...] \
    [--methods METHOD1,METHOD2,...] \
    [--output-dir PATH] [--resume]
```

Default output root: `outputs/in_distribution/cold_start/`. Each run writes to `{output_dir}/{timestamp}_{task_name}_{method_name}/`.

---

### Methods

**Module:** `runners/in_distribution/cold_start/methods/`

#### Visible item metadata (prompt columns)

What a method can see is an explicit experiment condition. Per candidate, and per history item:

| Method family | History columns | Candidate columns |
|---|---|---|
| `llm_yes_no` (default: `summary_visibility="none"`) | `item_title`, `item_genres`, `rating` | `item_title`, `item_genres` |
| `llm_preference_listwise_ranking` (Qwen3.6-{27B,35B-A3B}; default) | `item_title`, `item_genres`, `rating` | `item_title`, `item_genres` |
| `llm_preference_listwise_ranking` `_sumboth` methods | + `item_summary` (profile items only) | + `item_summary` |
| `agent4rec_preference_listwise_ranking` traits (Qwen3.6-27B) | Agent4Rec traits from the k-item history | `item_title`, train-only `item_rating_mean`, `item_genres`, and canonical summary when available |
| `agent4rec_preference_listwise_ranking` traits+taste (Qwen3.6-27B + GPT-4o-mini taste) | Agent4Rec traits and cached taste from the k-item history | `item_title`, train-only `item_rating_mean`, `item_genres`, canonical summary |
| `agent4rec_preference_listwise_ranking` traits+taste `_sumboth` methods | profile summaries added to the taste-generation prompt | same as traits+taste + candidate summary |
| `llm_yes_no` with `use_item_stats` (supported by the runner, **not** currently exposed as a named registry method) | + `item_rating_mean` ("average rating"), `item_rating_count` ("number of prior reviews") | + `item_rating_mean`, `item_rating_count` |
| `llm_yes_no` `_sumboth` methods | + `item_summary` (profile items only) | + `item_summary` |
| `agent4rec_yes_no` (default; **requires `item_stats` tasks**) | k-item profile encoded in the profile text (traits/taste) | `item_title`, `item_rating_mean` ("History ratings"), `item_genres`, `item_summary` |
| `agent4rec_yes_no` `_sumboth` methods (taste only) | profile summaries added to the taste-generation prompt | same as default |

Notes:

- **Agent4Rec default card now includes "History ratings"** (`item_rating_mean`), matching the interaction-prediction Agent4Rec default. All Agent4Rec methods require an `item_stats` task variant and fail fast at setup on plain tasks. The retired `_item_stats` method-name segment no longer exists.
- `item_rating_mean` / `item_rating_count` are computed from **warm train interactions only** (`TrainItemRatingStatistics`) — no cold-user activity leaks into item statistics (regression-tested). Items unseen in warm train have NaN means and the prompt formatter silently drops the field for them. The two LLM families differ in what item stats expose: `llm_yes_no` (when enabled) shows mean **and** count, Agent4Rec shows mean only.
- **Summaries are static item metadata** (canonical movie summaries, provenance-checked via `task_item_summary_metadata`; the check covers train/val/test **and** `online_session_history`). History-side summaries — `llm_yes_no` `_sumboth` and Agent4Rec profile summaries — can only ever cover `online_session_history` (the k-item profile): that is the only history frame the scorers fit on, so post-k items can never appear there. Agent4Rec profile summaries affect only the **taste-generation prompt** (traits never see them), so `_sumboth` exists only for taste methods; profile-summary runs use a separate `*_summary.jsonl` taste cache file.
- History `rating` is the user's own rating of a profile item; candidate rows have history-context columns blanked (`_without_history_context`) so no feedback about the target leaks.

#### Method families

- **Popularity** (`methods/popularity.py`): re-exports the interaction-prediction runner unchanged. Fits on `task.train`; with `target_like_ge4` as the target source the item scores are liked-item counts — a like-popularity baseline matched to the target definition. Threshold calibrated on val cold users (`macro_by_user_group_mean_f1`), never re-tuned on test. Method names: `popularity_f1_threshold` (pointwise), `popularity_ranking` (raw-score ranking).
- **ItemKNN** (`methods/item_knn.py`): `ColdItemKNNScorer` two-step fit — `fit_train(task.train)` builds the item-item cosine similarity from warm co-occurrence (binary, ignores ratings); `fit(task.online_session_history)` stores the k-item profiles. Scores aggregate neighbor similarities over profile items (`aggregation="mean"`, `n_neighbors=20`); candidates absent from warm train score 0.0 (see Metrics Caveats). Same val-threshold protocol as Popularity. Method names: `item_knn_cold_start[_smoke]`, `item_knn_cold_start_ranking[_smoke]` (smoke = first 25 candidate groups).
- **LLM yes/no** (`methods/llm_yes_no.py`): `LLMInteractionYesNoScorer` fitted on `task.online_session_history` (critical — `task.train` has no cold users). Hard binary yes/no parser, temperature 0. Two scoring modes per model: `batch` (one call per candidate group) and `itemwise` (one call per candidate — method names with `_itemwise_`). Naming pattern: `llm_yes_no[_itemwise]_{backend}_{model}[_sumboth]_{smoke|full}` with `{backend}_{model}` ∈ `ollama_llama31_8b`, `vllm_llama33_70b`, `vllm_qwen36_27b`, `vllm_qwen36_35b_a3b`; `_sumboth` adds history+candidate summaries (see Visible item metadata).
- **LLM direct listwise preference ranking** (`methods/llm_listwise_ranking.py`): `LLMPreferenceListwiseRankingScorer` fitted on `task.online_session_history` with target text "The user would rate the candidate movie at least 4 out of 5.", one strict complete-ranking call per candidate group (temperature 0, thinking disabled, max tokens 512). Method names: `llm_preference_listwise_ranking_vllm_qwen36_{27b|35b_a3b}[_sumboth]_{smoke|full}`; `_sumboth` adds history+candidate summaries (same summary provenance checks as `llm_yes_no` `_sumboth`). This runner is **ranking-only**: it scores val and test directly, writes only `metrics_ranking.json`, does not choose a validation threshold, and uses `test_failure_as_zero_group.macro_by_user_group_mean.ndcg@1` as the cross-m headline while keeping parsed-only @1/3/5/10 diagnostics.
- **Agent4Rec yes/no** (`methods/agent4rec_yes_no.py`): `Agent4RecYesNoScorer` + `Agent4RecProfileGenerator`, fitted on `task.online_session_history`. Profiles: deterministic social **traits** (activity, conformity, diversity) plus optional LLM-generated **taste** (`gpt-4o-mini`, cached per split seed under `outputs/agent4rec_taste_cache/`; requires `OPENAI_API_KEY` on cache miss; profile-summary runs use the separate `*_summary.jsonl` cache). Activity comes from fit-frame counts by default (degenerate at small k; full-dataset counts are an opt-in ablation via `activity_counts` — a documented metadata-visibility condition). The default item card includes train-only "History ratings", so **all Agent4Rec methods require `item_stats` tasks** and fail fast at setup on plain tasks. Naming pattern: `agent4rec_yes_no[_itemwise]_{backend}_{model}[_traits_taste_gpt4o_mini[_sumboth]]_{smoke|full}`; `_sumboth` (candidate+profile summaries) exists only for taste variants.
- **Agent4Rec direct listwise preference ranking** (`methods/agent4rec_listwise_ranking.py`): `Agent4RecPreferenceListwiseRankingScorer` fitted only on `task.online_session_history`, with `activity_counts=None`, then one strict complete-ranking call per original val/test candidate group for the same rating≥4 target. Qwen3.6-27B and Qwen3.6-35B-A3B methods cover `traits`, `traits_taste_gpt4o_mini_candidate_summary`, and `traits_taste_gpt4o_mini_sumboth`, each in smoke/full form. Scoring uses `vllm_local`, 128 workers, temperature 0, thinking disabled, and 1000 output tokens. Taste generation uses GPT-4o-mini; candidate-summary variants use the non-summary taste cache (`use_history_item_summaries=False`), while `_sumboth` variants use history-summary taste caching (`use_history_item_summaries=True`) because profile summaries are part of the taste prompt. These methods require `item_stats` tasks and are ranking-only: no threshold or `metrics.json`, val is developmental, test is reported, and the failure-aware NDCG@1 path above is the headline. Traits-only and traits+taste are distinct metadata/profile conditions, not a pure Agent4Rec-versus-history-LLM comparison: Agent4Rec sees train-only item means and may see canonical summaries.

All methods write `k`, `negative_ratio`, `sampling_seed`, `negative_strategy` columns into `predictions.parquet` (see Group-Id Uniqueness Contract).

**LLM failure handling.** Cold-start LLM runs carry an error budget: if the candidate-group failure rate exceeds `max_error_rate` (default 10%, checked after the first 20 groups and at the end of the run), the run aborts with `LLMErrorRateExceededError` and writes `llm_errors.jsonl` but no metrics. Below that budget, headline metrics are computed on successfully parsed groups only (issues.md #1), with failure-penalized variants published alongside — `test_failure_as_negative` (pointwise; failed rows counted as `no`) and `test_failure_as_zero_group` (ranking; groups with any missing score contribute zero utility) — plus a row-level `coverage` summary. This applies to `llm_yes_no`, both direct-listwise families, and `agent4rec_yes_no`. For listwise, errors are split-labeled (`val`/`test`) in `llm_errors.jsonl`, and val/test budgets are independent. Agent4Rec taste generation is a separate pre-scoring phase: cache misses use `profile_generator.taste_max_attempts`; exhausted taste retries abort before ranking and are not counted or labeled as candidate-group failures.

---

### Manifest (task-level, inside every run's `manifest.json`)

New/important keys:

| Key | Meaning |
|---|---|
| `target_source_column` | `"target_like_ge4"` |
| `targets_per_user` | C = 5 |
| `split_seed` / `sampling_seed` / `negative_strategy` | Seed roles and strategy, top level |
| `pool_k` | Eligibility k for the eval-user pool |
| `warm_item_coverage` | Strict coverage flag/policy, warm item count, and non-warm drop counters |
| `target_frame_sha256.{val,test}` | Frame-identity hash — must match across m/strategy/sampling_seed tasks |
| `users.evaluated_{val,test}` | Actually evaluated users (≤ EVAL_USERS) |
| `users.dropped_zero_qualifying_targets_{val,test}` | Frame-stage drops (distinct from splitter drops) |
| `users.dropped_nonwarm_history_{val,test}` | Users dropped because k-item history was not fully warm |
| `targets_per_user_distribution.{val,test}` | min/mean/max targets per user |
| `sampler_diagnostics.{val,test}.bin_fallback_groups` | Popularity-bin fallback count (present for both strategies; always 0 for `negu`) |

---

### Output Artifacts

| File | Contents |
|---|---|
| `manifest.json` | Full provenance: method, scorer config (`fit_on`, `k`; for Agent4Rec `profile_generator.activity_source`), decision rule, limits, candidate group summary, task manifest (above), git commit |
| `metrics.json` | Pointwise metrics. Popularity/ItemKNN: `val` + `test`. LLM yes/no families (`llm_yes_no`, `agent4rec_yes_no`): `test` (parsed-only) plus `test_failure_as_negative` and `coverage`. Headline: `test.macro_by_user_group_mean.f1` |
| `metrics_ranking.json` | Ranking metrics NDCG/HR @{1,3,5,10}. Both direct-listwise families report `val` and `test`, plus `*_failure_as_zero_group`, with headline `test_failure_as_zero_group.macro_by_user_group_mean.ndcg@1` (cross-m). Yes/no-based ranking artifacts remain tie-heavy diagnostics (see Metrics Caveats / issues #11). |
| `predictions.parquet` | Row-level scores/predictions for val and test, with `split`, `k`, `negative_ratio`, `sampling_seed`, `negative_strategy` columns |
| `llm_errors.jsonl` | One JSON object per failed LLM call (empty if all succeeded) |
