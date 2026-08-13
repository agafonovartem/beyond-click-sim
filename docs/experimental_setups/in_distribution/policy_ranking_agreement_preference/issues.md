# Issues: preference-target policy ranking agreement

Review of `runners/in_distribution/policy_ranking_agreement_preference/` (2026-07).
Scope: the new preference-target variant, diffed against the reference
`runners/in_distribution/policy_ranking_agreement/` and traced through the shared
machinery (`src/beyond_click_sim/tasks/policy_ranking.py`,
`src/beyond_click_sim/tasks/policies.py`, both preference scorers,
`src/beyond_click_sim/evaluation/policy_ranking.py`, the dataset adapters, and the
`preference_prediction` protocol). Setup tests pass (9/9 in
`tests/test_policy_ranking_preference_runner.py`).

Findings are ordered by severity. "Inherited" marks issues that also exist in the
reference interaction-based setup and were copied over, not introduced here.

Findings 7-8 were added by a later review (2026-08) of the LightGCN backend
replacement and are appended after the original review's severity ordering.
Both live in shared machinery (`policies_torch.py`, `policy_tuning.py`), so the
reference interaction setup is affected identically.

## 1. (High) Policies are not trained on preferences — implementation contradicts the intended experiment

**FIXED (2026-08).** Policies now train on the positive-preference target:

- `Policy` gained `positive_feedback_column` (`src/beyond_click_sim/tasks/policies.py`):
  when set, `_fit_frame`/`_positive_rows` keep only rows where that binary column
  is 1, with validation (column present, values in {0,1}) and post-filter
  `fit_row_count_`/`fit_user_count_` recorded for the manifest. Applied in
  `PopularityPolicy`, `ItemKNNPolicy`, `ALSPolicy`, `BPRPolicy`,
  `LightGCNPolicy`; `RandomPolicy` has no training signal and is unchanged.
  Default `None` preserves the interaction setup's behavior exactly.
- Seen-item exclusion still covers **all** consumed items (liked or not):
  `ALSPolicy.recommend`/`BPRPolicy.recommend` now build the seen-mask from the
  full recommend-time frame instead of the (filtered) fitted matrix, and
  `ItemKNNPolicy.recommend` builds user seed sets from positive rows only
  (a disliked item can be in the fitted item space via other users' positives,
  so item-space membership alone was not sufficient — caught by tests).
- The preference task builder constructs the roster via
  `_make_policies(DATASET_TARGET_SOURCE_COLUMNS[dataset_name])`
  (`task_builders.py`), so each trainable policy is fit on
  `target_like_ge4==1` (ml-1m) / `target_played_120==1` (steam) rows.
  `PopularityPolicy` with the filter doubles as the preference-popularity
  floor policy the review found missing.
- The task manifest's `policies` block records `positive_feedback_column`,
  `fit_rows`, `fit_users` per policy
  (`src/beyond_click_sim/tasks/policy_ranking.py`).
- Utility aggregation is restricted to **common-coverage users** (users with
  rows from every policy), because preference-trained CF policies produce no
  recommendations for users with zero positive train rows. Coverage stats
  (`per_policy` n_users/n_rows, `common_users`, `common_rows`) are written to
  `metrics.json`/`manifest.json` as `policy_coverage`
  (`methods/common.py`, all three runners).

**Accepted trade-off (documented, not fixed):** seen low-rated rows are *hard
negatives* that the binary filter drops from training — BPR/LightGCN could in
principle exploit them as explicit negatives, but that would make the training
signal non-uniform across policy families. Simulators still see them via
history. This is a candidate follow-up experiment.

Tests: `tests/test_policies.py::TestPositiveFeedbackFilter`,
`tests/test_policy_ranking_task_builder.py::test_manifest_records_policy_training_signal`,
`tests/test_policy_ranking_preference_runner.py::test_utilities_are_restricted_to_common_coverage_users`.

<details><summary>Original finding</summary>

The stated idea of the setup is to *train recommender policies on user preferences*
and evaluate them on real and simulated preference targets. Only the evaluation
side is implemented. Every policy in the roster is still trained on binary
interaction presence:

- `PolicyRankingTaskBuilder.build` passes raw `split.train` to `policy.fit`
  (`src/beyond_click_sim/tasks/policy_ranking.py:141-146`). The frame contains
  `target_like_ge4` / `target_played_120`, but no policy reads them.
- `ALSPolicy.fit` — ones-matrix over all train rows
  (`src/beyond_click_sim/tasks/policies.py:417-421`).
- `BPRPolicy.fit` — same ones-matrix (`policies.py:519-523`).
- `LightGCNPolicy.fit` — explicitly "Deduplicate: treat interactions as binary"
  (`policies.py:604-624`).
- `ItemKNNPolicy.fit` — ones csr_matrix (`policies.py:292-293`).
- `PopularityPolicy.fit` — raw interaction count `groupby(item).size()`
  (`policies.py:175-179`).

The module docstring
(`runners/in_distribution/policy_ranking_agreement_preference/task_builders.py:1-7`)
is honest about this ("The only difference is `target_source_column`"), which
conflicts with the intended-experiment description.

Why it matters:

- A claim like "simulators can rank policies that optimize preferences" is not
  supported — no policy optimizes the evaluated target. As implemented the
  experiment is "interaction-trained policies ranked by preference utility with
  preference simulators" (a target-shift ranking experiment).
- The roster lacks the natural floor for a preference target: a
  preference-popularity policy (recommend train items with the most
  preference-positive interactions). The included `PopularityPolicy`
  (interaction count) is systematically handicapped under
  `target_like_ge4` / `target_played_120`, which distorts the real policy
  ranking the agreement metrics are computed against.
- Provenance gap: `manifest["policies"]` records class/name/k/seed but not the
  training signal (`src/beyond_click_sim/tasks/policy_ranking.py:277-285`). A
  reader of a preference-task manifest could reasonably assume
  preference-trained policies.

</details>

## 2. (Medium) Real ("ground-truth") policy utilities are computed on the simulator-dependent scored subset

In both LLM methods, real utilities are computed only on rows the simulator
successfully scored:

- `methods/llm_yes_no.py:639-653` — `valid = scores.notna()`, then
  `compute_policy_utilities(X_scored, y_scored, scores_scored)` with the
  *filtered* `y_scored`.
- `methods/agent4rec_yes_no.py:341-354` — same pattern.
- `methods/popularity.py:58` — uses the full test set (PopularityScorer never
  yields NaN).

Consequences:

- The "real" ranking inside one run's `metrics.json` is not the task's
  ground-truth policy ranking; it is the ground-truth ranking restricted to
  rows this simulator could parse. Parse failures are not MCAR (they correlate
  with response length / candidate count / model verbosity), so simulator
  behavior contaminates the reference ranking.
- Two methods' Kendall tau values are computed against *different* real
  rankings, weakening cross-method comparison.
- There is no failure-penalized variant (unlike interaction-prediction and
  cold-start runners, which publish `test_failure_as_negative` /
  `test_failure_as_zero_group`; see root `issues.md` #1). The run-level error
  budget (`max_error_rate`) bounds but does not eliminate the effect.

Fix: compute `real_utilities` once on the full (smoke-capped) test set,
independent of scores — it is simulator-independent by construction — and/or add
a `test_failure_as_zero_utility` agreement variant where groups with missing
scores contribute 0 to the simulated mean.

## 3. (Low) `popularity_scorer` mixes two popularity notions

`PopularityScorer` sums the task target per item
(`src/beyond_click_sim/scorers/popularity.py:29`), so under this protocol it is
a preference-trained scorer (train preference-count per item). But it scores
recommendation lists that, for `PopularityPolicy`, were produced by
interaction-count ranking. The "popularity" row in the agreement table is
therefore a preference-popularity simulator evaluated on
interaction-popularity recommendations. Disclosed in the runner docstring
(`methods/popularity.py:36-38`), but it deserves a sentence in the paper and
reinforces the missing preference-popularity policy from Finding 1.

## 4. (Low, inherited) Agent4Rec Steam candidates get raw JSON strings

`methods/agent4rec_yes_no.py:305-318` never passes `json_list_columns` — the
reference interaction variant omits it too, but `methods/llm_yes_no.py` passes
`("item_genres_json", "item_tags_json")` for Steam, and so does the
interaction-prediction Agent4Rec runner. Result: Steam Agent4Rec prompts show
`genres: ["Action", "RPG"]` as a raw JSON string under a "genres" label, while
the plain LLM scorer shows `genres: Action, RPG`. Prompt-fidelity inconsistency
across methods within the same results table.

## 5. (Low, inherited) Itemwise smoke cap keeps 4 items per policy, not 4 users

In itemwise mode `_llm_group_ = user::policy::item`
(`methods/llm_yes_no.py:521-526`), so `max_llm_groups=25` gives
`per_policy = 25 // 6 = 4` *single-item* groups per policy
(`methods/llm_yes_no.py:534-544`) — potentially 4 items from 4 different users,
versus ~4 users × 10 items in batch mode. Smoke-only, but the two modes' smoke
runs differ ~10x in scale, and itemwise smoke covers very few users for a
per-user-utility experiment.

## 6. (Low, inherited) Manifest mislabels the grouping in itemwise mode

`manifest["scorer"]["candidate_group"]` is hardcoded `"user_id::policy"`
(`methods/llm_yes_no.py:692`) even when `scoring="itemwise"` actually groups by
`user::policy::item`. Provenance nit.

## 7. (High, shared) LightGCN backend replaced; its search space and defaults are unvalidated

The roster now fits `LightGCNTorchPolicy`
(`src/beyond_click_sim/tasks/policies_torch.py:38`) instead of the pure-NumPy
`LightGCNPolicy`; see `POLICY_ROSTER`
(`src/beyond_click_sim/tasks/policy_tuning.py:59`). The NumPy class is retained
as a reference implementation and fallback.

Motivation for the swap: the NumPy class applies **one full-batch Adam update
per epoch** (`batch_size` there only chunks gradient accumulation), whereas
He et al. (SIGIR 2020) Section 3.3 specify Adam "in a mini-batch manner" and
Section 4.1.2 gives batch size 1024-2048 with ~1000 epochs. On the ml-1m inner
split that is ~200 parameter updates against the paper's ~292,000 — a different
optimizer, not a slower version of the same one.

The consequence is already measurable. On the ml-1m inner validation split
(`target_interact`, split_seed 0), the NumPy LightGCN reaches ndcg@10 **0.498**
at the best configuration its old search box could reach, but **0.538** at 600
epochs. Its previously reported position in the ground-truth policy ranking was
therefore set by an undertrained optimizer, and 0.538 puts it level with
ItemKNN (0.536) rather than clearly below it.

Current state of the replacement:

- **Structurally verified.** Propagation matches the NumPy `A_norm` chain to
  ~1e-7 on both dispatch paths; the BPR loss matches the author-endorsed
  reference (`gusye1234/LightGCN-PyTorch`, cited as footnote 7 of the paper)
  exactly, given `lambda_reg = regularization / 2` and a **non-unique**
  `node_id` — passing `node_id=None` would regularise the whole embedding table
  on every mini-batch instead of the batch triples.
- **Not quality-validated.** Best CPU run so far is 0.505 at 20 epochs, still
  climbing, versus the 0.538 NumPy reference. Parity is untested. **This half
  of the finding remains open.**
- **Search space — FIXED (2026-08).** `SEARCH_SPACES["LightGCNTorchPolicy"]`
  previously carried the NumPy-era box (`n_layers` {2,3}, `learning_rate`
  log[0.001, 0.01]) and did not search `iterations`, so the class default of
  1000 epochs applied untested. It now searches `n_layers` {2,3} x
  `regularization` {1e-4, 1e-3, 1e-2} x `iterations` {300, 1000} = 12 configs,
  with lr fixed at 0.001. The box follows He et al. Sec 4.1.2 and RecBole's
  published LightGCN tuning range, narrowed for budget; `iterations` is
  searched in place of the early stopping both references use, since
  `Policy.fit` receives no validation data. Per-parameter rationale and the
  deviations from both references are recorded inline in `SEARCH_SPACES`.

Why it matters: LightGCN is one of six policies whose **real** ranking is the
ground truth the agreement metrics are measured against. Until the new backend
is tuned and shown to at least match the NumPy reference, any Kendall tau
computed on a roster containing it is agreement with an arbitrary policy
position, not with a defensible one.

Remaining fix: benchmark one fit on the GPU pod (12 configs x 12 jobs = 144
full-length fits), then gate on beating **0.538** ndcg@10 on the ml-1m inner
split before reporting any LightGCN number.

Stale references updated alongside: `in_distribution_scenarios.md` (policy
table now lists both backends and marks the NumPy one as reference-only; the
L2-regularization note now covers the mean-vs-sum loss scaling difference) and
`docs/policy_tuning.md` (search-space table, cost note, device note). This
file's Finding 1 still names `LightGCNPolicy` in the `positive_feedback_column`
list; that statement remains true of the NumPy class and the torch class
implements the same filtering, so the closed finding was left intact.

## 8. (Medium, shared) LightGCN results are not reproducible across runs or devices

`LightGCNTorchPolicy` is nondeterministic. Two identical fits (same seed, same
data, CPU, full ml-1m) differ by **1.07e-06** on embeddings after 3 epochs. The
cause is multithreaded reduction order, not seeding: on a 400-user subset,
`torch.set_num_threads(1)` gives bit-identical output (0.0) while 8 threads
gives 8.94e-08.

GPU does not avoid this. In torch 2.13 the following are documented
nondeterministic on CUDA, and the model uses all of them: `index_select`
backward (the `emb[node_id]` and `out[edge_label_index]` gathers), `index_add`
and `scatter_add_` (their backward), and `bmm` on sparse-dense CUDA tensors
(the CSR propagation path). On CUDA the mechanism is nondeterministic atomics,
generally a larger effect than CPU thread ordering. All four fall in the
category that `torch.use_deterministic_algorithms(True)` **makes deterministic**
rather than the category that raises, so the flag is usable for this model.

PyTorch additionally documents that results are **not** reproducible between
CPU and GPU executions even with identical seeds, and are guaranteed only "for
a specific platform, device, and PyTorch release". MPS is a third distinct
result: `aten::_to_sparse_csr` is unimplemented on the `SparseMPS` backend, so
this class dispatches dense `edge_index` + gather/scatter there instead of
sparse `spmm`.

Why it matters: `AGENTS.md` requires reproducible experiment outputs, and the
ground-truth policy ranking depends on LightGCN's fitted embeddings. Divergence
compounds over ~1000 epochs, and LightGCN and ItemKNN already sit within 0.002
ndcg of each other on ml-1m — well inside the range a nondeterministic fit
could reorder.

Partial mitigation already in place: `implementation` and `resolved_device` are
recorded per policy in the task manifest via `_policy_hyperparameters`
(`src/beyond_click_sim/tasks/policy_ranking.py:25`), so the backend and device
are auditable after the fact.

Fix: add a `deterministic` constructor flag that sets
`torch.use_deterministic_algorithms(True)` and is itself recorded in the
manifest (preferred over setting the global flag in a runner, which would
silently change behaviour for everything else in the process). Report device
alongside seed for any LightGCN number, and do not compare LightGCN values
across devices.

## Caveats to state in the paper (not bugs)

- Batch prompts present candidates C1...C10 in the policy's own rank order (test
  rows are concatenated in rank order from `Policy.recommend`). Mean-utility
  aggregation cancels pure position bias, but list-context effects (contrast,
  anchoring) can still shift a policy's yes-rate. The itemwise variant is the
  right control; good that both exist.
- With `batch_dedup_mode="ordered"`, policies emitting identical ordered lists
  share one LLM call and therefore get identical simulated scores for those
  users by construction. Symmetric with real utilities (same items => same
  targets), so no bias, but it couples those policies' simulated rankings
  deterministically.

## Verified as correct (evidence checked)

- Target definitions match ground truth and the preference-prediction protocol:
  `target_like_ge4` = `rating >= 4`
  (`src/beyond_click_sim/data/adapters/movielens.py:282`),
  `target_played_120` = `playtime_forever >= 120`
  (`src/beyond_click_sim/data/adapters/steam.py:272`);
  `DATASET_TARGET_SOURCE_COLUMNS` and `TARGET_DESCRIPTIONS` are byte-identical
  to `preference_prediction`, so the cross-experiment comparability claim holds.
- Task construction is identical to the reference except `target_source_column`
  (diff-verified): same filter, splitter, seeds (0, 1, 2), policy roster, k=10,
  eval_users=1000 sampler.
- No target leakage into test rows: `history_context_columns` masked to NA in
  test (`src/beyond_click_sim/tasks/policy_ranking.py:207-209`); test targets
  looked up only from held-out interactions with a 0 default for unobserved
  pairs (lines 132-135, 155-161); item enrichment is train-only (line 212).
  Splitting is per-user (`RandomFractionSplitter` default
  `group_column="user_id"`); with >= 10 interactions at 80/20 every eval user
  has train rows, so Agent4Rec's "no train rows" guard cannot fire in practice.
- The elicited question matches the graded label in both prompt styles,
  including the regression test for the inherited
  itemwise-interaction-prompt bug
  (`tests/test_policy_ranking_preference_runner.py:195-215`).
- `policy_recommendation_metrics` use the preference target, so held-out policy
  metrics and `real_utilities` are the same quantity (micro mean over rec
  rows) — internally consistent.
- Dedup logic is correct in both modes (itemwise by (user, item); batch by
  (user, item-tuple)), including score join-back; smoke caps are deterministic
  and proportional per policy.
- Taste-cache sharing with the interaction policy-ranking Agent4Rec variant is
  valid: taste generation is target-independent and train histories are
  byte-identical across the two setups, so cache hits return the correct taste.
- `policy_ranking_agreement_metrics` is sound: tau-b / average-rank tie
  handling, K=2 -> None with warning, K<2 raises.
- Task/method namespacing prevents preference and interaction outputs from
  colliding (registry disjointness is tested).
