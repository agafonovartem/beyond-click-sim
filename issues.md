# Known Issues

Tracked defects in the **current** code, especially the in-distribution interaction-prediction
pipeline (popularity + LLM yes/no scorers, pointwise and raw-score ranking) and the first
rating-regression protocol, including protocol-validity and provenance gaps that affect how current
outputs should be interpreted. Unimplemented future work — preference tasks, additional regression
targets/methods, additional negative samplers, stronger classic baselines, temporal/stratified
splitters, direct re-ranking prompts, policy-ranking, OOD, behavioral extrapolation, memorization
— is not listed here; it lives in the design notes
(`architecture_note.md`, `in_distribution_scenarios.md`, `notes.md`).

## 1. LLM metrics skip failed candidate groups / rows

LLM yes/no runs retry each candidate group up to `MAX_LLM_ATTEMPTS` times. LLM regression
runs retry each requested row with the same policy. If all attempts fail, the group/row remains
in `predictions.parquet` with null `score` / `prediction` where applicable, while reported
metrics are computed only on successfully parsed groups/rows.

This means headline metrics are conditional on successful parsing. Coverage is reported via
`llm_errors`, `scored_rows`, `requested_rows`, regression `coverage`, and requested/scored
candidate-group summaries where applicable, but the metric value itself is not failure-penalized.

## 2. `candidate_group` ids collide between val and test splits

`CappedUserInteractionCandidateSampler._candidate_group_id` builds the group id from only
the user and chunk position — `candidate:user:{user_id}:chunk:{chunk_position}`
(`src/beyond_click_sim/tasks/samplers.py:375-378`). The id does not encode the split, so
user `U`'s chunk `0` gets the **same** `candidate_group` value in both `task.val` and
`task.test`.

Within-split metrics are unaffected: both methods group val and test separately. The risk
is downstream. The popularity runner writes a single `predictions.parquet` containing both
splits, distinguished only by a `split` column while the `candidate_group` values are
identical across splits (`runners/in_distribution/interaction_prediction/methods/popularity.py:104-123`).
Any later analysis that does `groupby("candidate_group")` without also grouping on `split`
will silently merge a val group with its test counterpart, mixing splits in per-group
aggregates.

**Fix:** namespace the group id by split (e.g. prefix with the split name in the builder),
or document that `candidate_group` is unique only within a split and require analyses to
group by `(split, candidate_group)`.

## 3. Task registry key disagrees with `task.name` (provenance)

The task registry key — e.g. `ml-1m_cap20_eval_users1000_m1_seed0` — is what the CLI takes
and what drives the output directory name
(`runners/in_distribution/interaction_prediction/task_builders.py:111-132`,
`runners/in_distribution/interaction_prediction/run.py:24,95-97`). But the builder sets a
different `task.name` — `ml-1m_interaction_cap20_eval_users1000_m1_seed0` (note the extra
`interaction`) — and that name is what gets recorded inside `metrics.json` / `manifest.json`
(`task_builders.py:45`). So the output folder name and the `task` field stored inside it
disagree, and there are effectively two identifiers for the same task.

This is harmless to results but hurts provenance: matching a metrics file back to the run
folder or the CLI invocation requires knowing about the rename. It predates the eval1000
change (the full-scale builders have the same split) but eval1000 adds a third naming
variant.

**Fix:** make the registry key and `task.name` identical (pick one convention), or record
both the registry key and the builder name in the manifest.

## 4. Ranking headline `ndcg@5`/`hit_rate@5` is not directly comparable across `m`

The ranking headline is `test.macro_by_user_group_mean.ndcg@5` with `RANKING_KS = (1,3,5,10)`
(`runners/in_distribution/interaction_prediction/metrics.py:8,10`). In cap20 construction,
`max_positive_items = total_items // (m + 1)`, so full candidate groups usually contain up to
20 items for `m=1`, `m=3`, `m=9`, and `m=19`, while `m=2` has a maximum full-group size
of 18. Tail groups can be smaller when a user has too few held-out positives.

The larger comparability problem is not just group size, but the joint change in group-size
distribution and positive prevalence. At `m=1`, a full group has 10 positives; at `m=19`, a full
group has one positive. Therefore HR@5/NDCG@5 have different random-ranking baselines and
different difficulty across `m`, even when both groups have 20 candidates. For tail groups with
`group_size <= k`, `k_eff = min(k, group_size)` (`src/beyond_click_sim/evaluation/ranking.py:135`)
and hit-rate returns `1.0` whenever the group has a positive (`ranking.py:163-164`), so
`groups_with_size_lte@k` must be checked before interpreting `@k` metrics.

**Fix:** do not pool headline ranking metrics across different `m` values. Report each `m`
separately with `groups_with_size_lte@k`, positive prevalence, `groups_with_score_ties_fraction`,
and, ideally, a random-ranking baseline. Consider `@1` as the most stable cross-`m` diagnostic,
while keeping `@1/3/5/10` for within-setup analysis.

## 5. Ranking metric aggregation is O(candidate_groups × rows) (performance / repro-time, not correctness)

`_per_group_ranking_metrics` loops `for group_code in range(...)` and rescans the full code array
with `np.flatnonzero(group_codes == group_code)` per group
(`src/beyond_click_sim/evaluation/ranking.py:96-97`); the binary path is vectorized
(`src/beyond_click_sim/evaluation/binary.py:62-77`). Results are correct, but this is quadratic in
group count and on a full (non-eval1000) task with tens or hundreds of thousands of groups adds
avoidable minutes. Speedup only — no effect on metric values.

**Fix:** sort once by group code and slice contiguous blocks (or `np.split` on sorted codes)
instead of a per-group mask scan.

## 6. Rating-regression mean baseline predicts impossible user responses

The current `mean_regressor` baseline for MovieLens rating prediction fits the train mean and
returns that continuous value for every row. This is a legitimate MAE/RMSE regression baseline,
but it is not a literal user-response simulator: MovieLens users can only emit discrete ratings
1-5, while the train mean is usually non-integer (for example around `3.58` in the current
ML-1M runs).

This matters when comparing against an LLM asked to simulate a user's rating response. If the
LLM is constrained to output an integer rating, the continuous mean baseline has an information
and output-space advantage; if the LLM is allowed to output fractional ratings, it is no longer
strictly simulating the observed response format.

**Mitigation:** use the discrete `ModeRegressor` / most-common-rating baseline as the primary
simulator-style constant baseline and report `MeanRegressor` as a continuous MAE/RMSE diagnostic.
For LLM rating simulation, require a strict integer in the valid rating set rather than clamping
or accepting arbitrary continuous values.

## 7. LLM "use only provided information" prompts do not remove model priors

The current LLM prompt stance tells the model to use only the provided user history and item
metadata. This is a useful protocol instruction because it discourages hidden recommender-like
behavior and keeps prompts comparable across tasks. However, an LLM cannot actually disable its
pretraining prior or possible memorized knowledge about popular movies/games. In this project,
that prior is also part of the scientific question: a user simulator may be useful precisely
because it brings broad semantic/world knowledge, but the same prior can create unfairness or
dataset memorization artifacts.

This affects both yes/no interaction prediction and rating regression. A prompt instruction alone
should not be interpreted as proof that the model used only the visible metadata.

**Fix:** treat this as an interpretation caveat in result tables, and evaluate memorization /
metadata-visibility variants separately. When comparing to classic baselines, state explicitly
whether LLM world priors are considered allowed simulator signal or a leakage risk for that
experiment.

## 8. Candidate rows lack train-only item quality / popularity aggregate metadata

Current candidate rows expose raw item metadata such as title/genre/year where available, but they
do not expose reusable train-derived item aggregates such as `rating_count`, `mean_rating`,
`positive_rate`, or popularity quantiles. The only implemented population-level item signal is the
separate `PopularityScorer`, which learns a train target count internally and returns a score; that
signal is not materialized as candidate metadata for LLM prompts, tabular baselines, stratified
analysis, or manifest-visible metadata-visibility variants.

This creates an information-regime gap relative to the simulator literature we are comparing
against. AgentRecBench-style prompts include item-side fields such as average rating and review
count, Agent4Rec item profiles include quality and popularity, and SimUSER uses aggregated item
ratings / review-count visibility in parts of its setup. Without an explicit item-aggregate feature
layer, our "metadata visibility" setting is underspecified: it is unclear whether a simulator was
denied useful population context by design, or whether the feature simply has not been implemented.

**Fix:** add a train-only item aggregate feature builder and make visibility explicit per task/run.
At minimum, compute item `rating_count`, `mean_rating`, `rating_std`, `positive_rate`, and
popularity rank/quantile from the training split only, join them onto validation/test candidate rows,
and record whether these fields were hidden, used only for analysis, or exposed to a scorer/LLM
prompt. Do not compute these aggregates from validation/test interactions or the full dataset unless
the task intentionally defines a different "platform-visible global statistics" regime and records it
in the manifest.
