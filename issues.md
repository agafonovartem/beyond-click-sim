# Known Issues

Tracked defects in the **current** code, especially the in-distribution interaction-prediction
pipeline (popularity + LLM yes/no scorers, pointwise and raw-score ranking) and the first
rating-regression protocol, including protocol-validity and provenance gaps that affect how current
outputs should be interpreted. Unimplemented future work — additional regression targets/methods,
additional negative samplers, stronger classic baselines, temporal/stratified splitters, OOD,
behavioral extrapolation, memorization
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

**Update (2026-07):** the interaction-prediction and cold-start `llm_yes_no` runners, and the
cold-start `agent4rec_yes_no` runner, now also publish failure-penalized variants alongside the
parsed-only headline:
`test_failure_as_negative` (pointwise; failed rows counted as `no`),
`test_failure_as_zero_group` (ranking; groups with any missing score contribute zero utility),
and a row-level `coverage` summary. The headline `main_metric` remains the parsed-only value.

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

**Correction (2026-07, cold-start review):** an earlier version of this entry claimed the
same within-task collision for cold start. That was wrong: `ColdUserHoldoutSplitter` makes
val-cold and test-cold users disjoint, so no id can appear in both splits of one cold-start
parquet. The real cold-start risk was **cross-run** concatenation (same id across k/m/seed
runs). The redesigned cold-start pipeline addresses this: `candidate_group` is unique within
`(dataset, k, m, negative_strategy, sampling_seed, split)`, and cold-start runners write
`k`, `negative_ratio`, `sampling_seed`, `negative_strategy` as constant columns into
`predictions.parquet` (`cold_start_metadata` in
`runners/in_distribution/interaction_prediction/methods/common.py`). The within-task val/test
collision described above remains open for `interaction_prediction` only.

**Fix:** namespace the group id by split (e.g. prefix with the split name in the builder),
or document that `candidate_group` is unique only within a split and require analyses to
group by `(split, candidate_group)`.

## 3. Task registry key disagrees with `task.name` (provenance)

The task registry key — e.g. `ml-1m_cap20_eval_users1000_cg5_m1_seed0` — is what the CLI takes
and what drives the output directory name
(`runners/in_distribution/interaction_prediction/task_builders.py:111-132`,
`runners/in_distribution/interaction_prediction/run.py:24,95-97`). But the builder sets a
different `task.name` — `ml-1m_interaction_cap20_eval_users1000_cg5_m1_seed0` (note the extra
`interaction`) — and that name is what gets recorded inside `metrics.json` / `manifest.json`
(`task_builders.py:45`). So the output folder name and the `task` field stored inside it
disagree, and there are effectively two identifiers for the same task.

This is harmless to results but hurts provenance: matching a metrics file back to the run
folder or the CLI invocation requires knowing about the rename. It predates the eval1000
change (the full-scale builders have the same split) but reduced eval1000/cg5 adds another naming
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

**Update (2026-07, cold-start redesign):** the redesigned cold-start pipeline removes the
population confound (evaluated users and target frames are now byte-identical across m;
`target_frame_sha256` in the manifest). The metric-baseline caveat remains and is sharper
there: cold-start groups are 1 positive + m negatives, so group size = 1+m and @5 is
degenerate for every group when m < 4. Use @1 as the cross-m headline (documented in
`cold_start_spec.md`, Metrics Caveats). This entry remains open for the
interaction_prediction cap20 tasks described above.

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

## 6. LLM prompts may over-constrain useful model prior

The current LLM system prompts say `Use only the provided history and candidate information.`
This wording is not well aligned with the main research question. In this project, the LLM's
semantic/world prior is part of what we may want to evaluate as simulator signal, not something
that is automatically a defect. The prompt should prevent obvious protocol cheating, such as
assuming access to hidden held-out labels or future interactions, but it should not imply that
general model knowledge is disabled or forbidden by default.

The practical risk is prompt wording: we may unnecessarily discourage useful semantic reasoning
and later describe the protocol as if the model used only visible metadata. This is not a separate
experiment axis or a broad memorization caveat; it is a prompt-design cleanup.

**Fix:** remove or soften the restrictive sentence in the yes/no and regression system prompts.
Prefer neutral wording such as "Given the user's observed history and candidate item information,
predict the user's response" plus the existing output-format constraints. Add only a narrow
anti-cheating instruction if needed, e.g. do not assume access to hidden held-out labels, future
interactions, or evaluation answers.

## 7. Visible item-card metadata is not aligned clearly with competitor protocols

Our default ML-1M prompts expose basic item metadata such as title/genre/year. Explicit item-stats
variants additionally expose train-only `item_rating_mean` and `item_rating_count`. The problem is
not that every possible aggregate feature is missing; the problem is that the visible item-card
metadata regime is not stated clearly enough relative to the simulator papers we compare against.

Agent4Rec presents recommended movies with item-profile information. In the paper, item profiles
include quality, popularity, genre, and summary; in the released MovieLens simulation code, the
runtime recommended-item string contains the movie title, `History ratings`, and `Summary`
(`repos/Agent4Rec/simulation/arena.py`). The summary comes from `movies_augmentation.csv`, while
the historical rating is computed from MovieLens ratings in `3_get_movie_detail.py`; this appears to
use the full raw ratings file, not a train-only split.

AgentRecBench is not a MovieLens setup. Its Amazon/Yelp/Goodreads agents use platform-specific
item fields such as stars, review counts, average ratings, rating counts, descriptions, attributes,
authors, publication years, and similar books. Therefore fields like `review_count`,
`rating_number`, or `ratings_count` should be treated as dataset/platform metadata, not as a
universal recommender feature we must blindly reproduce for MovieLens.

SimUSER also treats item-side cues as part of the simulated environment: it defines aggregated item
rating and genre/category, and separately studies environment interventions such as showing the
number of reviews, positive/negative reviews, and thumbnails/posters. This supports treating
metadata visibility as an interface condition rather than a generic model-feature checklist.

For our ML-1M experiments, `item_rating_mean` is the closest train-only analog of Agent4Rec's
historical rating / platform average rating, while `item_rating_count` is a count/popularity signal
closer to review-count visibility. These fields are reasonable visible item-card variants, but their
visibility must be named and interpreted as such. Item-stats gains should not be described as
generic LLM improvement unless the table states that the prompt exposed average rating/count.

**Fix:** define the main ML-1M visible item-card protocol before final reporting: e.g.
`title+genres` versus `title+genres+average_rating+rating_count`, and optionally whether an
Agent4Rec-style summary condition is in scope. Keep using train-only aggregates for our protocol
unless a separate "public platform statistics" condition is intentionally defined. For rating
regression with item stats, compare against the matched train-only item baselines
(`item_mean_regressor` / `item_mode_regressor`) before claiming user-conditioned reasoning beyond
visible item-quality metadata.

## 8. Agent4Rec yes/no named ablations are incomplete

The current `Agent4RecYesNoScorer` implementation supports deterministic Agent4Rec social traits
(`activity`, `conformity`, `diversity`) from the train split and a cached LLM-generated `taste`
profile. It now has dataset-specific prompt/profile configuration for ML-1M and Steam, but the
planned Agent4Rec profile ablations are still incomplete:

- `traits` only — currently implemented;
- `taste` only — available through the generic runner path, but not exposed as a named default method;
- `traits + taste` — currently implemented for the Qwen + `gpt-4o-mini` taste runner.

The released Agent4Rec alignment code shows title, `History ratings`, and `Summary` in
`##recommended list##`. Its `movies_augmentation.csv` is vendored with source commits, license,
and SHA256 provenance, and MovieLens canonicalization exposes it as `item_summary`. The generic
yes/no runner supports `none`, taste-`profile`, `candidate`, and `both` summary usage with strict
task provenance checks. Canonical named ML-1M Agent4Rec methods now use candidate summaries when
the task carries canonical enrichment; explicit canonical opt-out and Steam remain no-summary.
Taste-only and the remaining summary placements are not yet uniformly available as named registry
entries. The Steam prompt uses game title, genres, tags, and playtime-derived profiles rather than
MovieLens-specific fields.

**Fix:** expose named ML-1M summary and `taste`-only methods if those ablations are selected for
interaction experiments. Any cross-dataset Agent4Rec-style reporting should state the
dataset-specific visible item fields and profile prompt version explicitly.

## 9. Agent4Rec vs history LLM comparison currently mixes multiple axes

Current `agent4rec_yes_no` and `llm_yes_no` runs are not a controlled comparison of "profile
module versus history prompt". They differ in several places at once: Agent4Rec uses social-trait
profile text instead of per-user history, different generation settings, a larger token budget, and
an item card that includes train-only `History ratings` (`item_rating_mean`). Any metric delta
between these methods therefore cannot be attributed cleanly to Agent4Rec-style profiles.

**Fix:** compare Agent4Rec against matched baselines: same temperature/token budget, same visible
item-card fields (including an `llm_yes_no_with_item_stats` baseline), and, if needed, a no-history
LLM variant. Record the controlled comparison axis explicitly in the manifest.

## 10. ItemKNN zero-score floor for items unseen in warm train (cold start)

`ColdItemKNNScorer.score()` initializes all scores to 0.0 and only fills candidates present in
the warm-train item index (`src/beyond_click_sim/scorers/item_knn.py:101,123-127`); profile items
absent from warm train are silently skipped (`item_knn.py:112-114`).

**Update (2026-07, strict warm-item coverage):** `ColdStartTaskBuilder` now enforces that
history, targets, and sampled negatives use only warm-train items
(`warm_item_coverage.policy = "strict"` in the task manifest). The previous
"candidate item absent from warm train" case should no longer arise for cold-start
candidate rows. Scores can still be exactly 0.0 for warm items with no usable neighbor
mass, so zero-score rate diagnostics remain useful when interpreting KNN-vs-LLM gaps.

**Fix/diagnostic:** report the fraction of scored candidates with score exactly 0.0 per run;
state remaining zero-score mass explicitly when interpreting KNN-vs-LLM gaps.

## 11. LLM "ranking" metrics are computed from binary 0/1 scores

All LLM yes/no methods emit scores in {0.0, 1.0}, so `metrics_ranking.json` for LLM methods ranks
groups that consist entirely of ties. With `RANKING_TIE_POLICY = "average"`
(`runners/in_distribution/interaction_prediction/metrics.py:11`) the reported NDCG/HR is the
expected value under uniform random tie-breaking — principled, but a deterministic function of
the per-group yes/no pattern, carrying no graded discrimination. `groups_with_score_ties_fraction`
is ~1.0 for every LLM run. In the redesigned cold start (1 positive + m negatives per group) this
reduces to "did the model say yes to the target and no to everything else."

Any pointwise-vs-ranking claim about these **yes/no-derived** LLM ranking runs must state that the
protocol is binary-score-with-expected-tie-breaking.

**Update (2026-07, cold-start listwise):** cold-start now has direct history-LLM and Agent4Rec
preference-listwise families (`llm_preference_listwise_ranking_*` and
`agent4rec_preference_listwise_ranking_*`) that elicit a complete within-group ranking and remove
binary-score ties by construction. Those are the appropriate protocols for ranking-focused
cold-start claims, subject to their forced-complete-order and candidate-order caveats. The warning
in this issue remains active for `llm_yes_no` and `agent4rec_yes_no` ranking artifacts, which are
still derived from binary scores.

**Fix/diagnostic:** report `groups_with_score_ties_fraction` alongside LLM ranking numbers; if
ranking agreement for LLMs becomes a paper claim, add a graded scoring variant (e.g. 1–5 rating
or pairwise/listwise ranking prompts) instead of deriving ranks from yes/no.
