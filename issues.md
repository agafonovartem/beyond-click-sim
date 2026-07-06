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

## 8. Agent4Rec yes/no scorer lacks Summary and has incomplete profile ablations

The current `Agent4RecYesNoScorer` implementation supports deterministic Agent4Rec social traits
(`activity`, `conformity`, `diversity`) from the train split and a cached LLM-generated `taste`
profile. It now has dataset-specific prompt/profile configuration for ML-1M and Steam, but the
planned Agent4Rec profile ablations are still incomplete:

- `traits` only — currently implemented;
- `taste` only — available through the generic runner path, but not exposed as a named default method;
- `traits + taste` — currently implemented for the Qwen + `gpt-4o-mini` taste runner.

The ML-1M scorer intentionally omits Agent4Rec's `Summary` field in `##recommended list##`. The
released Agent4Rec alignment code shows title, `History ratings`, and `Summary`, but `Summary`
comes from `movies_augmentation.csv`, an undocumented external augmentation rather than a
reproducible MovieLens field. Our current ML-1M prompt uses title, train-only `History ratings`
(`item_rating_mean`), and genres; this is a cleaner but not exact reproduction of their visible
item card. The Steam prompt uses game title, genres, tags, and playtime-derived profiles rather
than MovieLens-specific fields.

**Fix:** expose a named `taste`-only ablation if needed, and keep ML-1M `Summary` out unless we add
a reproducible summary artifact with clear provenance. Any cross-dataset Agent4Rec-style reporting
should state the dataset-specific visible item fields and profile prompt version explicitly.

## 9. Agent4Rec vs history LLM comparison currently mixes multiple axes

Current `agent4rec_yes_no` and `llm_yes_no` runs are not a controlled comparison of "profile
module versus history prompt". They differ in several places at once: Agent4Rec uses social-trait
profile text instead of per-user history, different generation settings, a larger token budget, and
an item card that includes train-only `History ratings` (`item_rating_mean`). Any metric delta
between these methods therefore cannot be attributed cleanly to Agent4Rec-style profiles.

**Fix:** compare Agent4Rec against matched baselines: same temperature/token budget, same visible
item-card fields (including an `llm_yes_no_with_item_stats` baseline), and, if needed, a no-history
LLM variant. Record the controlled comparison axis explicitly in the manifest.
