# Known Issues

Tracked defects in the **current** code, i.e. the in-distribution interaction-prediction
pipeline (popularity + LLM yes/no scorers). Unimplemented future work — preference &
regression tasks, ranking metrics, additional negative samplers, stronger classic
baselines, temporal/stratified splitters, policy-ranking, OOD, behavioral extrapolation,
memorization — is not listed here; it lives in the design notes (`architecture_note.md`,
`in_distribution_scenarios.md`, `notes.md`).

## 1. `MAX_LLM_ERRORS` aborts the whole run instead of skipping

`_score_groups` stops the entire scoring loop once the error count reaches `MAX_LLM_ERRORS`
(= 3): `runners/in_distribution/interaction_prediction/methods/llm_yes_no.py:37,271,293-296`.
Errored groups are already skipped and counted (their scores stay NaN and are excluded from
metrics); only the abort is wrong. On a full run (thousands of groups) a model will exceed 3
malformed/parse-failed outputs early, and the run aborts, discarding all work.

**Fix:** make the budget large or a fraction of the total group count, and skip-and-continue
instead of aborting. Keep reporting scored-vs-requested coverage.

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

## 4. Headline metric averages over candidate groups, not users (heavy-user weighting)

The reported metric (`test.macro_by_group.f1`) and the popularity threshold selection both
average every candidate group **equally**: `grouped_binary_classification_metrics`
(`src/beyond_click_sim/evaluation/binary.py:45-120`) and `find_best_group_threshold`
(`runners/in_distribution/interaction_prediction/methods/popularity.py:59-64`).

But a candidate group is a prompt-size **chunk**, not a user.
`CappedUserInteractionCandidateSampler` splits a user's held-out positives into
`g_u = ceil(P_u / max_positive_items)` groups, where
`max_positive_items = total_items // (negative_ratio + 1)`
(`src/beyond_click_sim/tasks/samplers.py:232,256-258`). So `g_u` grows with the user's
activity **and** with `negative_ratio`. Averaging equally over groups therefore weights each
user by their chunk count, not equally.

Concretely, with per-user group means `mean_u` the current metric is
`M_group = Σ_u (g_u / Σ g) · mean_u` (weight ∝ group count), whereas a per-user metric is
`M_user = Σ_u (1 / U) · mean_u` (equal weight). They coincide only if every user has the
same `g_u`. So the headline silently up-weights heavy users by an amount that also depends
on the `negative_ratio` sweep axis — exactly the users and the knob we are studying.

**Status:** fixed in code by reporting `macro_by_user_group_mean` and using the same
aggregation for popularity threshold selection. Old outputs with `main_metric =
test.macro_by_group.f1` still need recomputation or migration; fixed-prediction LLM runs
can be migrated from `predictions.parquet`, while popularity runs should be rerun because
the selected threshold can change.

## 5. Note: per-group F1 is quantized on small groups

Not a defect, a caveat to keep in mind when reading #4. Per-group F1 is computed from very
few positives once `negative_ratio` is large: at m=9 a group has 2 positives, so per-group
recall ∈ {0, 0.5, 1}; at m=19, 1 positive, recall ∈ {0, 1}
(`src/beyond_click_sim/tasks/samplers.py:232`). Averaging these coarse per-group F1 values
is a high-variance estimator that gets coarser as `negative_ratio` grows.

This quantization already existed in the old group-macro metric. The two-level fix in #4
re-weights users but **reuses the same per-group F1**, so it neither adds nor removes this
noise. This is intentional for the current protocol: the candidate group is the LLM prompt
context, and the fixed headline now changes only the user weighting, not the within-prompt
decision unit.
