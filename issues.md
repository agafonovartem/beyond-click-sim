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
