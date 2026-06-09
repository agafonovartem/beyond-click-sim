# Known Issues

Tracked defects in the **current** code, i.e. the in-distribution interaction-prediction
pipeline (popularity + LLM yes/no scorers). Unimplemented future work — preference &
regression tasks, ranking metrics, additional negative samplers, stronger classic
baselines, temporal/stratified splitters, policy-ranking, OOD, behavioral extrapolation,
memorization — is not listed here; it lives in the design notes (`architecture_note.md`,
`in_distribution_scenarios.md`, `notes.md`).

## 1. LLM candidate sets are not shuffled (position/label confound)

`CappedUserInteractionCandidateSampler.sample` emits each group's rows positives-first,
then negatives (`src/beyond_click_sim/tasks/samplers.py:230-252`). That order is preserved
through the builder and runner, and `LLMInteractionYesNoScorer.score` labels candidates
`C1..Cn` in row order (`src/beyond_click_sim/scorers/llm.py:147-157`). So in **every**
candidate group the LLM sees the true positives as `C1..Ck`, followed by the negatives.

This is not answer leakage (the model is never told the labels), but candidate position
becomes perfectly correlated with the label. LLMs have well-documented primacy/position
selection bias, so the measured alignment partly reflects presentation order rather than
user-simulation ability — most plausibly inflating the LLM numbers. The popularity
baseline is unaffected (it scores by item, ignoring order), so the LLM-vs-popularity
comparison is biased too.

**Fix:** shuffle candidates within each group using a fixed per-group seed before building
the prompt. Optionally add a position-bias diagnostic (permute order, measure score drift).

## 2. `MAX_LLM_ERRORS` aborts the whole run instead of skipping

`_score_groups` stops the entire scoring loop once the error count reaches `MAX_LLM_ERRORS`
(= 3): `runners/in_distribution/interaction_prediction/methods/llm_yes_no.py:37,271,293-296`.
Errored groups are already skipped and counted (their scores stay NaN and are excluded from
metrics); only the abort is wrong. On a full run (thousands of groups) a model will exceed 3
malformed/parse-failed outputs early, and the run aborts, discarding all work.

**Fix:** make the budget large or a fraction of the total group count, and skip-and-continue
instead of aborting. Keep reporting scored-vs-requested coverage.

## 3. Per-user candidate items are not guaranteed disjoint across splits (val vs test negatives)

Negatives exclude all observed items, so they never collide with any positive, and positives
are already item-disjoint across splits (one row per user-item in both datasets). The
remaining gap: val and test negatives come from independent `sampler.sample(...)` calls that
share the same per-user group id, so the same unobserved item can be drawn as a negative in
both val and test — i.e. the same `(user, item, target=0)` instance appears in two splits.

Changing the seed only makes the collision unlikely, not impossible (two independent draws
from the same pool can still overlap). A correct fix coordinates negative sampling across
splits: draw val and test negatives jointly per user, or pass val negatives as an exclusion
set when sampling test negatives, so per-user candidate items are guaranteed disjoint across
train/val/test.

Currently low-impact (val is not yet used to select anything for the LLM, and the popularity
threshold is a single scalar), but worth fixing for a clean evaluation protocol.
