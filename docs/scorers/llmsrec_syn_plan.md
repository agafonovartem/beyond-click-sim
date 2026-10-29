# LLMSRec-Syn implementation plan

Updated 2026-10-29. Status: architecture implemented; evaluation updates applied. The specification below is the accepted
plan; exact construction decisions, usage and validation evidence are recorded in
[the implementation description](llmsrec_syn_design.md).

Reproduce LLMSRec-Syn's architecture with the minimum changes needed for this
project's interaction-prediction evaluation: retrieve similar training users,
aggregate their histories and next items into one synthetic demonstration, and
ask a frozen LLM to rank the supplied target candidates. Evaluate that ranking
directly and through the existing validation-selected binary threshold.

## Sources and reproduction scope

The reference is *The Whole is Better than the Sum: Using Aggregated
Demonstrations in In-Context Learning for Sequential Recommendation*, supplied
arXiv:2403.10135v1, 15 March 2024. Use section 4, Figure 10 and the appendix
prompts, together with the authors' implementation:

- [Local paper](../../../papers/The%20Whole%20is%20Better%20than%20the%20Sum%20Using%20Aggregated%20Demonstrations%20in%20In-Context%20Learning%20for%20Sequential%20Recommendation.pdf)
- [Paper reading report](../../../docs/paper_readings/aggregated-demonstrations/reports/aggregated-demonstrations.md)
- [Authors' repository](https://github.com/demoleiwang/LLMSRec_Syn)

Before implementation, inspect and pin the actual `RankAggregated` code and its
configuration. Resolve history serialization for embeddings, duplicate items,
short histories, candidate sampling, prompt order and generation settings against
that source. The supplied manuscript takes precedence where author code conflicts
with its method description. Record each conflict and deliberate departure.

The source review found that the repository reverses the retrieved users' next-item
list, although section 4 places the most similar user's next item first. It also
reverses the aggregate history, uses `floor(50/K)` aggregation rounds, and specifies
`max_his_len: 10` in its model configuration. Reconcile these against the manuscript
and Figure 10 instead of silently copying them. Source evidence:
[aggregation code](https://github.com/demoleiwang/LLMSRec_Syn/blob/main/model/rankaggregated.py)
and [configuration](https://github.com/demoleiwang/LLMSRec_Syn/blob/main/props/RankAggregated.yaml).
These links are mutable; pin the revision before implementation.

The core method uses a frozen LLM; it does not require task-specific fine-tuning.
Implement the synthetic-demonstration architecture directly. Separate nearest-user
baselines, self-history ICL and LLMRank's bootstrap ensemble are outside this scope.
This is an architectural reproduction under the project's evaluation protocol;
the changed datasets/protocol/model settings do not establish reproduction of
the paper's published scores.

## Intended paper claim

The agreed method-description claim is:

> We reimplemented LLMSRec-Syn's retrieval and synthetic-demonstration architecture
> and adapted it to our interaction-prediction protocol, converting candidate
> rankings into binary predictions using a validation-selected threshold.

The paper description must also report the adaptations below, including the
target-specific temporal restriction. Implementation and verification supporting
this claim are recorded in the linked implementation description.

The 2026-10-29 evaluation update uses NDCG@5 for K selection separately in
each seed, adds Recall@5 to ranking summaries, and starts the study with ML-1M.
The initial shared 1,000-user cohort will be screened for deterministic data and
retrieval eligibility before scoring; it must not be selected by observed model
success or accuracy. Predicted-positive-fraction analysis is deferred. The
existing valid-only threshold selection coincides with the paper's strict rule
when all groups succeed; failures after retries require reconciliation before
strict protocol results are reported.

## Chronological data and demonstration pool

Support MovieLens and Steam through the same chronological pipeline. The user
will add Steam timestamps from review dates upstream. Steam execution depends
on those timestamps becoming available; do not substitute arbitrary interaction
order. Record that this represents review chronology, and use the upstream rules
for missing dates, repeated reviews and timestamp ties.

Preserve the project's temporal training, validation and test partitions. For
target user u, define `cutoff(u) = max(timestamp of u's training interactions)`.
Derive it from the complete training history, before applying the prompt history
budget. Use this same cutoff and training history when scoring validation and
test; do not advance it with validation outcomes or candidate timestamps.

For each other training user v, construct a target-specific eligible sequence:

`eligible(v, u) = events in v's training split with timestamp <= cutoff(u)`.

Apply these operations in order:

1. Intersect v's original training membership with the timestamp condition. Do
   not admit v's validation or test events, even if their dates precede the cutoff.
2. Preserve chronological order with the existing stable ordering for timestamp
   ties. Reserve the final event of this truncated sequence as its demonstration
   next-item label; the preceding events form its demonstration history.
3. Require at least two eligible events, so the history is nonempty. Exclude the
   target user. Build the pool from all other eligible training users, including
   those outside the evaluated-user subset.
4. Apply history budgets and embed the resulting prefix for retrieval. Both the
   history and its reserved next-item label must satisfy the cutoff.

For example, if u's training cutoff is day 10 and v's training events occur on
days 3, 6 and 15, v contributes history `[day 3]` and next-item label `day 6`.
The day-15 event cannot influence retrieval or the constructed demonstration.

Keep demonstration prefixes separate from full target histories. A user's
demonstration prefix and label can differ for different target cutoffs, so one
global final-training-event label per demonstration user is insufficient.
Validation and test outcomes must never enter retrieval, demonstrations or prompts.

The inclusive `<=` rule means no event is later than the target cutoff at the
recorded timestamp resolution. Equal timestamps can span existing partitions;
retain original split membership and document that their within-timestamp order
does not establish strict real-world precedence. Missing or unverifiable
timestamps must produce a clear error, not an arbitrary-order fallback.

This restriction closes the future-interaction path through demonstration users
and retrieval. It does not turn the shared benchmark into a fully historical
deployment simulation: catalog metadata, pretrained model knowledge and the
existing retrospective evaluation-candidate construction have their own temporal
assumptions. Do not claim those are solved by truncating interaction histories.

## Similar user retrieval

Embed the target-specific truncated demonstration prefixes and the target user's
training history, then retrieve the top K eligible users by cosine similarity.
Truncate before computing embeddings and selecting neighbors. Retrieving against
full-history embeddings and only truncating the displayed example would still
allow future events to influence selection. The paper's reference
embedding model is `text-embedding-ada-002`; make the backend configurable and
record any substitution. Cache embeddings using the training-split identity,
embedding model identity, serialization version and exact truncated input.
Key neighbor lists and constructed demonstrations additionally by target cutoff,
eligible-event provenance and sampling configuration. Reuse identical prefixes
across cutoffs where valid; never substitute a full-history embedding to save calls.

Make K configurable. The paper explores K from 1 through 7. Any selection of K
uses validation data, with the chosen configuration frozen before test evaluation.
Define deterministic tie-breaking. If fewer than K users qualify after temporal
filtering, report the affected group through explicit failure/coverage diagnostics;
do not silently add future events, self-examples or a zero-shot fallback.

## Synthetic demonstration construction

1. Order retrieved users by decreasing similarity to the target user.
2. Aggregate recent history items round-robin: the latest item from each user,
   then the second-latest from each, continuing to the aggregate history limit.
   Preserve and explicitly document the source method's presentation order.
3. Build a synthetic candidate set containing retrieved users' reserved next
   items plus randomly sampled catalog fillers, for 20 candidates in total.
4. Construct the synthetic answer with known next items first in retrieved-user
   similarity order, followed by randomly ordered remaining candidates.

Use the manuscript as the construction specification and the pinned author code
to clarify compatible edge cases. Document duplicate handling, candidate exclusions,
short histories and presentation order; preserve repeated history events where
the source requires them. Do not silently deduplicate history as a convenience.
Sampling and tie-breaking must be reproducible independently of worker scheduling
and retries. Random fillers use the permitted catalog; any interaction-derived
sampling rule must use only training events at or before the target cutoff.
Never use validation/test outcomes to screen demonstration candidates.

Use title-only inputs with a 50-item budget for the target history and a 50-item
aggregate-history budget in the initial reproduction configuration. This follows
the user's priority of staying as close to the paper as the evaluation permits.
Do not add genres, historical ratings/playtime, summaries or item statistics to
this configuration. Reuse existing formatting utilities only where they preserve
the intended paper inputs and prompts.

## LLM prompt and ranking score

Provide one completed synthetic example containing history, candidates and its
answer, followed by the target user's training history and actual task candidates.
Retain the paper's ranking objective. The proposed parsing adaptation is to use
candidate labels such as `C1`, `C2` and `C3` consistently in both the demonstration
and target request instead of matching generated item titles.
Preserve the paper's instruction template (A) and demonstration template (T3)
except for the explicit label-output contract, actual candidate count and domain
wording needed for Steam. Do not inherit an unrelated simulator/persona prompt
merely because the listwise scorer already supplies one.

Request one complete target ranking from a frozen LLM. Use one successful
generation per candidate group, with existing retries for request or parsing
failures. A valid but inaccurate ranking must never trigger a retry. Do not add
LLMRank candidate permutations or bootstrap aggregation.

Reuse strict ranking parsing and map a candidate at one-based rank r to
`score = n_candidates - r`, preserving the original task-row index. These are
ordinal scores, not interaction probabilities.

## Interaction prediction evaluation

Keep the existing task's candidate groups, positives, negatives and split
membership unchanged. Support multiple positives and variable target-group
sizes even though the synthetic demonstration contains 20 candidates. Validation
and test scoring both use training-only history and the same per-user cutoff.

Reuse the listwise evaluator for ranking metrics and the validation-selected
binary threshold. Retain existing coverage, retry and failure-aware reporting.
This adapter changes how LLMSRec-Syn is evaluated without changing its output
into per-candidate yes/no responses.

Select K and any other tuned configuration through a separate validation-only
scoring path independently for each repetition. Select K by validation NDCG@5
averaged within user across groups,
then across users, using the same fixed validation users, groups and candidate
rows for every K. Select any validation subset once, before the search.

Assign zero NDCG credit to every failed group, including groups with insufficient
eligible neighbors or exhausted request/parsing retries. Keep those groups in the
averaging population. Reuse `ranking_metrics_with_failed_groups_as_zero` and its
`macro_by_user_group_mean` result in the new runner without changing the helper.
Do not select K using only successful groups, drop failed users, or replace
missing item scores with zero: the last option would award random-tie ranking
credit instead of zero group utility. Keep retry budgets identical across K.
Resolve equal selection scores deterministically by choosing the smaller K.

Freeze K, models, metadata and prompts before invoking the existing combined
validation/test evaluator. That evaluator already scores both splits, so calling
it repeatedly during K selection would unnecessarily expose test results. The
binary threshold uses validation F1 under the evaluator's existing user/group
averaging and is fixed for test predictions. Neither retry decisions nor prompt
construction may depend on held-out labels.

Retain the existing interpretation of sampled unobserved items as negative
labels. They are not confirmed rejections after exposure; this limitation is
documented rather than changing the common experimental setup.

Accepted limitation, to disclose explicitly in the paper's method description
and discussion: rank thresholding cannot estimate absolute interaction willingness.
With a fixed threshold,
every fully parsed group of the same size receives the same number of positive
predictions. Raw rank scores also depend on group size. They do not express
absolute interaction confidence. For example, with `score = n_candidates - rank`
and threshold 15, a 20-item group receives five positive predictions, while a
10-item group receives none. The LLM can change which items receive a positive
prediction, but not how many within equally sized groups at that threshold.
Retain this adapter for the implementation and report direct ranking metrics
alongside binary metrics; do not present its scores as calibrated probabilities
or change the shared score mapping to address this limitation.

## Behavioral isolation

The initial acceptance checks below apply to the isolated architecture addition.
The subsequently authorized Recall@5 change adds metric fields to shared ranking
helpers; it must preserve all existing scores, predictions and metric values.


Behavioral isolation is an acceptance criterion against a recorded snapshot of
the current workspace, including existing uncommitted changes. It is not inferred
from adding a new class. Existing splitter changes and the user's upstream Steam
timestamp work must be recorded separately from this implementation.

- Add dedicated scorer, retrieval, prompt, cache, timestamp-mapping and runner
  modules. Reuse existing helpers without changing their behavior or defaults.
- Keep existing scorer/prompt/history-selection/parser/evaluation function bodies
  and task construction unchanged. Shared-file edits are limited to additive
  exports/imports and new method registry entries; preserve existing entries and
  default experiment selections.
- Handle the broader demonstration pool inside the new scorer. Preserve existing
  methods' evaluated-user filtering and stored history representations.
- Do not mutate shared Task frames, client configuration, prompt dictionaries or
  global random state. Use local random generators and separate output/cache paths.
- Keep embedding initialization and calls specific to the new method. Importing
  existing scorers must not require new credentials, trigger network calls or
  eagerly load an embedding model.
- Compare existing methods under fixed clients before and after the addition:
  request payloads, scores, thresholds and metric/prediction artifacts must match,
  apart from explicitly variable provenance such as elapsed time or run paths.

If implementation discovers a need to change a shared function's behavior, that
change falls outside this additive plan; use a method-local adapter or identify
the required scope change before proceeding.

## Implementation structure and diagnostics

Add a dedicated `LLMSRecSynScorer` and an interaction-prediction runner. Reuse the
existing LLM clients, metadata formatters, ranking parser and listwise evaluator.
Add focused retrieval, embedding-cache and demonstration-construction components
as needed, within the behavioral-isolation boundary.

Current materialized task training rows omit timestamps and canonical interaction
IDs. Add a method-specific mapping from task training rows to event timestamps
and identities, passed separately to the scorer; timestamps are control metadata,
not prompt features. The new runner may reconstruct the original filtered temporal
training split from the matching canonical snapshot and recorded filter/splitter
configuration using existing components. Verify its projection exactly matches
the supplied task's training rows, including event multiplicity and stable order,
before attaching metadata. A `(user_id, item_id)` join alone is insufficient for
repeated interactions. Record source fingerprints, event identities/occurrences
and mapping checks; reject stale, ambiguous or unsupported sources. Do not alter
shared task schemas or rebuild/replace evaluation candidates to obtain timestamps.

Save enough diagnostics to inspect and reconstruct each example: selected users,
similarities, history and reserved-next-item provenance, aggregate item order,
candidate identities and labels, demonstration answer, prompt version, random
seed and response/parsing outcome. Include the target cutoff, eligible-event
timestamps, prefix fingerprints and excluded-user counts/reasons so the temporal
restriction can be audited. Treat these as local row-level artifacts.

Record dataset and temporal split, timestamp semantics, candidate construction,
LLM and embedding model settings, visible metadata, history limits, K, seed,
source revision, prompt version, score mapping, validation threshold, coverage
and observed request/token usage in run metadata. Include the inclusive cutoff
rule, timestamp-mapping provenance and validation-only configuration-selection
procedure. Save the fixed validation-group identity/fingerprint and, for each K,
the failure-aware selection score, requested/successful/failed group counts,
failure reasons and final selection/tie-breaking rule. Report retrieval/embedding
cost as well as LLM generation cost.

## Verification and acceptance

Use small deterministic fixtures and fake LLM/embedding clients to verify:

- Training-only demonstration provenance, self-exclusion and separation of
  demonstration prefixes from full target histories.
- Both demonstration history and its reserved label satisfy the target cutoff.
  The label is selected after truncation, not taken from the full user sequence.
- Retrieval uses the truncated prefix. Modifying an excluded future suffix while
  preserving task membership and keeping it beyond the cutoff cannot change
  embeddings, selected users or the constructed demonstration.
- A demonstration user can have different prefixes and labels at two target
  cutoffs. Cache reuse cannot cross incompatible prefixes or eligibility sets.
- No validation/test events enter the pool even when their timestamps precede
  the cutoff. Validation and test requests keep the target's training cutoff.
- Timestamp ties, repeated items, short eligible sequences, insufficient
  neighbors, missing timestamps and mismatched canonical/task provenance.
- Similarity ordering, retrieval ties and round-robin aggregation.
- Candidate/answer consistency, duplicate handling and short histories.
- Reproducible sampling, cache invalidation and original-row score mapping.
- Strict parsing, retries and failure reporting without held-out-label access.
- Integration with ranking metrics and validation-only binary thresholds.
- Configuration selection cannot read test outcomes; changing test labels cannot
  change K or the threshold. No validation/test feedback enters scorer inputs.
- Every K is evaluated on identical validation groups, with failures retained as
  zero group NDCG in the same user/group averaging population. Verify that a
  configuration with a better successful-only score can lose after failures are
  included, and that missing scores never receive random-tie credit. Check the
  deterministic smaller-K tie-break.
- Existing methods retain their request/score/evaluation behavior on fixed
  fixtures, and adding/running the new scorer leaves shared task data and global
  random state unchanged.

Run relevant existing scorer, task-construction and evaluation regression tests.
Inspect a saved constructed prompt before any live experiment. Implementation
validation does not require launching paid LLM or embedding experiments.

## Adaptations to report

The initial configuration preserves the paper's titles and 50-item history
budgets. Remaining differences from the published experiment are explicit:

- The project's temporal fractional split, held-out-period interaction target,
  multi-positive groups, variable target candidate counts and negative sampler.
- Demonstration labels drawn from truncated training sequences, with a separate
  cutoff for each target user applied before retrieval and aggregation. This is
  an additional temporal restriction; the authors' released code uses examples
  supplied through `valid_data` instead.
- Candidate-label output, strict parsing/retries and downstream binary threshold
  selection, while retaining the LLM's ranking task. Explicitly disclose the
  adapter's inability to estimate absolute interaction willingness, fixed positive
  counts within equally sized groups and dependence on candidate-group size.
- K selection on a fixed validation population with failed groups assigned zero
  NDCG credit, using the project's user/group averaging.
- Steam review chronology, any substituted LLM/embedding model or title
  serialization, and differences in tuning, sample sizes and repetition budgets.
- Explicit resolution of manuscript/code conflicts and underspecified edge cases.

Comparisons against existing differently configured baselines compare complete
methods under the shared evaluation setup. They do not isolate the contribution
of synthetic demonstrations. A matched-input ablation is separate experimental
work, not a prerequisite or an added variant in this implementation plan.
