# LLMRank: design and method specification

Implemented on 2026-10-27; combined-output verification and documentation updated
on 2026-10-29. This document describes the recency-focused, three-ranking method
and its controlled adaptation to this project. One ensemble supplies both ranking
scores and thresholded classification labels; no separate classification or
ranking run is needed. Validation uses fake clients, not live model experiments.

## 1. Sources and the original method

**Primary manuscript:** Hou et al., *Large Language Models are Zero-Shot Rankers
for Recommender Systems*, supplied arXiv:2305.08845v2 (24 January 2024).
[Local PDF](../../../papers/Large%20Language%20Models%20are%20Zero-Shot%20Rankers%20for%20Recommender%20Systems.pdf),
[arXiv](https://arxiv.org/abs/2305.08845v2).

- Section 2.2 describes ranking externally supplied candidates using a frozen
  instruction-following LLM, chronological history, recency-focused prompting,
  and an alternative self-history prefix/successor demonstration.
- Section 3.2 describes bootstrapping: randomly reorder the candidates in each
  of three ranking rounds, then merge their rankings to reduce position bias.
  The manuscript does not give a precise merge formula.
- Section 3.3's final configuration uses five recent history items, the explicit
  latest-item reminder, and three ranking rounds. Self-history ICL is an
  alternative experiment, not a component of this final configuration.
- The paper normally uses GPT-3.5-Turbo with temperature 0.2. Its initial
  evaluation ranks 20 candidates including the held-out next item; the final
  multi-generator experiment reports 21 candidates without inserting that item
  (sections 3 and 3.3, Tables 2 and 4).

**Author implementation evidence:** saved files from
[RUCAIBox/LLMRank](https://github.com/RUCAIBox/LLMRank), retrieved 2026-10-25 from
the default branch. No immutable commit was returned, and correspondence with
the published implementation is unverified:

- [rank.py snapshot](../../../docs/short_paper_selection/architecture_implementation_2026_10_25/sources/hou_zero_shot/llmrank__model__rank.py.md):
  source lines 153-155 sum round scores; lines 233-261 parse titles and assign
  descending positional scores. Lines 128-149 use the ground-truth item when
  deciding some retries. We do **not** reproduce that label-dependent behavior.
- [rf.py snapshot](../../../docs/short_paper_selection/architecture_implementation_2026_10_25/sources/hou_zero_shot/llmrank__model__rf.py.md):
  source lines 10-27 implement the explicit latest-item reminder.
- The saved rank.py tiles candidate rows and shuffles their transpose (lines
  82-87). This can couple the permutations across rounds. We implement the
  manuscript's intended independent draws instead of copying that operation.

The author code additionally requests step-by-step reasoning and heuristically
matches item titles. Our output contract suppresses explanations and uses the
project's strict candidate-label parser. These are explicit adaptations.

## 2. Scope and comparison

The implementation retains the final recency-plus-bootstrap architecture. There are no separate
sequential, self-history ICL, retrieval, fine-tuning, persona, or simulation-loop
variants. The reusable scorer accepts configured item metadata, but its initial
runner supports only MovieLens interaction prediction. Steam's current canonical
ownership data has no interaction timestamps and cannot justify recency claims.

| Property | Existing yes/no | Existing listwise | LLMRank adaptation |
|---|---|---|---|
| Frozen LLM | Yes | Yes | Yes |
| Training | Store history | Store history | Store history |
| Output | Binary answer per candidate | One complete order | Three complete orders |
| Score | 0 or 1 | m - rank | Sum of three m - rank scores |
| Latest-item reminder | No | No | Yes, existing H label |
| Candidate permutations | Task order | Task order | Three independent draws |
| Ranking ties | Many binary ties | None within valid order | Possible after aggregation |
| Binary evaluation | Direct answers | Validation threshold | Validation threshold |

For a controlled comparison with the plain yes/no and listwise baselines, use the
**same materialized temporal Task**, LLM
backend/model/revision, metadata columns and values, 20 most recent training
items, temperature 0, generation cap 512, backend options, candidate-group limit,
and retry limit. Baseline prompts differ by their intended output task. LLMRank
uses three successful calls per group versus one for the baselines; this extra
inference budget is reported, not described as equal compute.

Task construction remains the project's 70/10/20 per-user temporal fractional
split, existing multi-positive groups, negative sampling and candidate-size cap.
It is not the paper's leave-one-out evaluation. Rebuild the Task and rerun all
methods; do not reuse scores from older random-split runs. Item statistics, when
enabled, are computed from training rows and exposed identically to those matched
baselines. This is not a description of every method in the wider comparison:
LLMSRec-Syn uses title-only inputs and a 50-item history budget. Comparing that
configuration with this LLMRank configuration evaluates the configured methods,
not architecture alone.

The current registered LLMRank entries use Qwen3-8B or Qwen3.6-27B, a 20-item
history budget, temperature 0, and richer metadata. MovieLens history rows include
titles, genres and the user's historical ratings; candidates include titles and
genres. The `with_item_stats` wrappers append training-derived mean ratings and
review counts to both. These are deliberate departures from the paper's final
five-item, title-only, GPT-3.5-Turbo configuration at temperature 0.2. Candidate
feedback from validation or test is never a prompt input. The generic runner can
disable item statistics, but still includes genres and historical ratings.

## 3. Architecture and interfaces

- `scorers/history/llmrank.py` defines `LLMRankScorer`, a subclass of
  `LLMInteractionListwiseRankingScorer`, publicly exported by both scorer packages.
- Inherit `fit(X, y, history_user_ids=None)` and the shared history-window and
  metadata formatters. Targets are not used to fit LLM weights. Training rows
  must already be ordered chronologically within each user.
- `score(X) -> pd.Series` returns float aggregate scores on the original row
  index, named `score`. Require one user per non-null candidate group, unique
  non-null item IDs within a group, and an unambiguous input index. Empty input
  returns an empty Series. Unknown users and empty selected histories fail before
  a request. There is no silent cold-start/recency fallback.
- The constructor retains parent client/model/formatting options and adds
  `bootstrap_seed=0` and `item_column="item_id"`. Defaults are history 20,
  temperature 0 and max_tokens 512. Exactly three rounds are implemented. Only
  the normal system/user prompt style is supported; the neutral IID prompt
  family is not a second LLMRank variant.
- `drain_diagnostics()` returns and clears completed attempt records safely after
  a scoring phase. Diagnostics are separate from the `score` return contract.
- `interaction_prediction/methods/llmrank.py` configures the scorer, checks temporal
  task provenance, and reuses the listwise evaluator and whole-group retry helper.
  Smoke/full wrappers match the existing LiteLLM Qwen3-8B and Qwen3.6-27B models.
- The shared listwise evaluator accepts optional score-mapping and after-split
  callback arguments. Defaults preserve existing listwise artifacts and behavior.

The temporal runner requires a MovieLens manifest with
`splitter.class="RandomFractionSplitter"`, `splitter.shuffle=false`, a timestamp
column name, and grouping by user. The splitter already sorts stably; feature
joins preserve row order. History selection takes the last N training rows.
Tied timestamps follow the existing stable input-order rule and can straddle
partitions; do not claim strict timestamp inequality or a unique latest event.
No timestamp column needs to become visible metadata in model prompts.

## 4. Exact prompt contract

Version: `llmrank_recency_bootstrap_v1`.

System message:

```text
You are simulating a user's relative preferences in a recommender system.
Given the user's chronological interaction history and candidate items, rank all candidates from most likely to least likely to be interacted with next.
Use only the provided history and candidate information.
The history is context only: never include history items in the ranking.
Do not explain or show your reasoning.
Return only the candidate labels in ranked order, one label per line.
```

User message (placeholders are filled without exposing targets or item IDs):

```text
User interaction history, in chronological order (oldest to newest):
{history}

Note that the user's most recent interaction in this history is {recent_history_label}.

Candidate items:
{candidates}

Rank every candidate from most likely to least likely to be interacted with next by this user.

Required output format:
The complete set of allowed candidate labels is:
{output_labels}
Return exactly one ranked line for each candidate label.
There are exactly {candidate_count} candidates, so your response must contain exactly {candidate_count} non-empty lines.
Use each candidate label exactly once.
Before returning, silently verify that every allowed label appears exactly once. If a label is missing or duplicated, correct the list.
Do not output item names or history items.
Do not include rank numbers, punctuation, explanations, or analysis.
The first character of your response must be C.
```

For a 20-item history the reminder refers to `H20`; a shorter history uses its
actual final H label. It adds no new fields or a 21st history item. Parent
formatting preserves separate history/candidate columns and JSON-list handling.

## 5. Permutations, mapping and scores

Construct a SHA-256 request key from bootstrap seed, typed user/group identifiers
and ordered candidate item IDs. Seed a local random generator with this digest;
draw three permutations sequentially. Never use global random state or Python
`hash()`. Identical input and seed give identical permutations independently of
thread scheduling and retries. Independent draws need not be distinct.

Each round relabels the permuted candidates C1 through Cm. The inverse mapping
restores original row positions before summing scores. Duplicate titles are safe
because parsing uses positional labels, not text matching. Return scores only
after all three complete rankings have parsed successfully.

For one-based rank r_b(i), use:

`score(i) = sum(n_candidates - rank_b(i) for b in 1..3)`.

The author's first positional score is m rather than m-1; subtracting one per
round shifts every complete aggregate by three and does not change its order.
Our convention matches the existing listwise scorer. These scores are ordinal,
not calibrated probabilities. Keep aggregate ties and use average-tie metrics.

Worked example, m=3:

| Round | Ranking in original candidate identities | A score | B score | C score |
|---|---|---:|---:|---:|
| 1 | B, A, C | 1 | 2 | 0 |
| 2 | A, C, B | 2 | 0 | 1 |
| 3 | B, C, A | 0 | 2 | 1 |
| Sum | B > A > C | 3 | 4 | 2 |

A cyclic set of orders A,B,C / B,C,A / C,A,B instead gives a three-way tie at 3.
Candidate-group membership never changes; permutations only change presentation.

## 6. Failures, retries and diagnostics

Reuse the five-attempt whole-group retry helper. Any request failure or invalid
ranking invalidates that entire ensemble attempt. Stop the attempt immediately;
the next attempt redraws the same permutations and recomputes all rounds. Never
reuse partial results. At most 15 scorer-issued requests are made per group,
excluding any SDK/provider retries. An exhausted group retains missing scores.
The existing evaluator reports parsed-only coverage and failure-aware metrics.

Retry decisions never depend on targets, relevance or ranking quality. A complete
but inaccurate ranking is a successful request. The model sees neither validation
nor test labels. Validation selects the binary threshold only after scoring.

Save `llmrank_rounds.jsonl` in each run directory. One record per group attempt
contains split, stable request key, user/group IDs, attempt number, each attempted
round's permutation and positional-to-item mapping, parsed labels/original item
order, response text, elapsed time, available token usage, errors, and ensemble
completion. Completed records are collected under a lock; no network requests
are serialized by that lock. Sort records deterministically before writing.
Local diagnostics are row-level artifacts and must not be committed as results.

The manifest records prompt version/source, model/backend settings, metadata,
history budget, three-round sum mapping, seed/seed algorithm, temporal splitter,
whole-ensemble retry rule, theoretical call bound and observed requests/usage.
Missing usage is marked unavailable rather than reported as zero. Bootstrap
diagnostics are flushed after each split, including fully failed splits. Record
paper and saved-code provenance plus hashes of relevant current source files;
the commit alone cannot describe a working tree with uncommitted splitter edits.

## 7. Evaluation and matched invocation

The runner uses `evaluate_listwise_scorer` to evaluate the aggregate scores
directly. NDCG@5 and Recall@5 are the principal ranking measures; cutoffs 1, 3,
and 10 and hit-rate remain available as diagnostics. Metrics average groups
within each user, then users; group-only averages are also saved. Aggregate ties
are retained and metrics use their expected value over tied orders.

The same aggregate scores supply classification: validation F1 selects one
threshold, independently within each run, and `prediction = (score >= threshold)`.
Do not replace the aggregate scores with ranks of the final ordering or with the
thresholded labels when computing ranking metrics. Neither classification nor
metric calculation makes additional LLM calls. With no retries, one validation
group and one test group require six calls total. There is no LLMSRec-Syn-style
neighbor-count K or configuration search in this runner; three bootstrap rounds
are fixed. Scoring uses the same training-only history for validation and test;
validation events do not extend the test history.

| Artifact | Combined-output contract |
| --- | --- |
| `predictions.parquet` | Original candidate fields and identifiers, `split`, reference `target`, aggregate `score`, and nullable binary `prediction`. |
| `metrics_ranking.json` | Direct NDCG, recall and hit-rate; tie diagnostics; successful-only and failed-group-zero metrics; coverage and the pointwise threshold. |
| `metrics.json` | Classification metrics for the same outputs, validation-selected threshold, successful-only and failed-row-negative metrics, and coverage. |
| `manifest.json` | Configuration, prompt/source provenance, both decision rules and threshold-selection details. |
| `llmrank_rounds.jsonl` | Per-attempt round mappings, generated rankings and aggregate scores; sufficient to audit candidate remapping. |
| `llmrank_request_summary.json` | Requests, retries, available token usage and elapsed request time by split. |
| `llm_errors.jsonl` | Terminal group failures by split after the retry budget is exhausted. |

Scores are ordinal, not calibrated interaction probabilities. Unlike thresholding
one complete ranking, thresholding the three-round sum need not give the same
positive count in equally sized groups: disagreement and ties can change that
count. This still does not establish absolute interaction willingness. Sampled
unobserved negatives are not verified rejections.

Failure behavior is unchanged. Exhausted groups retain missing scores and labels;
failure-inclusive ranking assigns zero utility to the group, and strict binary
metrics treat missing decisions as negative. Threshold fitting uses successfully
scored validation groups. The ranking `main_metric` points to the successful-only
`test` field; reporting the paper's strict ranking result when failures remain
requires `test_failure_as_zero_group.macro_by_user_group_mean`. If validation or
test is entirely unsuccessful, the evaluator raises after flushing both phases'
diagnostics, before writing final prediction/metric files. These cases need
reconciliation before strict protocol results are reported; this documentation
update does not change the evaluator.

Four registry entries are available:
`llmrank_litellm_qwen3_8b_with_item_stats_smoke/full` and
`llmrank_litellm_qwen36_27b_with_item_stats_smoke/full` (each suffix is a separate
method). Smoke limits each split to 25 groups; full uses all task groups. The
respective wrapper concurrency limits are 64 and 32. Both disable Qwen thinking,
use a 512-token response budget, and allow five whole-ensemble attempts per group.

To produce both evaluation views in one run, with live calls only when executed:

```python
from pathlib import Path
from runners.in_distribution.interaction_prediction.methods import METHOD_RUNNERS
from runners.in_distribution.interaction_prediction.task_builders import build_alignment_task

task = build_alignment_task(
    "ml-1m", negative_ratio=3, seed=0,
    max_eval_users=1000, max_candidate_groups_per_user=5,
    use_item_rating_stats=True,
)
method = "llmrank_litellm_qwen36_27b_with_item_stats_smoke"
METHOD_RUNNERS[method](task, Path("outputs/llmrank_combined_example"))
```

Use a fresh output directory; the current runner does not protect existing result
files against overwrite. Cohort eligibility screening is separate experiment
preparation: `max_eval_users=1000` alone does not establish the shared eligible
cohort described in the paper.

The following matched invocation is illustrative and performs live calls only
when explicitly run. Build once; all three generic runners accept these shared
settings. Bootstrap seed is additionally explicit for LLMRank.

```python
from pathlib import Path
from runners.in_distribution.interaction_prediction.methods import (
    llm_yes_no, llm_listwise_ranking, llmrank,
)
from runners.in_distribution.interaction_prediction.task_builders import build_alignment_task

task = build_alignment_task(
    "ml-1m", negative_ratio=3, seed=0,
    max_eval_users=1000, max_candidate_groups_per_user=5,
    use_item_rating_stats=True,
)
shared = dict(
    client_name="litellm_local", model="Qwen/Qwen3.6-27B",
    max_candidate_groups=25, max_history_items=20, temperature=0.0,
    max_tokens=512, max_llm_attempts=5, max_workers=1,
    use_item_stats=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for name, runner in [
    ("yes_no", llm_yes_no.run_method),
    ("listwise", llm_listwise_ranking.run_method),
    ("llmrank", llmrank.run_method),
]:
    runner(task, Path("outputs/llmrank_comparison") / name,
           method_name=f"matched_{name}", **shared)
```

Use a fresh output directory for each actual comparison. Verify manifests have
matching model/options/columns/history limits and identical test candidate rows.
LLMRank's bootstrap seed defaults to 0. The registered smoke/full wrappers are
conveniences; the matched generic invocation overrides the yes/no runner's
otherwise smaller token limit. No live experiments are launched by tests.

## 8. Validation and acceptance

Use fake clients to test chronological newest-item emphasis, fixed history,
metadata parity, target exclusion, original-index preservation, variable-size
groups, independent permutation mapping, exact sums/ties, deterministic serial
and parallel execution, and repeated permutations on retries. Exercise missing,
unknown and duplicate labels, empty responses, client failures, unknown users,
empty histories, exhausted attempts and the 15-call bound. Test threshold
selection against changed test labels, both metric artifacts, truthful score
mapping, provenance, round diagnostics and unavailable token usage. Include a
small fake-client three-method comparison using one temporally built Task.

Run existing splitter, task-construction, yes/no and listwise suites unchanged,
then relevant scorer/registry regression tests. Preserve unrelated uncommitted
splitter, notebook and test changes. Do not update final_results.md or claim
recommendation-quality improvements from fake-client tests.

## 9. Paper-ready method description

We implement a controlled adaptation of the recency-focused LLMRank method of
Hou et al. A frozen Qwen model receives the 20 most recent training interactions
in chronological order, with an explicit reminder identifying the last one.
History includes titles, genres and historical ratings; the registered variants
also expose training-derived item mean ratings and review counts in history and
candidates. Each candidate group is independently permuted three times, and a
complete candidate-label ranking is requested for each presentation. Labels are
mapped back to the original items before summing positional scores m-r across
the three rounds. Aggregate ties are retained. NDCG@5 and Recall@5 evaluate these
scores directly, while a validation-F1-selected threshold produces binary labels
from the same scores without another generation pass.

The implementation substitutes Qwen3-8B or Qwen3.6-27B at temperature 0 for the
paper's reference GPT-3.5-Turbo at 0.2, uses 20 history items rather than five,
and adds metadata beyond titles. It adapts prompts and parsing to candidate
labels and uses the shared fractional temporal split, variable-size multi-positive
candidate groups, and sampled negatives. No parameters are fine-tuned, and no
held-out labels influence generation or retries. The three-round sum follows
the inspected author-code snapshots; those snapshots are not pinned to an
immutable author commit. This is an architectural adaptation, not a reproduction
of published scores. Different metadata and history budgets from LLMSRec-Syn
prevent interpreting their comparison as an isolated architecture comparison.

## 10. Implementation validation

Completed on 2026-10-27: 177 selected tests passed, including 35 new LLMRank
tests and the existing splitter, task-builder, yes/no, listwise, scorer-contract,
cold-start, preference and saved-prediction evaluation regression tests. The new
integration test builds one temporal MovieLens-like canonical Task and runs all
three methods with fake clients, checking candidate identities, hidden feedback,
visible metadata, 20-row history windows and shared generation settings.

Ruff checks and formatting checks passed for the new Python files; the shared
evaluator also passes lint. Local document-source links resolve. No live LLM
calls or recommendation-quality experiments were performed, and no run was
added to final_results.md. Existing uncommitted temporal splitter and unrelated
notebook/test edits were preserved.

On 2026-10-29, combined-output verification passed 181 focused and regression
tests across LLMRank, LLMSRec-Syn, ranking metrics, yes/no and listwise runners,
cold-start/preference paths, scorer contracts, temporal splits and task building.
The focused LLMRank/ranking subset contains 50 passing tests. The new six-item
fixture supplies three disagreeing rankings with aggregate scores
`[14, 12, 10, 3, 3, 3]`. The bottom three items tie across positions 4--6;
with positives A and F, expected Recall@5 is 5/6. Its binary mapping would give
a different ranking metric, so the test detects accidental evaluation of binary
labels in place of aggregate scores. It also verifies the selected threshold's
saved labels and six calls for one validation plus one test group. Extended
checks verify request invariance to test labels, missing failed-group labels,
coverage, and failure-inclusive NDCG@5/Recall@5 and F1.

Ruff lint and formatting checks passed. The updated manuscript compiled
successfully. Only this document, the runner tests and the manuscript were
changed in this follow-up; source fingerprints confirm that scorer, runner,
registry and shared evaluator implementations remained unchanged. Existing
failure semantics and method defaults were preserved. No live model calls,
experiment runs, result registrations or commits were made.
