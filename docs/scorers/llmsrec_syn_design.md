# LLMSRec-Syn implementation and evaluation

Implemented on 2026-10-28; evaluation alignment updated 2026-10-29 according to the
[accepted plan](llmsrec_syn_plan.md). The scorer retrieves similar training users,
combines them into one synthetic demonstration and requests a candidate ranking
from a frozen LLM. The existing evaluator converts rank scores into binary
predictions using a validation-selected threshold. No live model experiment was
required for implementation validation.

## Sources and fidelity decisions

The primary reference is *The Whole is Better than the Sum: Using Aggregated
Demonstrations in In-Context Learning for Sequential Recommendation*, supplied
arXiv:2403.10135v1, 15 March 2024, especially section 4, Figure 10 and Table 5.
[Local manuscript](../../../papers/The%20Whole%20is%20Better%20than%20the%20Sum%20Using%20Aggregated%20Demonstrations%20in%20In-Context%20Learning%20for%20Sequential%20Recommendation.pdf).

Author repository revision pinned during implementation:
[`1272c62785d920c406d7233c772a704e960c224a`](https://github.com/demoleiwang/LLMSRec_Syn/tree/1272c62785d920c406d7233c772a704e960c224a).
This identifies the inspected code; correspondence with the code used for the
published results is not established. The manuscript takes precedence on conflicts.

| Author source at this revision | Observation and implementation decision |
| --- | --- |
| `model/rankaggregated.py` | Reverses both aggregate history and the next-item list. We retain section 4's recent-first round-robin construction order and decreasing-similarity positive-label order. Figure 10 does not specify an additional reversal. |
| `model/rankaggregated.py` | Uses `floor(50/K)` rounds and can revisit short histories through negative indexing. We stop at 50 actual events, skip exhausted prefixes and never wrap them. |
| `model/rankaggregated.py` | Uses title lists without reliably resolving shared next-item identities. We preserve repeated history events, but use unique candidate item IDs; shared next items keep their first, most-similar position. |
| `props/RankAggregated.yaml` | Specifies history 10, temperature 0, max tokens 512, and `boots: 0`. We use the manuscript's history 50 and retain the other three settings. |
| `run_test.py` and `trainer.py` | The entry point defaults to `sim="multivector"`; the trainer builds multi-hot item vectors. We implement the manuscript's frozen LLM embedding retrieval instead. |
| `model/rank.py` | Loads a separate MovieLens title field and moves trailing `, The` / `, A` to the front. Our local MovieLens formatter removes the canonical trailing release year and applies the same article handling. Steam titles remain unchanged. |
| `model/rankaggregated.py` | Reads demonstration sequences and labels from `valid_data`. We derive both from training events eligible at the target user's cutoff. |

Inspected file SHA-256 values:

```text
model/rankaggregated.py fff7af850b96f45d7464e1c0eabc03aff34ea983f7ec821e679eff9c43df606b
model/rank.py           9c47df6a682ea0095b58b898445935fbd0759f64e7716a9a4a0e79fac236e0e2
trainer.py              da00c3d97a24093f25c40b0ee12846313d37e45f30ae6ec615c84ccbbcf70d32
run_test.py             c25aaa39e6a82ab49ba58493725e0b1f45b8e5d4ff0bb51d78bc037963def985
props/RankAggregated.yaml c2f1ec208656dc80f13cec4e74eb4791b66c75b3efccacde4e67374d37dc9c2e
```

## Components and behavioral isolation

- [Scorer](../../src/beyond_click_sim/scorers/history/llmsrec_syn.py):
  `LLMSRecSynScorer`, exported by both scorer packages. `fit` stores verified
  observed training events; `score` returns positional scores on the original
  row index. Neither operation fits LLM weights.
- [Embedding cache](../../src/beyond_click_sim/scorers/history/llmsrec_syn_retrieval.py):
  OpenAI-compatible batch embedding requests, cosine normalization and a SQLite
  cache. Keys include exact input, serialization version, backend/model identity
  and training-source fingerprint. RAM retention is bounded; identical prefixes
  can reuse embeddings at different cutoffs.
- [Prompts](../../src/beyond_click_sim/scorers/history/llmsrec_syn_prompts.py):
  manuscript templates A/T3 adapted to candidate-label output and Steam's reviewed
  games. One user message contains one completed demonstration followed by the
  target request. There is no simulator-persona system prompt.
- [Timestamp reconstruction](../../runners/in_distribution/interaction_prediction/methods/_llmsrec_syn_data.py):
  a method-local helper recovers event metadata from the selected canonical
  snapshot. It reconstructs only the recorded training split and verifies its
  full task-row projection, multiplicity and order. Event identity uses canonical
  interaction IDs, or source row positions scoped by source fingerprints.
- [Runner](../../runners/in_distribution/interaction_prediction/methods/llmsrec_syn.py):
  validates input provenance, tunes K on validation only, then calls the existing
  listwise evaluator for final validation/test ranking and binary evaluation.

The initial 2026-10-28 implementation preserved shared algorithms, prompts,
splitters, samplers and evaluators; existing-file edits only added scorer exports
and six method registry entries. The user-requested 2026-10-29 follow-up adds
recall to the shared ranking metric and failure-inclusive aggregation helpers.
Existing hit-rate/NDCG values and scorer predictions are preserved. Dependencies
and default experiment selections are unchanged. No new client is constructed
at import time; model calls begin only when the new method is explicitly run.

The timestamp helper supports the existing temporal `RandomFractionSplitter`
and minimum-interaction/user-sampling filters, including their sequential
composition. It verifies the standard training-only item statistics when those
are present in a Task, but never exposes those statistics to this scorer.
Unsupported provenance or mismatched training projections fail before client
construction. Task rows do not retain original canonical timestamps/hashes, so
the selected canonical snapshot supplies those timestamps: retain its recorded
fingerprints for reproducibility. Matching row projections alone cannot prove
that a mutable source's timestamps never changed since an old Task was built.

## Exact temporal and construction rules

For target user u, let `cutoff(u)` be the maximum timestamp of the complete
training history, before limiting prompt history. For each other user v:

1. Intersect v's original training membership with `timestamp <= cutoff(u)`.
2. Preserve chronological order, using source order to resolve timestamp ties.
3. Reserve the final eligible event as the demonstration next-item label.
4. Use up to 50 preceding events as its history; require at least one such event.

Validation/test events remain excluded even when their dates precede the cutoff.
Both validation and test scoring retain the same target training cutoff. Steam
requires upstream review timestamps; no arbitrary ordering fallback is provided.
Numeric timestamps retain their source units; datetime timestamps are compared
in UTC nanoseconds. Inclusive cutoff comparisons do not establish ordering within
equal or coarsely recorded timestamps.

Embed each eligible demonstration prefix and the target's last 50 training
titles in chronological order. The exact embedding text is a JSON array of
normalized titles, preserving duplicates and order; IDs and next-item labels are
not embedded. Byte-level serialization is an explicit choice because the
manuscript does not specify it. Rank users by cosine similarity and break ties
by their typed identifier key. Self-users are removed, not assigned similarity
zero. Cache only cutoff-compatible retrieval results; different target cutoffs
can yield different prefixes, labels and selected users.

Take recent events round-robin in decreasing user-similarity order until the
aggregate reaches 50 events or all prefixes are exhausted. Preserve repeated
events. Do not reverse the constructed aggregate. The target's own history is
presented oldest to newest; the synthetic history is an aggregation, not one
person's chronological trajectory.

Build 20 distinct synthetic candidate item IDs: unique reserved next items first,
then random catalog fillers excluding those positives. Filler titles can overlap
history, as the paper does not specify a blanket history exclusion. Shared next
items keep the position of the most similar contributing user. Shuffle fillers
for the answer, then independently shuffle the candidate presentation. Local
random generators are seeded from the configured seed, target identity/cutoff
and selected prefix/label provenance. Retries and worker scheduling do not change
the constructed example. Titles shared by distinct items remain distinct labels.

The completed demonstration answer lists positive candidate labels in decreasing
user-similarity order, then randomized fillers. The target keeps its actual task
candidate membership and presentation order. Output labels C1 through Cn must
occur exactly once according to the existing ranking parser. Labels restrict
the answer to candidates, including candidates whose titles overlap history;
history-only items are never added. There is no bootstrap ensemble or quality-
dependent retry. Fewer than K eligible users produce an explicit group failure.

## Configuration selection and evaluation

The default search is K = 1 through 7. Selection is independent for each
dataset, ratio and repetition; different seeds may select different configurations.
NDCG@5 is the selection and primary reporting metric, with Recall@5 also reported.
Other existing cutoffs remain secondary diagnostics. A single-element `k_values`
runs a fixed
configuration through the same validation path. Every K receives identical
validation groups, candidates and retry budgets. Select K by NDCG@5 averaged
within user across groups, then across users. Failed groups contribute zero
NDCG and remain in that averaging population. Missing scores are not filled
with zero item scores, which would award random-tie credit. Equal selection
scores choose the smaller K. If every configuration fails, save diagnostics and
stop before accessing test rows.

After freezing K, reuse the existing combined evaluator. It re-scores validation
and then test with the chosen configuration; these additional calls are reported.
Each successful group ranking gives one-based rank r the score `n_candidates-r`.
Validation F1, using the existing user/group averaging over successfully parsed
validation groups, selects the binary threshold. The paper targets a fixed cohort
with complete outputs; if failures persist, strict metrics are already available
but failure-inclusive threshold selection still needs reconciliation. Users must
not be replaced after observing response failures or accuracy. Test labels do not
select K, the threshold, prompts or retry behavior.
Existing parsed-only, coverage and failure-aware metrics remain available.

Accepted limitation for disclosure in the paper: rank thresholding cannot estimate
absolute interaction willingness. A fixed threshold yields the same positive
count in equally sized complete groups, and its effect depends on group size.
For example, threshold 15 selects five items from a 20-item group and none from
a 10-item group. Scores are ordinal, not calibrated probabilities. Sampled
unobserved items retain the benchmark's negative-label interpretation, not
confirmed rejection after exposure.

This reproduces the retrieval/aggregation architecture under the project's
fractional temporal split, multi-positive candidate groups, negative sampling and
binary evaluation. Target-specific demonstration cutoffs, label parsing and Steam
are explicit adaptations. Backend/model substitutions and experiment repetition
budgets must also be reported. Catalog metadata, pretrained model knowledge and
retrospective evaluation-candidate construction are not made historically
available by truncating demonstration interactions.

The initial study uses ML-1M. A shared 1,000-user cohort is to be screened for
training-history, title, timestamp and sufficient-neighbor eligibility before any
model calls, then frozen across methods. No response-dependent exclusion is
allowed; data eligibility cannot guarantee API or parser success. Cohort
materialization is separate experiment preparation, not performed by this runner.
Steam remains supported for a future dated review-event source. The optional
predicted-positive-fraction diagnostic is deferred to analysis of saved rows.

## Running the method

Six registry entries are available: `llmsrec_syn_openai_smoke/full`,
`llmsrec_syn_litellm_qwen3_8b_smoke/full`, and
`llmsrec_syn_litellm_qwen36_27b_smoke/full`. These are separate concrete names,
for example `llmsrec_syn_litellm_qwen36_27b_smoke`. Smoke limits each evaluation
split to 25 groups, while keeping all eligible training users as the demonstration
pool. OpenAI wrappers use GPT-3.5-Turbo as the reference model. Qwen wrappers use
the existing LiteLLM endpoint and disable thinking through its existing option.

Embedding and generation backends are configured independently. All wrappers
default to OpenAI `text-embedding-ada-002` for retrieval, including Qwen generation
wrappers. An OpenAI embedding credential is therefore required for those defaults.
Use the generic runner for other providers or a historical model unavailable on
the current account. An explicit embedding model identity can record an immutable
deployment/revision behind an alias. Reusing a cache path is safe only with a
truthful model identity; source/input changes receive distinct cache keys.

Example invocation, which makes model calls when executed:

```python
from pathlib import Path
from runners.in_distribution.interaction_prediction.task_builders import build_alignment_task
from runners.in_distribution.interaction_prediction.methods.llmsrec_syn import run_method

task = build_alignment_task(
    "ml-1m", negative_ratio=3, seed=0,
    max_eval_users=1000, max_candidate_groups_per_user=5,
)
run_method(
    task,
    Path("outputs/llmsrec_syn_ml1m_example"),
    client_name="litellm_local",
    model="Qwen/Qwen3.6-27B",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    embedding_client_name="openai",
    embedding_model="text-embedding-ada-002",
    embedding_cache_path=Path("data/cache/llmsrec_syn_embeddings.sqlite3"),
    k_values=(1, 2, 3, 4, 5, 6, 7),
    max_candidate_groups=25,
    max_workers=32,
)
```

Use a fresh result directory. `canonical_dataset` can supply an explicit matching
`CanonicalDataset` snapshot; otherwise the runner uses the current canonical
dataset location. Existing results are not overwritten.

## Artifacts and validation

The runner writes the existing `manifest.json`, `predictions.parquet`, binary
and ranking metric files, plus:

- `llmsrec_syn_selection.json`: fixed validation population fingerprint, every K's
  failure-aware metrics/errors, selected K and tie rule.
- `llmsrec_syn_attempts.jsonl`: per-attempt cutoffs, selected users, event/prefix
  provenance, exact demonstrations/prompts, target row mappings, responses,
  parsing failures, timing and available usage.
- `llmsrec_syn_request_summary.json`: phase-specific generation counts and
  embedding/cache statistics. Unavailable usage is reported as unavailable.
- `llmsrec_syn_embeddings.sqlite3`: default local embedding cache unless a shared
  cache path is explicitly supplied.

Large row-level artifacts and embedding caches are local outputs, not committed
research results. No run was added to `final_results.md`.

Initial implementation validation on 2026-10-28 (before the metric follow-up):

- All 64 new scorer/runner tests pass, covering temporal boundaries, cache
  separation, duplicate items/titles, short histories, strict parsing/retries,
  failure-aware K selection, threshold independence from test labels, numeric
  and datetime timestamps, both datasets and source projection checks.
- The full suite reports 761 passed and three existing Agent4Rec failures. All
  three reproduce using the pre-implementation exports/registry snapshot: two
  expect cache statistics without the existing `errors` field, and one expects
  a different taste-model client configuration. Existing implementations/tests
  were preserved.
- Before/after fixed-client snapshots for existing yes/no, listwise and LLMRank
  methods have identical requests, predictions and metric artifacts. Fingerprints
  of 226 existing source/test/dependency files show changes only to the three
  additive export/registry files.
- A real MovieLens task with 694,335 training rows and 6,040 training users passed
  end-to-end with fake generation/embedding clients: two validation and two test
  groups, K in {1,2}, eight generation calls, 97 embedding batches and no scoring
  errors. The constructed prompt was inspected, including cutoff compliance,
  50 aggregate events and 20 synthetic candidates. This checks integration, not
  recommendation quality.

Evaluation follow-up validation on 2026-10-29:

- Nine additional cases cover multi-positive recall, exact expectations over tied
  orders, groups without positives, group/user averaging with failures, and a
  K-selection example where NDCG@5 and NDCG@10 prefer different configurations.
- The focused scorer/runner/metric checks report 94 passed; the full suite reports
  770 passed and the same three pre-existing Agent4Rec failures listed above.
- Fixed-client yes/no, listwise and LLMRank snapshots preserve requests,
  predictions and all prior metric values; only `recall@k` fields are added.
- No live experiment or eligibility-cohort selection was performed.

## Paper method description

We reimplemented LLMSRec-Syn's retrieval and synthetic-demonstration architecture
and adapted it to our interaction-prediction protocol, converting candidate
rankings into binary predictions using a validation-selected threshold. For each
target user, demonstration histories and their next-item labels are restricted
to training events no later than that user's training cutoff before embedding
and retrieval. We use title-only inputs, one aggregated example, and candidate-label
ranking output. K is selected on a fixed validation population using NDCG@5 with
failed groups assigned zero credit. The frozen LLM's positional scores are then
thresholded using validation F1. This adapter evaluates relative ranking under
the benchmark's candidate construction; it does not estimate absolute interaction
willingness, and equally sized groups receive a fixed number of positive
predictions at a given threshold.
